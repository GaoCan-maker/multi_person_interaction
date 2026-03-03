import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from utils.llm_client import get_api_key_from_env, openai_chat_completions


_JSON_OBJ_RE = re.compile(r"\{[\s\S]*\}")


@dataclass
class LLMGraphConfig:
    enabled: bool
    base_url: str
    api_key_env: str
    model: str
    temperature: float = 0.2
    max_tokens: int = 800
    timeout: int = 60
    cache_file: Optional[str] = None


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", s)
        s = re.sub(r"\n```$", "", s)
    return s.strip()


def _extract_json_obj(s: str) -> str:
    s = _strip_code_fences(s)
    
    # 方法1: 通过平衡大括号找到完整的 JSON 对象（更准确）
    json_str = None
    brace_count = 0
    start_idx = -1
    
    for i, char in enumerate(s):
        if char == '{':
            if start_idx == -1:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                json_str = s[start_idx:i+1]
                break
    
    # 方法2: 如果方法1失败，使用正则表达式作为后备
    if json_str is None:
        m = _JSON_OBJ_RE.search(s)
        if m:
            json_str = m.group(0)
    
    if json_str is None:
        raise ValueError(f"No JSON object found in LLM output. Content preview: {s[:500]}")
    
    return json_str


def _validate_graph(obj: Dict[str, Any]) -> Dict[str, Any]:
    n = int(obj.get("n_person"))
    g_in = obj.get("in")
    g_out = obj.get("out", None)
    person_text = obj.get("person_text")
    pair_dists = obj.get("pair_dists", None)

    if n < 2:
        raise ValueError("n_person must be >= 2")
    if not (isinstance(g_in, list) and len(g_in) == n):
        raise ValueError("'in' must be a list of length n_person")
    # 'out' is optional; if missing/invalid we will derive it from 'in' to keep the graph consistent.
    if g_out is not None and not (isinstance(g_out, list) and len(g_out) == n):
        g_out = None
    if not (isinstance(person_text, list) and len(person_text) == n):
        raise ValueError("'person_text' must be a list of length n_person")

    for j in range(n):
        # Allow empty in[j] (a "root" node). The sampling loop will fall back to a dummy motion-condition.
        if not isinstance(g_in[j], list):
            raise ValueError(f"in[{j}] must be a list")
        for idx in g_in[j]:
            if not isinstance(idx, int):
                raise ValueError("All indices in 'in' must be int")
            if idx < 0 or idx >= n:
                raise ValueError("Index out of range in 'in'")
            if idx == j:
                # Self-conditioning makes the current code behave oddly (b = imgs[j]).
                # Disallow by default to keep semantics clear.
                raise ValueError("Self edges are not allowed (idx == j)")
        if not isinstance(person_text[j], str) or not person_text[j].strip():
            raise ValueError(f"person_text[{j}] must be a non-empty string")

    def _normalize_pair_dists(
        raw: Any, n_: int, g_in_: List[List[int]]
    ) -> Optional[List[List[List[float]]]]:
        """
        Normalize to n x n matrix where pair_dists[i][j] is [d_min, d_max] for i!=j.
        Diagonal entries are [0.0, 0.0].
        If `raw` is missing/invalid, return a conservative fallback derived from graph connectivity.
        """
        if isinstance(raw, list) and len(raw) == n_:
            out_pd: List[List[List[float]]] = []
            ok = True
            for i in range(n_):
                row = raw[i]
                if not (isinstance(row, list) and len(row) == n_):
                    ok = False
                    break
                out_row: List[List[float]] = []
                for j in range(n_):
                    if i == j:
                        out_row.append([0.0, 0.0])
                        continue
                    cell = row[j]
                    if not (isinstance(cell, list) and len(cell) == 2):
                        ok = False
                        break
                    d_min, d_max = cell[0], cell[1]
                    try:
                        d_min = float(d_min)
                        d_max = float(d_max)
                    except Exception:
                        ok = False
                        break
                    if d_min < 0 or d_max < 0 or d_min > d_max:
                        ok = False
                        break
                    out_row.append([d_min, d_max])
                if not ok:
                    break
                out_pd.append(out_row)
            if ok:
                return out_pd

        # Fallback: if LLM output misses pair ranges, derive simple priors.
        # Interacting pairs are allowed to be closer than unrelated pairs.
        related = [[False for _ in range(n_)] for _ in range(n_)]
        for tgt in range(n_):
            for src in g_in_[tgt]:
                if src == tgt:
                    continue
                related[src][tgt] = True
                related[tgt][src] = True
        out_pd = []
        for i in range(n_):
            row = []
            for j in range(n_):
                if i == j:
                    row.append([0.0, 0.0])
                elif related[i][j]:
                    row.append([0.3, 1.6])
                else:
                    row.append([0.8, 2.5])
            out_pd.append(row)
        return out_pd

    def _derive_out_from_in(g_in_: List[List[int]], n_: int) -> List[List[int]]:
        out_ = [[] for _ in range(n_)]
        for tgt in range(n_):
            for src in g_in_[tgt]:
                if src == tgt:
                    continue
                out_[src].append(tgt)
        # dedup + stable sort
        out_ = [sorted(set(xs)) for xs in out_]
        return out_

    # Validate/normalize OUT if provided; otherwise derive from IN.
    if g_out is None:
        out_fixed = _derive_out_from_in(g_in, n)
    else:
        out_fixed: List[List[int]] = []
        for j in range(n):
            if not isinstance(g_out[j], list):
                out_fixed.append([])
                continue
            xs = []
            for idx in g_out[j]:
                if not isinstance(idx, int):
                    continue
                if idx < 0 or idx >= n or idx == j:
                    continue
                xs.append(idx)
            out_fixed.append(sorted(set(xs)))

        # Enforce basic consistency: for every edge src->tgt in IN, make sure tgt is in OUT[src].
        derived = _derive_out_from_in(g_in, n)
        for src in range(n):
            out_fixed[src] = sorted(set(out_fixed[src]).union(set(derived[src])))

    pair_dists_fixed = _normalize_pair_dists(pair_dists, n, g_in)
    return {"n_person": n, "in": g_in, "out": out_fixed, "person_text": person_text, "pair_dists": pair_dists_fixed}


def _cache_lookup(cache_file: str, prompt: str) -> Optional[Dict[str, Any]]:
    if not cache_file or not os.path.exists(cache_file):
        return None
    with open(cache_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("prompt") == prompt and "graph" in obj:
                return obj["graph"]
    return None


def _cache_append(cache_file: str, prompt: str, graph: Dict[str, Any]) -> None:
    if not cache_file:
        return
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, "a", encoding="utf-8") as f:
        f.write(json.dumps({"prompt": prompt, "graph": graph}, ensure_ascii=False) + "\n")


def generate_interaction_graph(prompt: str, cfg: LLMGraphConfig) -> Dict[str, Any]:
    """
    Return:
      {
        "in": List[List[int]],
        "out": List[List[int]],
        "person_text": List[str],              # [j] = action description for person j
        "pair_dists": List[List[List[float]]],  # [i][j] = [d_min, d_max]
      }
    where person_text[j] describes what person j does in the scene.
    """
    if cfg.cache_file:
        cached = _cache_lookup(cfg.cache_file, prompt)
        if cached is not None:
            return _validate_graph(cached)

    api_key = get_api_key_from_env(cfg.api_key_env)

    system = (
        "You generate interaction graphs for multi-person motion generation. "
        "Return JSON only; no code fences, no extra text."
    )
    user = f"""
Given this scene description:
{prompt}

Create an interaction graph in the following JSON schema:
{{
  "n_person": <int, >=2>,
  "in": <list[list[int]] length n_person>,
  "out": <list[list[int]] length n_person>,
  "person_text": <list[str] length n_person>,
  "pair_dists": <list[list[list[float,float]]], shape n_person x n_person>
}}

Rules:
- Indices are 0..n_person-1.
- Semantics:
  - in[j] lists who directly INFLUENCES / CONDITIONS person j (directed edges i -> j).
  - out[j] lists who person j directly INFLUENCES (directed edges j -> k).
  - These are NOT required to be identical; they describe opposite directions.
- in[j] must not include j itself. out[j] must not include j itself.
- You may use an empty list for in[j] if person j does not depend on any other person.
- Consistency requirement: for every edge i -> j that appears in in[j], j must also appear in out[i].
- person_text[j] is ONE sentence describing what person j does in the entire scene.
  This sentence will be used as the text condition when generating person j's motion.
  Guidelines for person_text:
    - Describe person j's concrete physical actions (body movement, posture, direction, speed).
    - You may reference other persons by index (e.g. "reaches toward Person 0") to preserve relational context.
    - Keep it concise (1-2 sentences), motion-focused, and avoid abstract or meta language.
    - Each person must have a DISTINCT, UNIQUE action description to maximize motion diversity.
- pair_dists requirements:
  - pair_dists[i][j] gives a reasonable root-distance range [d_min, d_max] between person i and person j for this scene.
  - Use meters. Ensure 0 <= d_min <= d_max.
  - For i == j, use [0.0, 0.0].
  - The matrix should be symmetric: pair_dists[i][j] == pair_dists[j][i].
  - Provide ranges for ALL person pairs, not only interacting edges.
"""
    # Some OpenAI-compatible servers don't support response_format; try with it first, then retry.
    try:
        content = openai_chat_completions(
            base_url=cfg.base_url,
            api_key=api_key,
            model=cfg.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user.strip()}],
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            timeout=cfg.timeout,
            extra_body={"response_format": {"type": "json_object"}},
        )
    except Exception:
        content = openai_chat_completions(
            base_url=cfg.base_url,
            api_key=api_key,
            model=cfg.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user.strip()}],
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            timeout=cfg.timeout,
            extra_body=None,
        )

    # 提取并解析 JSON，添加详细的错误处理
    try:
        extracted_json = _extract_json_obj(content)
        obj = json.loads(extracted_json)
    except (ValueError, json.JSONDecodeError) as e:
        print(f"Error parsing JSON from LLM response:")
        print(f"Raw content (first 1000 chars):\n{content[:1000]}")
        try:
            extracted = _extract_json_obj(content)
            print(f"Extracted JSON (first 1000 chars):\n{extracted[:1000]}")
        except Exception as e2:
            print(f"Failed to extract JSON: {e2}")
        raise ValueError(f"Failed to parse JSON from LLM output: {e}") from e
    
    graph = _validate_graph(obj)
    if cfg.cache_file:
        _cache_append(cfg.cache_file, prompt, graph)
    return graph


