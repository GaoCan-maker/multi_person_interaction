#!/usr/bin/env python3
"""
tools/eval_with_llm.py

Comprehensive multi-person motion evaluation — four metrics in one pass:

  1. PeneBone  — bone-level penetration depth (PyBullet, radius 2 cm)
                 Lower is better (fewer collisions).

  2. APD       — Average Pairwise Distance (intra-scene action diversity)
                 Average L2 distance between per-person normalised velocity
                 vectors within the same scene. Higher is better.

  3. PED       — Pair-Embedding Diversity (InterCLIP-based)
                 For every C(N,2) pair in a scene, get the InterCLIP embedding
                 of that paired motion; compute mean L2 distance across all pair
                 embeddings.  Higher means pairs are semantically more distinct
                 (i.e. more diverse intra-scene interactions).
                 Requires *_feat.npy files saved by infer_multi.py.

  4. LLM Score — GPT-4o-mini vision judge (inspired by AToM: Aligning
                 Text-to-Motion at Event-Level with GPT-4Vision Reward).
                 NOTE: gpt-4o-mini fully supports vision input and is the
                 direct successor of gpt-4-vision-preview — same capability,
                 lower cost.  Pass --llm_model gpt-4o for maximum accuracy.
                 Three sub-scores (1-10 each):
                   • diversity     — how distinct are different persons' actions?
                   • reasonableness— physical plausibility / no severe clipping
                   • alignment     — how well motions match the scene description

Usage
-----
  python tools/eval_with_llm.py \\
      --results_dir  results/ \\
      --graph_dir    results/graphs/ \\
      --prompt_file  prompts.txt \\
      --llm_model    gpt-4o-mini \\
      --person_num   3 \\
      --out          eval_report.csv

  # Disable LLM scoring (compute PeneBone + APD + PED only):
  python tools/eval_with_llm.py --no_llm

  # Disable PeneBone (faster):
  python tools/eval_with_llm.py --no_penelope

  # Disable InterCLIP embedding metric:
  python tools/eval_with_llm.py --no_interclip

Environment
-----------
  export OPENAI_API_KEY=sk-...   (or set --llm_api_key_env to another var name)
  export LLM_BASE_URL=https://api.deepbricks.ai/v1/   (optional override)
"""

import os
import sys
import json
import argparse
import base64
import io
import re
import csv
import time
from itertools import combinations
from typing import Any, Dict, List, Optional

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")  # headless backend — must be before pyplot import
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3  # noqa: F401  (registers the 3D projection)

# ── project root on path ─────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.llm_client import openai_chat_completions, get_api_key_from_env
from eval_model.collision import CollisionDepth
from configs import get_config
from datasets.evaluator import EvaluatorModelWrapper


# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────

# SMPL-H 22-joint kinematic chains (same as collision.py / paramUtil)
KINEMATIC_CHAIN = [
    [0, 2, 5, 8, 11],
    [0, 1, 4, 7, 10],
    [0, 3, 6, 9, 12, 15],
    [9, 14, 17, 19, 21],
    [9, 13, 16, 18, 20],
]

# Distinct colours for up to 8 persons
PERSON_COLORS = ["red", "royalblue", "green", "darkorange",
                 "purple", "deeppink", "brown", "teal"]

# LLM judge prompts (AToM-style structured scoring)
_SYSTEM_PROMPT = """\
You are an expert evaluator of computer-generated multi-person motion animations.
You will be shown a strip of {n_keyframes} key frames sampled uniformly from a \
generated animation.  Each person is drawn in a distinct colour and labelled \
P1, P2, … etc.

Score the motion on EXACTLY the three dimensions below.
Return your answer as a valid JSON object with EXACTLY these keys:
  "diversity":      integer 1-10
  "reasonableness": integer 1-10
  "alignment":      integer 1-10
  "rationale":      string  (≤ 60 words)

SCORING CRITERIA
----------------
diversity (1-10)
  How visually distinct are the different persons' motions from each other?
  1  = all persons perform the same motion throughout
  5  = some variation, but motions are broadly similar
  10 = each person has a clearly distinct, unique motion pattern

reasonableness (1-10)
  Physical plausibility — appropriate spacing, no severe body intersection,
  natural body postures and natural transitions.
  1  = persons clip / intersect severely, highly unnatural postures
  5  = generally acceptable with minor artifacts
  10 = fully natural, physically plausible interaction

alignment (1-10)
  How well do the generated motions match the provided scene description?
  1  = motions completely ignore the described scenario
  5  = partial match — some actions fit, others do not
  10 = motions precisely and coherently enact the described scenario

Output JSON ONLY — no markdown fences, no extra text.\
"""

_USER_TEMPLATE = """\
Scene description:
"{prompt}"

Number of persons: {n_persons}
The {n_keyframes} frames below are sampled at uniform intervals \
(left = start, right = end of the animation).

Please evaluate and return JSON.\
"""


# ─────────────────────────────────────────────────────────────────────────────
#  1. PeneBone  — bone-level collision depth
# ─────────────────────────────────────────────────────────────────────────────

def compute_penebone(joints_all: np.ndarray) -> float:
    """
    joints_all: (person_num, T, 22, 3)
    Returns mean per-frame per-pair collision depth (metres).
    Lower is better.
    """
    person_num, T = joints_all.shape[:2]
    total_depth = 0.0
    num_pairs = 0
    for i, j in combinations(range(person_num), 2):
        detector = CollisionDepth(joints_all[i], joints_all[j])
        total_depth += abs(detector.check_depth())
        num_pairs += 1
    return total_depth / (num_pairs * T) if num_pairs > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  2. APD  — Average Pairwise Distance (intra-scene action diversity)
# ─────────────────────────────────────────────────────────────────────────────

def compute_apd(joints_all: np.ndarray) -> float:
    """
    joints_all: (person_num, T, 22, 3)
    Intra-scene diversity: mean pairwise L2 distance of per-person normalised
    velocity feature vectors.  Higher is better (more diverse actions).
    """
    person_num = joints_all.shape[0]
    if person_num < 2:
        return 0.0

    vel = np.diff(joints_all, axis=1)             # (P, T-1, 22, 3)
    feats = vel.reshape(person_num, -1).astype(np.float64)  # (P, D)
    norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
    feats_norm = feats / norms

    dists = [
        float(np.linalg.norm(feats_norm[i] - feats_norm[j]))
        for i, j in combinations(range(person_num), 2)
    ]
    return float(np.mean(dists))


# ─────────────────────────────────────────────────────────────────────────────
#  3. PED  — Pair-Embedding Diversity (InterCLIP-based)
# ─────────────────────────────────────────────────────────────────────────────

def compute_ped(feat_all: np.ndarray, eval_wrapper: "EvaluatorModelWrapper",
                device: torch.device) -> float:
    """
    Pair-Embedding Diversity using the trained InterCLIP evaluator model.

    For each C(N,2) pair (i, j) in a scene, feed (feat_i, feat_j) through the
    InterCLIP motion encoder to obtain one semantic embedding per pair.
    PED = mean L2 distance between all pair embeddings.

    Higher PED → pair interactions are semantically more distinct
                 → people in the scene are doing more diverse things.

    Args:
        feat_all:     (person_num, T, 262) de-normalised full-feature array
                      saved by infer_multi.py as *_feat.npy.
        eval_wrapper: initialised EvaluatorModelWrapper (InterCLIP).
        device:       torch device.

    Returns:
        float PED value, or 0.0 if only one person or model error.
    """
    person_num, T, feat_dim = feat_all.shape
    if person_num < 2:
        return 0.0

    pair_embeddings: List[np.ndarray] = []

    for i, j in combinations(range(person_num), 2):
        motion1 = torch.from_numpy(feat_all[i:i + 1]).float()  # (1, T, 262)
        motion2 = torch.from_numpy(feat_all[j:j + 1]).float()  # (1, T, 262)
        motion_lens = torch.LongTensor([T])

        # EvaluatorModelWrapper.get_motion_embeddings signature:
        #   (name, text, motion1, motion2, motion_lens)
        batch_data = ("", [""], motion1, motion2, motion_lens)
        try:
            emb = eval_wrapper.get_motion_embeddings(batch_data)  # (1, D)
            pair_embeddings.append(emb.cpu().numpy().squeeze(0))   # (D,)
        except Exception as e:
            print(f"    [PED] embedding error for pair ({i},{j}): {e}")
            return float("nan")

    if len(pair_embeddings) < 2:
        return 0.0

    # Mean pairwise L2 distance between all pair embeddings
    emb_mat = np.stack(pair_embeddings, axis=0)  # (C(N,2), D)
    n = len(emb_mat)
    dists = []
    for a, b in combinations(range(n), 2):
        dists.append(float(np.linalg.norm(emb_mat[a] - emb_mat[b])))

    return float(np.mean(dists))


# ─────────────────────────────────────────────────────────────────────────────
#  4a. Key-frame rendering → PNG bytes
# ─────────────────────────────────────────────────────────────────────────────

def _render_single_frame(ax, joints_all: np.ndarray, frame_idx: int,
                         person_colors: List[str]) -> None:
    """Render all persons at one frame on a given 3D axis."""
    ax.cla()
    # Determine data range
    all_xyz = joints_all[:, frame_idx]  # (P, 22, 3)
    margin = 0.3
    xmin, xmax = all_xyz[..., 0].min() - margin, all_xyz[..., 0].max() + margin
    ymin, ymax = max(0, all_xyz[..., 1].min() - 0.1), all_xyz[..., 1].max() + margin
    zmin, zmax = all_xyz[..., 2].min() - margin, all_xyz[..., 2].max() + margin

    ax.set_xlim3d([xmin, xmax])
    ax.set_ylim3d([ymin, ymax])
    ax.set_zlim3d([zmin, zmax])
    ax.view_init(elev=110, azim=-90)
    ax.set_axis_off()
    ax.set_facecolor("white")

    for p_idx in range(len(joints_all)):
        color = person_colors[p_idx % len(person_colors)]
        pose = joints_all[p_idx, frame_idx]  # (22, 3)
        for chain in KINEMATIC_CHAIN:
            ax.plot3D(pose[chain, 0], pose[chain, 1], pose[chain, 2],
                      color=color, linewidth=2.0)
        # label near the head joint (index 15)
        head = pose[15]
        ax.text(head[0], head[1] + 0.15, head[2],
                f"P{p_idx + 1}", color=color, fontsize=8, fontweight="bold")


def render_keyframes_png(joints_all: np.ndarray, n_frames: int = 5) -> bytes:
    """
    joints_all: (person_num, T, 22, 3)
    Renders n_frames key frames side-by-side as a single PNG.
    Returns PNG bytes suitable for base-64 encoding.
    """
    T = joints_all.shape[1]

    # Normalise ground plane: shift all persons so the lowest point is at y=0
    y_min = joints_all[..., 1].min()
    joints_norm = joints_all.copy()
    joints_norm[..., 1] -= y_min

    # Uniformly spaced frame indices
    indices = [int(round(i * (T - 1) / (n_frames - 1))) for i in range(n_frames)]

    colors = PERSON_COLORS[: joints_all.shape[0]]

    fig, axes = plt.subplots(
        1, n_frames,
        figsize=(4 * n_frames, 4),
        subplot_kw={"projection": "3d"},
    )
    if n_frames == 1:
        axes = [axes]

    fig.patch.set_facecolor("white")

    for col, frame_idx in enumerate(indices):
        _render_single_frame(axes[col], joints_norm, frame_idx, colors)
        axes[col].set_title(f"frame {frame_idx}", fontsize=9, pad=2)

    plt.tight_layout(pad=0.5)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=90, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def png_to_base64(png_bytes: bytes) -> str:
    return base64.b64encode(png_bytes).decode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
#  4b. LLM-as-judge call + score parsing
# ─────────────────────────────────────────────────────────────────────────────

def _extract_scores(raw: str) -> Optional[Dict[str, Any]]:
    """Parse JSON dict from LLM response; return None on failure."""
    raw = raw.strip()
    # Strip markdown fences if present
    raw = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)

    # Find the outermost {...} block
    start = raw.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(raw)):
        if raw[i] == "{":
            depth += 1
        elif raw[i] == "}":
            depth -= 1
            if depth == 0:
                raw = raw[start: i + 1]
                break

    try:
        obj = json.loads(raw)
        for key in ("diversity", "reasonableness", "alignment"):
            obj[key] = int(obj[key])
        assert 1 <= obj["diversity"] <= 10
        assert 1 <= obj["reasonableness"] <= 10
        assert 1 <= obj["alignment"] <= 10
        return obj
    except Exception:
        return None


def llm_judge(
    png_bytes: bytes,
    prompt: str,
    n_persons: int,
    n_keyframes: int,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float = 0.1,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """
    Call a GPT-4V-compatible endpoint with key-frame images and return scores.

    Returns dict with keys: diversity, reasonableness, alignment (int 1-10),
    and rationale (str).  On repeated failure returns all -1.
    """
    b64 = png_to_base64(png_bytes)
    system_msg = _SYSTEM_PROMPT.format(n_keyframes=n_keyframes)
    user_text = _USER_TEMPLATE.format(
        prompt=prompt,
        n_persons=n_persons,
        n_keyframes=n_keyframes,
    )

    # OpenAI-compatible vision message format
    # (openai_chat_completions serialises messages as-is, so list content works)
    messages = [
        {"role": "system", "content": system_msg},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{b64}",
                        "detail": "high",
                    },
                },
            ],
        },
    ]

    for attempt in range(max_retries):
        try:
            raw = openai_chat_completions(
                base_url=base_url,
                api_key=api_key,
                model=model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=300,
                timeout=90,
            )
            scores = _extract_scores(raw)
            if scores is not None:
                return scores
            print(f"    [LLM] JSON parse failed (attempt {attempt + 1}), "
                  f"raw: {raw[:200]}")
        except Exception as e:
            print(f"    [LLM] Request error (attempt {attempt + 1}): {e}")
            time.sleep(2 * (attempt + 1))

    return {
        "diversity": -1,
        "reasonableness": -1,
        "alignment": -1,
        "rationale": "error",
    }


# ─────────────────────────────────────────────────────────────────────────────
#  5. Prompt recovery helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_graph_json(graph_dir: str, safe_name: str) -> Optional[Dict]:
    """
    Load results/graphs/{safe_name}_graph.json to recover the original prompt
    and inter_graph metadata saved during inference.
    """
    path = os.path.join(graph_dir, f"{safe_name}_graph.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def safe_name_from_npy(fname: str) -> str:
    """
    'Three_people_are_fighting_3p.npy'  →  'Three_people_are_fighting'
    Strips the trailing _Np suffix.
    """
    base = fname[: -len(".npy")]
    return re.sub(r"_\d+p$", "", base)


def load_prompts(prompt_file: str) -> Dict[int, str]:
    """Return {1-based line number: prompt text}."""
    result = {}
    with open(prompt_file, encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                result[idx] = line
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  6. Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate multi-person motion: PeneBone + APD + PED + LLM Judge"
    )
    parser.add_argument("--results_dir", default="results/",
                        help="Directory containing *_Np.npy result files")
    parser.add_argument("--graph_dir", default="results/graphs/",
                        help="Directory containing *_graph.json files "
                             "(written by infer_multi.py; used to recover prompts)")
    parser.add_argument("--prompt_file", default="prompts.txt",
                        help="Fallback prompt source if graph JSON is unavailable")
    parser.add_argument("--llm_base_url",
                        default=os.environ.get("LLM_BASE_URL",
                                               "https://api.deepbricks.ai/v1/"),
                        help="OpenAI-compatible Chat Completions base URL")
    parser.add_argument("--llm_api_key_env", default="OPENAI_API_KEY",
                        help="Environment variable holding the API key")
    parser.add_argument("--llm_model", default="gpt-4o-mini",
                        help="Vision-capable model name (e.g. gpt-4o, gpt-4o-mini). "
                             "gpt-4o-mini supports vision and is the cost-efficient "
                             "successor to gpt-4-vision-preview.")
    parser.add_argument("--person_num", type=int, default=None,
                        help="Only evaluate N-person scenes; omit to evaluate all")
    parser.add_argument("--n_keyframes", type=int, default=5,
                        help="Number of key frames to sample for visual LLM scoring")
    parser.add_argument("--no_penelope", action="store_true",
                        help="Skip PeneBone (much faster; only APD + PED + LLM)")
    parser.add_argument("--no_llm", action="store_true",
                        help="Skip LLM scoring (only PeneBone + APD + PED)")
    # ── InterCLIP / PED args ──────────────────────────────────────────────────
    parser.add_argument("--eval_model_cfg", default="configs/eval_model.yaml",
                        help="Path to InterCLIP evaluator config "
                             "(default: configs/eval_model.yaml)")
    parser.add_argument("--eval_model_ckpt", default="eval_model/interclip.ckpt",
                        help="Path to InterCLIP checkpoint "
                             "(default: eval_model/interclip.ckpt)")
    parser.add_argument("--no_interclip", action="store_true",
                        help="Skip InterCLIP Pair-Embedding Diversity (PED) metric")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Torch device for InterCLIP inference (default: auto)")
    parser.add_argument("--out", default="eval_report.csv",
                        help="Output CSV file path")
    args = parser.parse_args()

    device = torch.device(args.device)

    # ── InterCLIP evaluator setup ─────────────────────────────────────────────
    # Note: build_models() inside EvaluatorModelWrapper loads the checkpoint
    # from the hardcoded path 'eval_model/interclip.ckpt' relative to cwd.
    # Make sure to run this script from the project root, or pass --no_interclip.
    eval_wrapper: Optional[EvaluatorModelWrapper] = None
    if not args.no_interclip:
        cfg_path = os.path.join(PROJECT_ROOT, args.eval_model_cfg)
        ckpt_path = os.path.join(PROJECT_ROOT, args.eval_model_ckpt)
        if not os.path.isfile(cfg_path):
            print(f"Warning: eval_model_cfg not found at {cfg_path}. "
                  "PED metric disabled. Pass --no_interclip to suppress.")
            args.no_interclip = True
        elif not os.path.isfile(ckpt_path):
            print(f"Warning: eval_model_ckpt not found at {ckpt_path}. "
                  "PED metric disabled.")
            args.no_interclip = True
        else:
            # Ensure cwd is project root so build_models finds 'eval_model/interclip.ckpt'
            _orig_cwd = os.getcwd()
            os.chdir(PROJECT_ROOT)
            try:
                eval_cfg = get_config(cfg_path)
                eval_wrapper = EvaluatorModelWrapper(eval_cfg, device)
                print(f"Loaded InterCLIP evaluator  ({ckpt_path})")
            except Exception as e:
                print(f"Warning: Failed to load InterCLIP evaluator: {e}\n"
                      "PED metric disabled.")
                args.no_interclip = True
            finally:
                os.chdir(_orig_cwd)

    # ── LLM setup ────────────────────────────────────────────────────────────
    api_key: Optional[str] = None
    if not args.no_llm:
        try:
            api_key = get_api_key_from_env(args.llm_api_key_env)
        except RuntimeError as e:
            print(f"Warning: {e}\nLLM scoring disabled.")
            args.no_llm = True

    # ── Fallback prompt map (line-number based) ───────────────────────────────
    prompt_map: Dict[int, str] = {}
    if os.path.isfile(args.prompt_file):
        prompt_map = load_prompts(args.prompt_file)
        print(f"Loaded {len(prompt_map)} prompts from {args.prompt_file}")

    # ── Find result .npy files ────────────────────────────────────────────────
    npy_files = sorted(
        f for f in os.listdir(args.results_dir)
        if f.endswith(".npy") and re.search(r"_\d+p\.npy$", f)
    )
    if args.person_num is not None:
        npy_files = [f for f in npy_files
                     if f.endswith(f"{args.person_num}p.npy")]

    if not npy_files:
        print("No *_Np.npy result files found in", args.results_dir)
        return

    print(f"\nEvaluating {len(npy_files)} scene(s) …\n")

    rows: List[Dict] = []
    penebone_list: List[float] = []
    apd_list:      List[float] = []
    ped_list:      List[float] = []
    div_list:      List[int]   = []
    reas_list:     List[int]   = []
    align_list:    List[int]   = []

    for fname in npy_files:
        fpath = os.path.join(args.results_dir, fname)
        joints = np.load(fpath)  # expected: (person_num, T, 22, 3)

        if joints.ndim != 4 or joints.shape[2] != 22 or joints.shape[3] != 3:
            print(f"  ⚠ Skip {fname} — unexpected shape {joints.shape}")
            continue

        person_num, T = joints.shape[:2]

        # ── Recover original prompt ───────────────────────────────────────────
        sname = safe_name_from_npy(fname)
        graph_data = load_graph_json(args.graph_dir, sname)
        if graph_data and "prompt" in graph_data:
            prompt_text = graph_data["prompt"]
        else:
            # Last-resort: match by safe_name against all prompts in prompt_map
            prompt_text = ""
            for p in prompt_map.values():
                import re as _re
                candidate = _re.sub(r"[^\w\-\. ]+", "_", p.strip(),
                                    flags=_re.UNICODE).strip().replace(" ", "_")
                if candidate[:48] == sname[:48]:
                    prompt_text = p
                    break

        print(f"▶ {fname}  ({person_num}p, T={T})")
        if prompt_text:
            short = prompt_text[:80] + ("…" if len(prompt_text) > 80 else "")
            print(f"  Prompt: {short}")

        # ── PeneBone ──────────────────────────────────────────────────────────
        if args.no_penelope:
            penebone = float("nan")
        else:
            print("  Computing PeneBone …")
            penebone = compute_penebone(joints)
            penebone_list.append(penebone)
            print(f"  PeneBone        = {penebone:.6f} m (↓ lower = less collision)")

        # ── APD ───────────────────────────────────────────────────────────────
        apd = compute_apd(joints)
        apd_list.append(apd)
        print(f"  APD             = {apd:.4f} (↑ higher = more diverse actions)")

        # ── PED (InterCLIP Pair-Embedding Diversity) ───────────────────────────
        ped = float("nan")
        if not args.no_interclip and eval_wrapper is not None:
            # Requires *_feat.npy (262-dim full features) saved by infer_multi.py
            feat_fname = fname.replace(f"_{person_num}p.npy",
                                       f"_{person_num}p_feat.npy")
            feat_path = os.path.join(args.results_dir, feat_fname)
            if os.path.isfile(feat_path):
                print("  Computing PED (InterCLIP) …")
                feat_all = np.load(feat_path)  # (P, T, 262)
                ped = compute_ped(feat_all, eval_wrapper, device)
                ped_list.append(ped)
                print(f"  PED (embed div) = {ped:.4f} (↑ higher = more diverse pairs)")
            else:
                print(f"  PED: skipped — {feat_fname} not found "
                      f"(run infer_multi.py to generate *_feat.npy)")

        # ── LLM Judge ─────────────────────────────────────────────────────────
        llm_scores: Dict[str, Any] = {
            "diversity": float("nan"),
            "reasonableness": float("nan"),
            "alignment": float("nan"),
            "rationale": "skipped",
        }

        if not args.no_llm:
            print(f"  Rendering {args.n_keyframes} key frames …")
            png_bytes = render_keyframes_png(joints, n_frames=args.n_keyframes)
            print("  Calling LLM judge …")
            llm_scores = llm_judge(
                png_bytes=png_bytes,
                prompt=prompt_text,
                n_persons=person_num,
                n_keyframes=args.n_keyframes,
                base_url=args.llm_base_url,
                api_key=api_key,      # type: ignore[arg-type]
                model=args.llm_model,
            )
            d  = llm_scores["diversity"]
            r  = llm_scores["reasonableness"]
            a  = llm_scores["alignment"]
            rat = llm_scores.get("rationale", "")

            if d != -1:
                div_list.append(d)
                reas_list.append(r)
                align_list.append(a)

            print(f"  LLM diversity   = {d}/10")
            print(f"  LLM reasonable  = {r}/10")
            print(f"  LLM alignment   = {a}/10")
            print(f"  LLM rationale   : {rat}")

        print()

        rows.append({
            "file":               fname,
            "person_num":         person_num,
            "frames":             T,
            "prompt":             prompt_text,
            "penebone":           penebone,
            "apd":                apd,
            "ped":                ped,
            "llm_diversity":      llm_scores["diversity"],
            "llm_reasonableness": llm_scores["reasonableness"],
            "llm_alignment":      llm_scores["alignment"],
            "llm_rationale":      llm_scores.get("rationale", ""),
        })

    # ── Aggregate summary ─────────────────────────────────────────────────────
    print("=" * 64)
    print(f"Scenes evaluated   : {len(rows)}")
    if penebone_list:
        print(f"PeneBone  (mean)   : {np.mean(penebone_list):.6f} m   "
              f"↓ lower  = fewer collisions")
    if apd_list:
        print(f"APD       (mean)   : {np.mean(apd_list):.4f}     "
              f"↑ higher = more diverse actions")
    if ped_list:
        print(f"PED       (mean)   : {np.mean(ped_list):.4f}     "
              f"↑ higher = more distinct pair interactions")
    if div_list:
        print(f"LLM Diversity      : {np.mean(div_list):.2f} / 10")
    if reas_list:
        print(f"LLM Reasonableness : {np.mean(reas_list):.2f} / 10")
    if align_list:
        print(f"LLM Alignment      : {np.mean(align_list):.2f} / 10")
    print("=" * 64)

    # ── Write CSV ─────────────────────────────────────────────────────────────
    fieldnames = [
        "file", "person_num", "frames", "prompt",
        "penebone", "apd", "ped",
        "llm_diversity", "llm_reasonableness", "llm_alignment", "llm_rationale",
    ]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nReport saved → {args.out}")


if __name__ == "__main__":
    main()

