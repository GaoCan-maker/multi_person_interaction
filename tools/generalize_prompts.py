#!/usr/bin/env python3
"""
Script to generalize two-person interaction descriptions to multi-person scenarios.
Reads test.txt, processes each annot file, and generates generalized prompts.
"""

import os
import sys
import re
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.llm_client import get_api_key_from_env, openai_chat_completions
from configs import get_config


def read_annot_file(annot_path: str) -> str:
    """Read and combine all lines from an annot file, trying multiple encodings."""
    if not os.path.exists(annot_path):
        print(f"Warning: {annot_path} does not exist, skipping.")
        return None
    
    # Try multiple encodings
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'gbk', 'gb2312']
    
    for encoding in encodings:
        try:
            with open(annot_path, 'r', encoding=encoding, errors='replace') as f:
                lines = [line.strip() for line in f if line.strip()]
            
            if not lines:
                print(f"Warning: {annot_path} is empty, skipping.")
                return None
            
            # Combine all descriptions with newlines
            return "\n".join(lines)
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Warning: Error reading {annot_path} with {encoding}: {e}")
            continue
    
    print(f"Warning: Could not read {annot_path} with any encoding, skipping.")
    return None


def generalize_to_multi_person(two_person_desc: str, cfg) -> str:
    """Use LLM to generalize a two-person interaction to a multi-person scenario."""
    api_key = get_api_key_from_env(cfg.LLM.API_KEY_ENV)
    
    system = (
        "You are a helpful assistant that generalizes two-person interaction descriptions "
        "to multi-person scenarios while keeping the core actions unchanged. "
        "You MUST always specify the exact number of people (e.g., '3 people', '4 people') "
        "and NEVER use vague terms like 'several', 'multiple', 'a group of', or 'many'."
    )
    
    user = f"""Given this two-person interaction description:
{two_person_desc}

Please generalize it to a multi-person scenario where:
1. The core actions/interactions remain the same
2. You MUST specify the exact number of people (e.g., "3 people", "4 people", "5 people")
3. DO NOT use vague terms like "several people", "multiple people", "a group of people", "many people"
4. Choose an appropriate number (typically 3-5 people) that makes sense for the interaction
5. The description should be concise (one sentence, under 30 words)

Examples of good outputs:
- "Three people greet each other by shaking hands."
- "Four people are fighting."
- "Five people are dancing together."

Examples of BAD outputs (DO NOT use):
- "Several people greet each other..."
- "Multiple people are fighting..."
- "A group of people..."

Return ONLY the generalized description with explicit number, no additional text or explanation."""
    
    try:
        content = openai_chat_completions(
            base_url=cfg.LLM.BASE_URL,
            api_key=api_key,
            model=cfg.LLM.MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            temperature=cfg.LLM.TEMPERATURE,
            max_tokens=200,  # Shorter for single sentence
            timeout=cfg.LLM.TIMEOUT,
        )
        # Clean up the response - remove quotes if present, strip whitespace
        content = content.strip().strip('"').strip("'")
        # Remove any leading/trailing punctuation that might be from formatting
        if content.startswith('"') and content.endswith('"'):
            content = content[1:-1]
        if content.startswith("'") and content.endswith("'"):
            content = content[1:-1]
        content = content.strip()
        
        # Verify that the content contains an explicit number (3-10 people)
        # Check for patterns like "3 people", "four people", "5 persons", etc.
        number_pattern = r'\b(three|four|five|six|seven|eight|nine|ten|\d+)\s+(people|persons|individuals)\b'
        vague_patterns = [
            r'\b(several|multiple|many|a group of|some|a few)\s+(people|persons|individuals)\b',
            r'\bpeople\b.*\b(several|multiple|many|group)\b',
        ]
        
        # Check for vague terms
        has_vague = any(re.search(pattern, content, re.IGNORECASE) for pattern in vague_patterns)
        has_explicit_number = re.search(number_pattern, content, re.IGNORECASE)
        
        if has_vague or not has_explicit_number:
            print(f"  ⚠️  Warning: Generated description lacks explicit number or contains vague terms: {content}")
            print(f"     This will be skipped. Please check the LLM response.")
            return None
        
        return content
    except Exception as e:
        print(f"  Error calling LLM API: {e}")
        return None


def main():
    # Load config
    infer_cfg = get_config("configs/infer.yaml")
    
    # Paths
    test_file = "data/interhuman_processed/test.txt"
    annots_dir = "data/interhuman_processed/annots"
    output_file = "prompts.txt"
    
    # Read test.txt
    if not os.path.exists(test_file):
        print(f"Error: {test_file} not found!")
        return
    
    with open(test_file, 'r', encoding='utf-8') as f:
        test_ids = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(test_ids)} test IDs to process.")
    
    # Load existing prompts count for resume
    existing_prompt_count = 0
    start_index = 0
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8', errors='ignore') as f:
            existing_prompt_count = sum(1 for line in f if line.strip())
        if existing_prompt_count > 0:
            print(f"Found existing {output_file} with {existing_prompt_count} prompts.")
            # Optionally skip already processed IDs (if count matches)
            if existing_prompt_count < len(test_ids):
                start_index = existing_prompt_count
                print(f"Resuming from index {start_index + 1} (skipping first {start_index} IDs).")
            else:
                print(f"All prompts already generated! Exiting.")
                return
        print(f"Will append new prompts to existing file.")
    
    # Open output file in append mode (will create if doesn't exist)
    output_f = open(output_file, 'a', encoding='utf-8')
    
    # Process each ID
    generated_count = 0
    failed_count = 0
    skipped_count = 0
    
    try:
        for idx, test_id in enumerate(test_ids[start_index:], start_index + 1):
            print(f"\n[{idx}/{len(test_ids)}] Processing ID: {test_id}")
            
            annot_path = os.path.join(annots_dir, f"{test_id}.txt")
            two_person_desc = read_annot_file(annot_path)
            
            if two_person_desc is None:
                skipped_count += 1
                continue
            
            print(f"  Original: {two_person_desc[:80]}...")
            
            # Generalize to multi-person
            generalized = generalize_to_multi_person(two_person_desc, infer_cfg)
            
            if generalized is None or not generalized.strip():
                print(f"  ❌ Failed to generate prompt")
                failed_count += 1
                continue
            
            print(f"  ✅ Generated: {generalized}")
            
            # Immediately write to file
            output_f.write(generalized + "\n")
            output_f.flush()  # Force write to disk immediately
            generated_count += 1
            
            # Small delay to avoid rate limiting (optional)
            time.sleep(0.1)
    
    finally:
        output_f.close()
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  ✅ Successfully generated: {generated_count} new prompts")
    print(f"  ❌ Failed (API errors): {failed_count}")
    print(f"  ⏭️  Skipped (missing/empty files): {skipped_count}")
    print(f"  📝 Total processed: {len(test_ids)}")
    print(f"  📄 Total prompts in {output_file}: {existing_prompt_count + generated_count}")
    print(f"✅ Done! All prompts saved to {output_file}")


if __name__ == "__main__":
    main()

