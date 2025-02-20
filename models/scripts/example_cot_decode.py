#!/usr/bin/env python
# example_cot_decode.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.
#
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import os
import random
import json
import sys
import time
import gc
import re
from pathlib import Path
from typing import Optional

import fire
from datasets import load_dataset, Dataset

# Set environment variables for distributed initialization in a single-process setup.
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"

PAGE_SIZE = 100

# Define the system prompt enforcing XML answer format.
SYSTEM_PROMPT = """ 
"You are a helpful assistant. 
First, come up with a step by step reasoning process to answer the question. Make sure you understand all the facts of the question.
Second, provide the answer to the question. The answer is always an integer, float or one of the provided options. Only provide the integer, floating point or option answer in the
<answer>{number}</answer> tags.
The <answer>(answer here)</answer> must be in a separate new line with no extra spaces before or after the tags.
Example 1:
Betty had 10 apples. ... 
<answer>10</answer>
Example 2:
John went to school. ... 
<answer>10.5</answer>
Example 3:
Ben studied for 9 hours. ... 
<answer>odd</answer>
"""

def extract_hash_answer(text: str) -> Optional[str]:
    # For GSM8K, sometimes the answer string is preceded by "####"
    if "####" not in text:
        return text.strip()
    return text.split("####")[1].strip()

def get_gsm8k_questions(split="train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split]  # type: ignore
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    })  # type: ignore
    return data  # type: ignore

# Load GSM8K test dataset and shuffle with a fixed random seed.
gsm8k_dataset = get_gsm8k_questions("test")
random.seed(42)
all_examples = list(gsm8k_dataset)
random.shuffle(all_examples)

def get_page(examples, page_num, page_size=PAGE_SIZE):
    start = (page_num - 1) * page_size
    end = start + page_size
    return examples[start:end]

default_page = 1
page_examples = get_page(all_examples, default_page, PAGE_SIZE)
print("Using page number:", default_page, "with", len(page_examples), "problems.")

# Add the model package to the Python path.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from llama_models.datatypes import RawMessage, StopReason
from llama_models.llama3.reference_impl.generation import Llama

# Helper: Extract answer (content inside <answer> tags) using regex.
ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
def extract_answer(text: str) -> Optional[str]:
    match = ANSWER_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return None

# Helper: Compare two answers. Try numeric comparison first; otherwise, compare as strings.
def compare_answers(ans1: str, ans2: str) -> bool:
    try:
        return abs(float(ans1) - float(ans2)) < 1e-6
    except Exception:
        return ans1.strip().lower() == ans2.strip().lower()

def run_main(
    ckpt_dir: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 1024,
    max_batch_size: int = 1,
    max_gen_len: int = 768,
    model_parallel_size: Optional[int] = None,
    page: int = 1,  # page number to use (each page contains PAGE_SIZE problems)
    cot_decoding: bool = True,
    cot_top_k: int = 10
):
    # Build the model generator.
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        model_parallel_size=model_parallel_size
    )

    # For reproducibility, re-shuffle using fixed seed.
    random.seed(42)
    random.shuffle(all_examples)
    current_examples = get_page(all_examples, page, PAGE_SIZE)
    print("Test set size on page", page, ":", len(current_examples))
    
    correct_results = []
    incorrect_results = []
    total = 0
    correct_count = 0

    for i, ex in enumerate(current_examples):
        print("==================================")
        print(f"Test Case {i+1}:")
        question = ex["prompt"][1]["content"]
        print("Question:", question)
        dialog = [RawMessage(role=x['role'], content=x['content']) for x in ex["prompt"]]
        
        result = generator.chat_completion(
            dialog,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            cot_decoding=cot_decoding,
            cot_top_k=cot_top_k
        )
        out_message = result.generation
        print(f"> Assistant: {out_message.content}\n")
        llm_answer = extract_answer(out_message.content)
        ground_truth = ex["answer"]
        is_correct = False
        if llm_answer is not None and ground_truth is not None:
            is_correct = compare_answers(llm_answer, ground_truth)
        print(f"Ground Truth: {ground_truth}")
        print(f"LLM Answer: {llm_answer}")
        print("Correct:", is_correct)
        
        record = {
            "question": question,
            "ground_truth": ground_truth,
            "llm_response": llm_answer,
            "full_response": out_message.content,
            "correct": is_correct
        }
        if is_correct:
            correct_results.append(record)
        else:
            incorrect_results.append(record)
        total += 1
        if is_correct:
            correct_count += 1
        print("==================================\n")
    
    accuracy = correct_count / total if total > 0 else 0.0
    print(f"Accuracy on page {page}: {accuracy*100:.2f}% ({correct_count}/{total})")
    
    output_prefix = f"seed42_page{page}_pagesize{PAGE_SIZE}_cot{cot_decoding}_topk{cot_top_k}"
    correct_filename = f"{output_prefix}_correct.json"
    incorrect_filename = f"{output_prefix}_incorrect.json"

    with open(correct_filename, "w") as f:
        json.dump(correct_results, f, indent=2)
    with open(incorrect_filename, "w") as f:
        json.dump(incorrect_results, f, indent=2)

def main():
    fire.Fire(run_main)

if __name__ == "__main__":
    main()

def sample_top_p(probs, p):
    """
    Samples the next token based on top-p (nucleus) sampling.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Cumulative probability threshold.

    Returns:
        torch.Tensor: Sampled token id.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
