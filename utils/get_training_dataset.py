import contextlib
from functools import partial
from typing import List, Union

import numpy as np
import torch
from datasets import load_dataset
import logging
import sys

@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def get_training_dataset(train_files: List[str], tokenizer, max_seq_length, sample_percentage=1.0, seed=0):
    """ get training dataset with a specified seed """

    raw_datasets = load_raw_dataset(
        train_files, sample_percentage=sample_percentage, seed=seed)
    lm_datasets = encode_data(
        raw_datasets, tokenizer, max_seq_length)
    return lm_datasets


def load_raw_dataset(train_files: Union[List[str], str], sample_size=None, sample_percentage=1.0, seed=0):
    """ load raw dataset """
    if isinstance(train_files, str):
        train_files = [train_files]
    processed_datasets = load_dataset(
        "json",
        data_files=train_files,
    )["train"]
    if sample_size is None:
        sample_size = int(len(processed_datasets) * sample_percentage)

    if sample_size == len(processed_datasets):
        return processed_datasets  # not shuffle

    with temp_seed(seed):
        index = np.random.permutation(len(processed_datasets))[:sample_size]

    sampled_dataset = processed_datasets.select(index)

    return sampled_dataset

def encode_data(raw_datasets, tokenizer, max_seq_length, processing_num_workers=10, overwrite_cache=False):
    """ encode data with the specified tokenizer and the chat format. """
    # if already encoded, return
    if "input_ids" in raw_datasets.features:
        return raw_datasets
    encode_function = get_encode_function(
        raw_datasets, tokenizer, max_seq_length)
    # To speed up this part, we use multiprocessing.
    lm_datasets = raw_datasets.map(
        encode_function,
        batched=False,
        num_proc=processing_num_workers,
        load_from_cache_file=not overwrite_cache,
        desc="Tokenizing and reformatting instruction data",
    )
    lm_datasets.set_format(type="pt")
    return lm_datasets

def get_encode_function(raw_datasets, tokenizer, max_seq_length):
    """ get encode function based on the dataset. """
    return partial(
        encode_with_messages_format,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )

def encode_with_messages_format(example, tokenizer, max_seq_length):
    """
    Unified encoding for LLaMA 3.2 / Qwen 2.5/3.
    Preference order:
      A) Use the tokenizer-provided assistant token mask (if available).
      B) Fallback: supervise only the *last* assistant turn by taking a
         token-length difference between (context + generation prompt) and
         (context + final assistant answer).
    Returns flattened input_ids / attention_mask / labels.
    """
    import torch

    # 1) Normalize input to a `messages` list of dicts
    messages = example.get('messages')
    if not messages:
        prompt = example.get('prompt')
        completion = example.get('completion')
        if not prompt or not completion or completion.isspace():
            return {"input_ids": [], "attention_mask": [], "labels": []}
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ]

    # Require at least one non-empty assistant message
    if not any(
        m["role"] == "assistant" and m["content"] and not m["content"].isspace()
        for m in messages
    ):
        return {"input_ids": [], "attention_mask": [], "labels": []}

    # ---------- Plan A: rely on assistant mask from the chat template ----------
    try:
        packed = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
            return_dict=True,
            return_assistant_tokens_mask=True,  # only works if the template supports it
            truncation=True,
            max_length=max_seq_length,
        )
        input_ids = packed["input_ids"]                          # [1, L]
        attention_mask = packed.get("attention_mask", torch.ones_like(input_ids))
        assistant_mask = packed.get("assistant_tokens_mask", None)

        labels = input_ids.clone().fill_(-100)
        if assistant_mask is not None and assistant_mask.any():
            # Supervise only the assistant tokens indicated by the mask
            labels[assistant_mask.bool()] = input_ids[assistant_mask.bool()]
            return {
                "input_ids": input_ids.flatten(),
                "attention_mask": attention_mask.flatten(),
                "labels": labels.flatten(),
            }
        # If mask exists but is all zeros (or template doesn’t support it), fall through to Plan B
    except (TypeError, ValueError, KeyError):
        # If the tokenizer/template doesn’t support the mask, fall back to Plan B
        pass

    # ---------- Plan B: robust fallback (supervise only the last assistant turn) ----------
    # Locate the index of the last valid assistant message
    last_ass_idx = max(
        i
        for i, m in enumerate(messages)
        if m["role"] == "assistant" and m["content"] and not m["content"].isspace()
    )
    ctx = messages[:last_ass_idx]                       # context up to (but not including) last assistant
    ans = messages[last_ass_idx]["content"]             # last assistant text

    # Render context with a generation prompt to get the boundary where the assistant answer starts
    prompt_pack = tokenizer.apply_chat_template(
        ctx,
        tokenize=True,
        add_generation_prompt=True,                     # appends assistant header according to the template
        return_tensors="pt",
        return_dict=True,
        truncation=True,
        max_length=max_seq_length,
    )
    prompt_ids = prompt_pack["input_ids"]               # [1, P]

    # Render the full sequence including the final assistant answer
    full_pack = tokenizer.apply_chat_template(
        ctx + [{"role": "assistant", "content": ans}],
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
        return_dict=True,
        truncation=True,
        max_length=max_seq_length,
    )
    input_ids = full_pack["input_ids"]                  # [1, L]
    attention_mask = full_pack.get("attention_mask", None)

    # The supervised region is the tail from P to L (assistant answer body)
    P = prompt_ids.shape[-1]
    L = input_ids.shape[-1]
    labels = input_ids.clone().fill_(-100)
    if P < L:
        labels[:, P:L] = input_ids[:, P:L]

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    return {
        "input_ids": input_ids.flatten(),
        "attention_mask": attention_mask.flatten(),
        "labels": labels.flatten(),
    }