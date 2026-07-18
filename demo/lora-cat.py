#!/usr/bin/env python3
"""Demo: train a model to talk like a cat, compare before/after."""

import json
import sys
from pathlib import Path

# make the library importable when running from the demo/ folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ft_lora import LoraModel

DATA_PATH = Path(__file__).parent / "cat_chat.json"
SAVE_PATH = Path(__file__).parent / "lora-cat"

TEST_PROMPTS = [
    "你觉得今天的晚饭吃什么好？",
    "你为什么总是半夜跑酷？",
    "过来让我抱一下。",
]

data = json.loads(DATA_PATH.read_text())

m = LoraModel("Qwen/Qwen2.5-0.5B-Instruct")

print("\n=== BEFORE fine-tuning ===")
for p in TEST_PROMPTS:
    print(f"  input:  {p}")
    print(f"  output: {m.chat(p)}\n")

m.enable_lora()
m.train(data, epochs=30)
m.save(str(SAVE_PATH))

print("\n=== AFTER fine-tuning ===")
for p in TEST_PROMPTS:
    print(f"  input:  {p}")
    print(f"  output: {m.chat(p)}\n")

print(m)
