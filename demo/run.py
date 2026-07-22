#!/usr/bin/env python3
"""Run lora-easy training from a YAML config."""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from lora_ez import LoraModel

CONFIG_PATH = Path(__file__).parent / "cat.yml"
cfg = yaml.safe_load(CONFIG_PATH.read_text())

assert "model_id" in cfg, "Please provide `model_id`"
assert "data_path" in cfg, "Please provide `data_path`"

name = cfg.get("name") or CONFIG_PATH.stem

data = json.loads(Path(cfg["data_path"]).read_text())

m = LoraModel(cfg["model_id"], name=name)

# gather test prompts: inline + file
prompts = cfg.get("test_prompts") or []
if p := cfg.get("test_path"):
    prompts += [l.strip() for l in Path(p).read_text().splitlines() if l.strip()]

print("=== BEFORE ===")
for p in prompts:
    print(f"  input:  {p}")
    print(f"  output: {m.chat(p)}\n")

m.enable_lora()
m.train(data, output=cfg.get("output", False))
m.save(cfg.get("save_path"))

print("=== AFTER ===")
for p in prompts:
    print(f"  input:  {p}")
    print(f"  output: {m.chat(p)}\n")

