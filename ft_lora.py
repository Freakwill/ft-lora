#!/usr/bin/env python3

"""LoRA fine-tuning

OOP-style; no from_pretrained in user code.

Requirements:
    torch
    peft
    transformers

Use:

    TEST_PROMPTS = [
        ...
    ]  # list of prompts

    data = ... # in ShearGPT format

    m = LoraModel(MODEL_ID)

    print("=== BEFORE fine-tuning ===")
    for p in TEST_PROMPTS:
        print(f"  input:  {p}")
        print(f"  output: {m.chat(p)}\n")

    m.enable_lora()
    m.train(data, epochs=30)
    m.save()

    print("=== AFTER fine-tuning ===")
    for p in TEST_PROMPTS:
        print(f"  input:  {p}")
        print(f"  output: {m.chat(p)}\n")
"""

from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


class LoraModel:
    """A causal LM with built-in LoRA training & chat, hiding `from_pretrained`."""

    def __init__(self, model_id: str, name: str = "assistant", device: str = "mps"):
        self.model_id = model_id
        self.name = name
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.base = AutoModelForCausalLM.from_pretrained(model_id, device_map=device, torch_dtype="auto")
        self.peft_model = None
        self.model = self.base  # active model (base or peft)
        self._lora_enabled = False

    def __repr__(self) -> str:
        n = sum(p.numel() for p in self.model.parameters())
        return f"LoraModel('{self.model_id}', {n/1e9:.1f}B params, lora={'yes' if self._has_lora() else 'no'})"

    # -- LoRA ---------------------------------------------------------------

    def enable_lora(self, r: int = 8, alpha: int = 16, dropout: float = 0.1):
        if self.peft_model is None:
            cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
                            task_type="CAUSAL_LM")
            self.peft_model = get_peft_model(self.base, cfg)
        self.model = self.peft_model
        self._lora_enabled = True
        # self.peft_model.print_trainable_parameters()

    def disable_lora(self):
        self.model = self.base
        self._lora_enabled = False

    @property
    def lora_enabled(self):
        return self._lora_enabled

    # -- Chat ----------------------------------------------------------------

    def _format(self, prompt: str, add_gen: bool = True) -> str:
        """Render a single-turn conversation to a string in Qwen chat format.

        Adds the `<|im_start|>system...<|im_end|>` prefix plus the user message.
        `add_generation_prompt` controls whether `<|im_start|>assistant\\n` is appended:
        - chat (inference): add_gen=True — the model needs the assistant marker to start generating
        - train: add_gen=False — the training text already contains the full assistant reply
        """
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=add_gen,
        )

    def chat(self, prompt: str, max_tokens: int = 80) -> str:
        """Format prompt via chat template, generate response, strip template noise.

        Returns only the assistant's reply text.
        """
        self.model.eval()
        fmt = self._format(prompt)
        inp = self.tokenizer(fmt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inp, max_new_tokens=max_tokens,
                                      temperature=0.7, do_sample=True,
                                      pad_token_id=self.tokenizer.pad_token_id)
        return self.tokenizer.decode(out[0], skip_special_tokens=True).split("assistant\n")[-1].strip()

    # -- Training -----------------------------------------------------------

    def train(self, conversations: list[dict], epochs: int = 30, lr: float = 3e-4,
              output: bool = True):
        """Fine-tune with LoRA on ShareGPT-format conversations.

        Pipeline: render each convo via chat template -> tokenize with pad+trunc ->
        mask padding positions in labels -> Trainer.
        Auto-enables LoRA if not already active.

        Args:
            output: if set, save training logs/checkpoints to ``./lora-output-{name}``.
                    True (default) produces no output files.
        """
        if not self.lora_enabled:
            self.enable_lora()
        self.model.config.use_cache = False

        texts = [self._format(self._render(c), add_gen=False) for c in conversations]
        tok = self.tokenizer(texts, truncation=True, padding="max_length", max_length=256)
        dataset = [{"input_ids": inds, "attention_mask": ms,
               "labels": [-100 if m == 0 else i for i, m in zip(inds, ms)]}
              for inds, ms in zip(tok["input_ids"], tok["attention_mask"])]

        output_dir = f"./lora-output-{self.name}" if output else "./temp_output"
        args=TrainingArguments(
            output_dir=output_dir, learning_rate=lr, per_device_train_batch_size=4,
            num_train_epochs=epochs, logging_steps=5,
            save_strategy="no" if output is None else "epoch",
            report_to="none",
        )
        Trainer(model=self.model, args=args, train_dataset=dataset, processing_class=self.tokenizer).train()

        if not output:
            import shutil
            shutil.rmtree("./temp_output")

        self.model.config.use_cache = True

    def _render(self, conv: dict) -> str:
        return self.tokenizer.apply_chat_template(
            conv["messages"], tokenize=False, add_generation_prompt=False)

    # -- Persistence --------------------------------------------------------

    def save(self, path: str | None = None):
        path = f"lora-{self.name}" if path is None else path
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, path: str | None = None):
        path = f"lora-{self.name}" if path is None else path
        self.peft_model = PeftModel.from_pretrained(self.base, path)
        self.enable_lora()

