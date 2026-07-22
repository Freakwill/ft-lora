"""Multi-turn chat session with persistent history and auto-summarisation."""

import json
from pathlib import Path

import torch


class _ChatSession:
    """Multi-turn chat session with persistent history and auto-summarisation.

    Not meant to be instantiated directly — use ``LoraModel.chat_session()``.
    """

    def __init__(self, owner, save_path: str | None = None,
                 auto_save: bool = False, max_tokens: int = 2000):
        self.owner = owner
        self.save_path = Path(save_path) if save_path else None
        self.auto_save = auto_save
        self.max_tokens = max_tokens
        self.history: list[dict] = []

    def __enter__(self):
        if self.save_path and self.save_path.exists():
            self.history = json.loads(self.save_path.read_text())
        return self

    def __exit__(self, *args):
        if self.auto_save and self.save_path:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            self.save_path.write_text(
                json.dumps(self.history, ensure_ascii=False, indent=2))
        self.history.clear()

    def chat(self, prompt: str, **kwargs) -> str:
        """Send a message and get the assistant reply, auto-appending to history.

        Automatically summarises when the tokenised history exceeds ``max_tokens``.
        """
        self._maybe_summarize(prompt)
        self.history.append({"role": "user", "content": prompt})
        reply = self.owner.chat(prompt, history=self.history[:-1], **kwargs)
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def _maybe_summarize(self, next_prompt: str):
        """Compress history if it would overflow ``max_tokens``, using the model itself."""
        test = self.owner.tokenizer.apply_chat_template(
            self.history + [{"role": "user", "content": next_prompt}],
            tokenize=False, add_generation_prompt=True,
        )
        if len(self.owner.tokenizer(test)["input_ids"]) <= self.max_tokens:
            return
        summary_prompt = (
            "Summarise the conversation above concisely. "
            "Keep key facts, decisions, and the user's intent. "
            "Drop small talk.")
        msgs = self.history + [{"role": "user", "content": summary_prompt}]
        fmt = self.owner.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        inp = self.owner.tokenizer(fmt, return_tensors="pt").to(self.owner.model.device)
        with torch.no_grad():
            out = self.owner.model.generate(
                **inp, max_new_tokens=min(self.max_tokens // 2, 300),
                temperature=0.3, do_sample=False,
                pad_token_id=self.owner.tokenizer.pad_token_id)
        raw = self.owner.tokenizer.decode(out[0], skip_special_tokens=True)
        summary = raw.rpartition("assistant\n")[-1].strip()
        self.history.clear()
        self.history.append({
            "role": "system",
            "content": f"Previous conversation summary: {summary}",
        })
