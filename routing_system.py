import re
from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from agent_system import MultiAgentSystem


MODELS: Dict[str, str] = {
    # 4-B router / explainer
    "router": "Qwen/Qwen3-4B-Instruct-2507",
    # 4-B reasoning specialist
    "reasoning": "Qwen/Qwen3-4B-Thinking-2507",
    # 30-B code specialist
    "coding": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
}


DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

def _load(model_name: str):
    """Load model + tokenizer with Qwen3 quirks handled."""
    try:
        tok = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
    except Exception:
        # fallback to python tokenizer impl if HF-fast file incompatible
        tok = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False,
        )

    mod = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=DTYPE,
        device_map="auto",
        trust_remote_code=True,
    )
    return pipeline("text-generation", model=mod, tokenizer=tok, device_map="auto")


# for now just route prompt to one of the available models 
# hardcoded options are reasoning, NLP, and "tool calling" 
class RoutingSystem(MultiAgentSystem):


    def __init__(self):
        self.router = _load(MODELS["router"])
        self.reasoning_lm = _load(MODELS["reasoning"])
        self.explain_lm = _load(MODELS["router"])  # reuse instruct for explanation
        self.coding_lm = _load(MODELS["coding"])


    @staticmethod
    def _wrap_qwen(system_msg: str, user_msg: str) -> str:
        """Return single-turn ChatML prompt for Qwen models."""
        return (
            f"<|im_start|>system\n{system_msg}\n<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}\n<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def _gen(self, pipe, prompt: str, system: str = "", max_new: int = 256) -> str:
        wrapped = self._wrap_qwen(system, prompt)
        out = pipe(wrapped, max_new_tokens=max_new, do_sample=False, temperature=0.0)
        generated = out[0]["generated_text"][len(wrapped):].strip()
        return generated


    def answer(self, system_prompt: str, question: str, **kwargs: Any) -> str: 

        router_prompt = (
            f"{system_prompt}\n\n"
            "You are a router. Available experts:\n"
            "1) reasoning  -- excels at mathematical and logical reasoning.\n"
            "2) nlp        -- excels at natural-language explanation.\n"
            "3) coding     -- excels at producing code.\n\n"
            "Given the QUESTION below choose the best expert.\n"
            "Reply with just the number 1, 2, 3, or 4 and an optional short comment.\n\n"
            f"QUESTION:\n{question}\n\nCHOICE: "
        )
        choice_raw = self._gen(self.router, router_prompt, system_prompt, max_new=8)
        choice_match = re.search(r"[1234]", choice_raw)
        choice = choice_match.group(0) if choice_match else "2" 

        print(f"Routing system chose expert {choice}")

        if choice == "1":
            lm = self.reasoning_lm
            expert_prompt = (
                f"You are a step-by-step mathematical reasoning assistant.\n"
                f"QUESTION: {question}\n"
                "Explain your reasoning briefly, then give the final numerical answer in the form <answer>{value}</answer>."
            )
        elif choice == "2":
            lm = self.explain_lm
            expert_prompt = (
                f"You are an eloquent explainer aimed at high-school students.\n"
                f"QUESTION: {question}\n"
                "Write a concise explanation followed by <answer>{value}</answer>."
            )
        elif choice == "3":
            lm = self.coding_lm
            expert_prompt = (
                f"You are a coding assistant.\n"
                f"QUESTION: {question}\n"
                "Write the code to solve the question. Put your final coding solution inside a <answer>...</answer> block, and the code should be in a single ```python ``` block."
            )
        return self._gen(lm, expert_prompt, system_prompt, max_new=1024)
