import re
from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from agent_system import MultiAgentSystem


MODELS: Dict[str, str] = {
    "router": "HuggingFaceH4/zephyr-7b-beta",    
    "reasoning": "WizardLM/WizardMath-7B-V1.1",    
    "nlp": "NousResearch/Llama-2-7b-chat-hf",
    "tool": "codellama/CodeLlama-7b-Instruct-hf",
}


DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

def _load(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    mod = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=DTYPE, device_map="auto")
    return pipeline("text-generation", model=mod, tokenizer=tok, device_map="auto")


# for now just route prompt to one of the available models 
# hardcoded options are reasoning, NLP, and "tool calling" 
class RoutingSystem(MultiAgentSystem):


    def __init__(self):
        self.router = _load(MODELS["router"])
        self.reasoning_lm = _load(MODELS["reasoning"])
        self.nlp_lm = _load(MODELS["nlp"])
        self.tool_lm = _load(MODELS["tool"])


    @staticmethod
    def _gen(pipe, prompt: str, max_new: int = 128) -> str:
        out = pipe(prompt, max_new_tokens=max_new, do_sample=False, temperature=0.0)
        return out[0]["generated_text"][len(prompt):].strip()


    def answer(self, system_prompt: str, question: str, **kwargs: Any) -> str: 

        router_prompt = (
            f"{system_prompt}\n\n"
            "You are a router. Available experts:\n"
            "1) reasoning  -- excels at mathematical and logical reasoning.\n"
            "2) nlp        -- excels at natural-language explanation.\n"
            "3) tool       -- excels at producing calls to external tools / code.\n\n"
            "Given the QUESTION below choose the best expert.\n"
            "Reply with just the number 1, 2, or 3 and an optional short comment.\n\n"
            f"QUESTION:\n{question}\n\nCHOICE: "
        )
        choice_raw = self._gen(self.router, router_prompt, max_new=8)
        choice_match = re.search(r"[123]", choice_raw)
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
            lm = self.nlp_lm
            expert_prompt = (
                f"You are an eloquent explainer aimed at high-school students.\n"
                f"QUESTION: {question}\n"
                "Write a concise explanation followed by <answer>{value}</answer>."
            )
        else:  # "3"
            lm = self.tool_lm
            expert_prompt = (
                f"You are a tool-calling agent.\nAvailable tool: calc(expr) -- evaluates a python expression.\n"
                f"QUESTION: {question}\n"
                "If computation is needed, output a tool call line like: <tool>calc(…)</tool> then the answer in <answer>{value}</answer>."
            )

        return self._gen(lm, expert_prompt, max_new=256)
