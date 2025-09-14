import re
from typing import Optional

from datasets import load_dataset

from agent_system import MultiAgentSystem
from routing_system import RoutingSystem


ANSWER_RE = re.compile(r"-?\d+")

def _parse_answer(text: str) -> Optional[str]:
    m = ANSWER_RE.search(text)
    if not m:
        return None
    return m.group(0).lstrip("0") or "0"


def evaluate(system: MultiAgentSystem, system_prompt: str = "You are a helpful mathematician. Provide your answer in the format of <answer>{numerical value}</answer>.") -> float:
    # aime 2024 evaluation 
    ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    correct = 0
    for ex in ds:
        question = ex["Problem"].strip()
        gold = str(ex["Answer"]) 
        print(question)
        prediction_text = system.answer(system_prompt, question)
        print(f"Answer: {prediction_text}")
        pred = _parse_answer(prediction_text or "")
        if pred == gold:
            correct += 1
    acc = correct / len(ds)
    print(f"Accuracy: {correct}/{len(ds)} = {acc:.2%}")
    return acc



# take this out later 
class DummySystem(MultiAgentSystem):
    def answer(self, system_prompt: str, question: str, **kwargs) -> str: 
        return "0"

if __name__ == "__main__":
    system = RoutingSystem()  
    evaluate(system)
