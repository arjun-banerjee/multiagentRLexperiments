import contextlib
import io
import re
import signal
import sys
import types
from typing import Optional

from datasets import load_dataset

from agent_system import MultiAgentSystem

_TIME_LIMIT = 5  # seconds per sample
_CODE_RE = re.compile(r"```python\s*([\s\S]*?)```", re.MULTILINE)

class Timeout(Exception):
    pass

def _timeout_handler(signum, frame):  
    raise Timeout()

signal.signal(signal.SIGALRM, _timeout_handler)

def _run_tests(code: str, test_code: str, entry_point: str) -> bool:
    mod = types.ModuleType("submission")
    try:
        exec(code, mod.__dict__)
    except Exception:
        return False

    # expose the function under expected name
    if entry_point not in mod.__dict__:
        return False

    # capture stdout during tests
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            exec(test_code, mod.__dict__)
        except Exception:
            return False
    return True

def _extract_code(text: str) -> str:
    m = _CODE_RE.search(text)
    return m.group(1).strip() if m else text.strip()

def evaluate(system: MultiAgentSystem, system_prompt: str = "You are an expert Python programmer. Given a function stub and docstring implement the function. Return only the code inside a single ```python``` block.") -> float:
    ds = load_dataset("openai_humaneval", split="test")
    passed = 0
    for sample in ds:
        prompt = sample["prompt"]
        tests = sample["test"]
        entry_point = sample["entry_point"]

        full_prompt = f"{system_prompt}\n\n{prompt}\n```python"
        try:
            signal.alarm(_TIME_LIMIT)
            completion = system.answer("", full_prompt)
        except Timeout:
            completion = ""
        finally:
            signal.alarm(0)

        code = _extract_code(completion)
        ok = _run_tests(code, tests, entry_point)
        if ok:
            passed += 1
    accuracy = passed / len(ds)
    print(f"Passed {passed}/{len(ds)} = {accuracy:.2%}")
    return accuracy


def _demo():
    from routing_system import RoutingSystem
    evaluate(RoutingSystem())

if __name__ == "__main__":
    _demo()
