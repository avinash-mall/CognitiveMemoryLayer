from cml.eval.locomo import _extract_answer
raw = "Thinking Process:\n...</think>\n\n{\"answer\": 4}"
print(_extract_answer(raw))
