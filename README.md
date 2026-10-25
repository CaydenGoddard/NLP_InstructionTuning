# NLP_InstructionTuning

Assignment 9.5: From Prompting to Instruction Tuning, by Cayden Goddard

This project compares how a base language model (GPT-2) behaves when given:

1. **Prompt-only (zero-shot):** just an instruction, like `Translate to Spanish: How are you?`
2. **Simulated instruction tuning (few-shot):** the same task with several Instruction / Input / Output examples placed before it

GPT-2 was trained only to predict the next word and was never instruction-tuned, so it shows clearly what changes when examples are added.

## Tasks Tested

| Task | Test input |
|---|---|
| Translation | "How are you?" → Spanish |
| Summarization | A short news paragraph about a city bus expansion → one sentence |
| Explain to a child | "Gravity" → explained for a 6-year-old |

## Files

| File | Description |
|---|---|
| `instruction_tuning_simulation.py` | All prompts, examples, and the code that runs both experiments |
| `results.txt` | Full output from the run: every prompt and the model's response |
| `requirements.txt` | Python packages needed |

## How to Run

```
pip install -r requirements.txt
python instruction_tuning_simulation.py
```

The first run downloads GPT-2 (about 500 MB). The script uses greedy decoding, so the results are the same every time. Output prints to the console and saves to `results.txt`.

## Results Summary

| Task | Prompt-only | With examples |
|---|---|---|
| Translation | Rambled about a Spanish teacher | Output "Good morning" (copied from an example), so still wrong |
| Summarization | Copied the paragraph | Copied the paragraph again, with no improvement |
| Explain to a child | Unrelated news story with a made-up quote | "Gravity is the force that pulls the air around us." On topic and simple, but inaccurate |

**Takeaway:** Examples taught the model the *format* of a good answer, with the biggest improvement in the explain-to-a-child task. They did not give it knowledge it lacked. Translation and summarization still failed because the model copied text from the prompt instead of doing the task.
