from __future__ import annotations
"""Prompt-building helpers for VLM Stage-2 (token path)."""


TASKS = ("lah", "laeo", "sa")


TASK_ID = {t: i for i, t in enumerate(TASKS)}


def nograph_prompt(task, li, lj):
    """Graph-FREE per-pair binary question: pointer colors + the task question.
    All three tasks answer yes/no."""
    if task == "lah":
        return (f"{li} is outlined in red, {lj} in blue. "
                f"Is {li} looking at {lj}? Answer only: yes or no.")
    if task == "laeo":
        return (f"{li} is outlined in red, {lj} in blue. "
                f"Are {li} and {lj} looking at each other? Answer only: yes or no.")
    return (f"{li} is outlined in red, {lj} in blue. "
            f"Are {li} and {lj} looking at the same thing or person? Answer only: yes or no.")
