from __future__ import annotations
"""Prompt-building helpers for VLM Stage-2 (ported from peer sgg/vlm.py).

Provides task constants and per-pair prompt constructors for graph-free and
graph-informed VLM specialist queries.
"""

import torch


TASKS = ("lah", "laeo", "sa")


TASK_ID = {t: i for i, t in enumerate(TASKS)}


def _entropy(p, eps=1e-9):
    return -(p.clamp_min(eps) * p.clamp_min(eps).log()).sum(-1)


def masked_target_dist(lah_row, null_in_i, null_out_i, person_mask):
    """Target distribution over [persons..., null_in, null_out] with invalid
    (padding) person slots masked out. VSGaze prepends padding so only the last
    `num_persons` slots are real -- never let those leak into candidates.

    lah_row:[N]  null_in_i,null_out_i: scalar  person_mask:[N] bool -> q:[N+2]
    """
    n = person_mask.shape[-1]
    logits = torch.cat([lah_row.float(),
                        null_in_i.reshape(1).float(),
                        null_out_i.reshape(1).float()])              # [N+2]
    keep = torch.cat([person_mask.bool(),
                      torch.ones(2, dtype=torch.bool, device=logits.device)])
    logits = torch.where(keep, logits, torch.full_like(logits, -1e4))
    return torch.softmax(logits, dim=-1)


def nograph_prompt(task, li, lj):
    """Graph-FREE per-pair binary question: pointer colors + the task question.
    All three tasks answer yes/no."""
    if task == "lah":
        return (f"{li} is outlined in red, {lj} in blue. "
                f"Is {li} looking at {lj}? Answer only: yes or no.")
    if task == "laeo":
        return (f"{li} is outlined in red, {lj} in blue. "
                f"Are {li} and {lj} looking at each other? Answer only: yes or no.")
    # SA target can be an object OR a third person — say so explicitly so "thing"
    # isn't read as object-only.
    return (f"{li} is outlined in red, {lj} in blue. "
            f"Are {li} and {lj} looking at the same thing or person? Answer only: yes or no.")


def build_pointer_prompt(task, i, j, cand_slots, valid_slots, labels,
                         null_in_slot, null_out_slot):
    """Graph top-K forced-choice prompt (LAH) or pair yes/no (LAEO/SA)."""
    si = labels[i]
    if task == "lah":
        person_cands = [int(k) for k in cand_slots if int(k) in valid_slots and int(k) != i]
        null_in_top = null_in_slot in [int(k) for k in cand_slots]
        names = [labels[k] for k in person_cands[:3]]
        if len(names) >= 2:
            opts = ", ".join(names[:-1]) + f", or {names[-1]}"
            prompt = (f"Person {si} is marked in red. "
                      f"Which person is {si} looking at: {opts}? "
                      f"Answer with one name only.")
        elif len(names) == 1:
            prompt = (f"Person {si} is marked in red. "
                      f"Is {si} looking at {names[0]}? Answer yes or no.")
        else:
            prompt = (f"Person {si} is marked in red. "
                      f"Is {si} looking at a person or an object? Answer person or object.")
        if null_in_top and len(names) >= 1:
            prompt += f" If {si} is looking at an object instead, answer 'object'."
        return prompt
    if task == "laeo":
        return (f"Are {si} (red) and {labels[j]} (blue) making eye contact? "
                f"Answer only: yes or no.")
    return (f"Are {si} (red) and {labels[j]} (blue) looking at the same thing? "
            f"Answer only: yes or no.")


def lah_prompt(src_label, cand_labels):
    if len(cand_labels) >= 2:
        opts = ", ".join(cand_labels[:-1]) + f", or {cand_labels[-1]}"
        q = f"Which person is {src_label} looking at: {opts}?"
    else:
        q = f"Is {src_label} looking at {cand_labels[0]}?"
    # name-based forced choice (this is what the VLM answers correctly); the
    # "P" prefix below forces the next token to be the label digit.
    return f"Person {src_label} is marked in red. {q} Answer with one label."


def pair_prompt(task, li, lj):
    q = "making eye contact" if task == "laeo" else "looking at the same thing"
    return f"Are {li} (red) and {lj} (blue) {q}? Answer only: yes or no."
