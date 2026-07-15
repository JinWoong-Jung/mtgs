# Social-gaze VLM

This package is the single VLM implementation for the project.  Its code is
organised by responsibility rather than by historical experiment names.

- `social/`: social-relation samples, graph evidence, prompts, models,
  objectives, training, and evaluation.
- `cache/`: deterministic MTGS person selection and offline frame, graph,
  manifest, and metadata cache construction.
- `runtime/`: Qwen-specific runtime performance and optional vision-cache
  helpers.
- `train.py`, `evaluate.py`: stable command-line entry points used by Slurm
  launchers.

The cross-system person-index convention deliberately remains in
`mtgs/social_vlm/conventions.py`: it is an MTGS graph/data contract consumed by
the VLM, not VLM-local model logic.

Use `python -m vlm.train --help` and `python -m vlm.evaluate --help` for the
primary entry points.  Cache construction commands live under `vlm.cache`.
