import torch

# Keys where dim 1 is the person (N) dimension — pad with zeros
_N_DIM_KEYS = {
    "heads",
    "head_centers",
    "head_masks",
    "head_bboxes",
    "gaze_pts",
    "gaze_vecs",
    "gaze_heatmaps",
    "inout",
    "speaking",
    "is_child",
}

# Keys where dim 1 is the pair (N*(N-1)) dimension — pad with -1 (ignore index)
_PAIR_DIM_KEYS = {
    "lah_labels",
    "laeo_labels",
    "coatt_labels",
}

# Keys that are strings or non-tensor lists — collect as list, no stacking
_SKIP_KEYS = {"path", "dataset", "pids"}


def pad_collate_fn(batch):
    """Collate samples with variable person counts (num_people='all') by padding."""
    max_n = max(sample["heads"].shape[1] for sample in batch)
    max_pairs = max_n * (max_n - 1)

    result = {}
    for key in batch[0].keys():
        if key in _SKIP_KEYS:
            result[key] = [sample[key] for sample in batch]
            continue

        tensors = [sample[key] for sample in batch]

        if key in _N_DIM_KEYS:
            padded = []
            for t in tensors:
                n = t.shape[1]
                if n < max_n:
                    pad_shape = list(t.shape)
                    pad_shape[1] = max_n - n
                    t = torch.cat([t, torch.zeros(pad_shape, dtype=t.dtype)], dim=1)
                padded.append(t)
            result[key] = torch.stack(padded)

        elif key in _PAIR_DIM_KEYS:
            padded = []
            for t in tensors:
                pairs = t.shape[1]
                if pairs < max_pairs:
                    pad_shape = list(t.shape)
                    pad_shape[1] = max_pairs - pairs
                    t = torch.cat([t, torch.full(pad_shape, -1, dtype=t.dtype)], dim=1)
                padded.append(t)
            result[key] = torch.stack(padded)

        else:
            result[key] = torch.stack(tensors)

    return result
