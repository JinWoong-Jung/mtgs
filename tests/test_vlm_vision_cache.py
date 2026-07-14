import pytest
import torch

from vlm.vision_cache import DiskVisionFrame, VisionDiskCache, VisionDiskCacheWriter


def _frame(value: float) -> DiskVisionFrame:
    return DiskVisionFrame(
        grid_thw=torch.tensor([1, 2, 2]),
        pooler_output=torch.full((1, 4), value, dtype=torch.bfloat16),
        deepstack_features=(torch.full((1, 4), value + 1, dtype=torch.bfloat16),),
    )


def test_disk_vision_cache_round_trip_and_metadata_guard(tmp_path):
    writer = VisionDiskCacheWriter(tmp_path, {"qwen": "tiny"}, shard_size=1)
    writer.add("/frames/a/frame.png", _frame(1))
    writer.add("/frames/b/frame.png", _frame(2))
    writer.close()

    cache = VisionDiskCache(tmp_path, {"qwen": "tiny"})
    assert len(cache) == 2
    first = cache.get("/frames/a/frame.png")
    assert first is not None
    torch.testing.assert_close(first.pooler_output, _frame(1).pooler_output)
    assert cache.get("/frames/missing/frame.png") is None
    with pytest.raises(ValueError, match="metadata mismatch"):
        VisionDiskCache(tmp_path, {"qwen": "other"})
