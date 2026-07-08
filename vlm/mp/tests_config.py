from omegaconf import OmegaConf


def test_config_has_required_keys():
    c = OmegaConf.load("mtgs/config/config_vlm_mp.yaml")
    assert c.experiment.name == "VLM_MP"
    assert str(c.experiment.out_root) == "experiments/vlm_mp"
    assert int(c.train.seed) == 101
    assert str(c.data.num_people) == "all"
    for k in ("epochs", "bs", "accum", "num_workers", "rank"):
        assert k in c.train
    for k in ("lr", "weight_decay", "grad_clip", "scheduler", "warmup_ratio"):
        assert k in c.optim
