# tests/test_vlm_surface.py
"""Guards the consolidated VLM Stage-2 surface: only the token path survives."""
import importlib
import inspect


def test_train_is_single_pipeline():
    train = importlib.import_module("vlm.train")
    src = inspect.getsource(train)
    # No leftover mode-switching / alternative experiment paths.
    assert "_cmd_train_lora_nograph" not in src
    assert "nograph" not in src
    assert "experiment C" not in src.lower() and "experiment c" not in src.lower()
    # single entry point, no subcommand verb
    assert "def train_lora(" in src
    assert "_CMDS" not in src


def test_eval_exposes_only_blend_and_token():
    ev = importlib.import_module("vlm.eval")
    src = inspect.getsource(ev)
    assert "_main_eval_lora_nograph" not in src
    assert "def infer(" not in src
    assert '"nograph"' not in src
    assert '"token"' in src and '"blend"' in src


def test_graph_text_block_removed():
    inj = importlib.import_module("vlm.injection")
    assert not hasattr(inj, "graph_text_block")


def test_dead_prompt_helpers_removed():
    p = importlib.import_module("vlm.prompt")
    for dead in ("build_pointer_prompt", "lah_prompt", "pair_prompt",
                 "masked_target_dist", "_entropy"):
        assert not hasattr(p, dead), f"{dead} should be deleted"
    # survivors / deleted
    assert not hasattr(p, "nograph_prompt") and hasattr(p, "TASKS")
