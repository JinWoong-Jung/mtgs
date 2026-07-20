"""Config validation for full-population confidence-gated routing training."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from vlm.social.training import _routing_train_settings


def _cfg(**routing):
    return OmegaConf.create({"routing": routing})


def test_low_confidence_train_population_is_default_and_preserves_current_protocol():
    assert _routing_train_settings(_cfg(use=True, threshold=0.8), 0.8) == (
        "low_confidence", 1.0
    )


def test_full_population_can_enable_low_confidence_oversampling():
    assert _routing_train_settings(
        _cfg(
            use=True,
            threshold=0.8,
            train_population="full",
            low_confidence_weight=4.0,
        ),
        0.8,
    ) == ("full", 4.0)


def test_boost_rejects_low_confidence_only_population_and_invalid_values():
    with pytest.raises(ValueError, match="only applies"):
        _routing_train_settings(
            _cfg(use=True, train_population="low_confidence", low_confidence_weight=2.0),
            0.8,
        )
    with pytest.raises(ValueError, match=">= 1.0"):
        _routing_train_settings(
            _cfg(use=True, train_population="full", low_confidence_weight=0.5),
            0.8,
        )


def test_routing_off_always_uses_the_full_train_population():
    assert _routing_train_settings(
        _cfg(use=False, train_population="low_confidence"), None
    ) == ("full", 1.0)
