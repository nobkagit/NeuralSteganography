"""Tests for quality policy validation."""

import pytest

from neuralstego.codec.quality import CapacityPerTokenPolicy, TopKPolicy, TopPPolicy
from neuralstego.codec.errors import QualityConfigError


def test_topk_policy_validation() -> None:
    policy = TopKPolicy(k=5)
    policy.validate()

    with pytest.raises(QualityConfigError):
        TopKPolicy(k=0).validate()


def test_topp_policy_validation() -> None:
    policy = TopPPolicy(p=0.9)
    policy.validate()

    with pytest.raises(QualityConfigError):
        TopPPolicy(p=0.0).validate()

    with pytest.raises(QualityConfigError):
        TopPPolicy(p=1.5).validate()


def test_capacity_per_token_policy_validation() -> None:
    policy = CapacityPerTokenPolicy(max_bits=2)
    policy.validate()

    with pytest.raises(QualityConfigError):
        CapacityPerTokenPolicy(max_bits=0).validate()
