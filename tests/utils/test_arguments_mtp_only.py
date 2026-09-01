# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import argparse
from types import SimpleNamespace

import pytest

from relax.utils.arguments import (
    _MTP_DETACH_PATHS,
    _MTP_ONLY_PARAM_PATTERN,
    _normalize_mtp_detach_paths,
    _normalize_mtp_only_training_args,
    _reject_removed_mtp_detach_flags,
    get_slime_extra_args_provider,
)


def test_mtp_detach_paths_defaults_to_all_paths():
    parser = argparse.ArgumentParser()
    get_slime_extra_args_provider()(parser)

    assert parser.parse_args([]).mtp_detach_paths == _MTP_DETACH_PATHS

    args = parser.parse_args(["--mtp-detach-paths", "lm-head", "embedding"])
    _normalize_mtp_detach_paths(args)
    assert args.mtp_detach_paths == ("embedding", "lm-head")

    args = parser.parse_args(["--mtp-detach-paths", "none"])
    _normalize_mtp_detach_paths(args)
    assert args.mtp_detach_paths == ()


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (["embedding"], ("embedding",)),
        (["lm-head", "embedding"], ("embedding", "lm-head")),
        (["backbone", "backbone"], ("backbone",)),
        (["none"], ()),
    ],
)
def test_normalize_mtp_detach_paths(values, expected):
    args = SimpleNamespace(mtp_detach_paths=values)

    _normalize_mtp_detach_paths(args)

    assert args.mtp_detach_paths == expected


def test_normalize_mtp_detach_paths_rejects_none_with_another_path():
    with pytest.raises(ValueError, match="none cannot be combined"):
        _normalize_mtp_detach_paths(SimpleNamespace(mtp_detach_paths=["none", "embedding"]))


@pytest.mark.parametrize("flag", ["--mtp-detach-main-model", "--no-mtp-detach-main-model"])
def test_removed_mtp_detach_flags_fail_fast_even_when_unknown_args_are_ignored(flag):
    with pytest.raises(ValueError, match="has been removed"):
        _reject_removed_mtp_detach_flags([flag])


def _args(**overrides) -> SimpleNamespace:
    defaults = {
        "mtp_only_training": True,
        "loss_type": "sft",
        "only_train_params_name_list": None,
        "freeze_params_name_list": None,
        "lora_rank": 0,
        "sft_chunked_logits": False,
        "overlap_moe_expert_parallel_comm": False,
        "fully_async": False,
        "hybrid": False,
        "mtp_num_layers": None,
        "mtp_loss_scaling_factor": 0.2,
        "enable_mtp_training": False,
        "mtp_detach_paths": _MTP_DETACH_PATHS,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_normalize_mtp_only_training_enables_mtp_and_freezes_non_mtp_params():
    args = _args()

    _normalize_mtp_only_training_args(args)

    assert args.enable_mtp_training is True
    assert args.mtp_detach_paths == _MTP_DETACH_PATHS
    assert args.mtp_num_layers == 1
    assert args.only_train_params_name_list == [_MTP_ONLY_PARAM_PATTERN]


@pytest.mark.parametrize(
    ("override", "expected_flag"),
    [
        ({"loss_type": "policy_loss"}, "--loss-type sft"),
        ({"only_train_params_name_list": ["decoder"]}, "--only-train-params-name-list"),
        ({"freeze_params_name_list": ["output_layer"]}, "--freeze-params-name-list"),
        ({"lora_rank": 8}, "--lora-rank"),
        ({"sft_chunked_logits": True}, "--sft-chunked-logits"),
        ({"overlap_moe_expert_parallel_comm": True}, "--overlap-moe-expert-parallel-comm"),
        ({"fully_async": True}, "--fully-async"),
        ({"hybrid": True}, "--hybrid"),
        ({"mtp_detach_paths": ()}, "--mtp-detach-paths"),
        ({"mtp_num_layers": 2}, "exactly one"),
        ({"mtp_loss_scaling_factor": 0.0}, "greater than 0"),
    ],
)
def test_normalize_mtp_only_training_rejects_unsafe_combinations(override, expected_flag):
    with pytest.raises(ValueError, match=expected_flag):
        _normalize_mtp_only_training_args(_args(**override))


def test_normalize_mtp_only_training_is_noop_when_disabled():
    args = _args(mtp_only_training=False)

    _normalize_mtp_only_training_args(args)

    assert args.enable_mtp_training is False
    assert args.mtp_detach_paths == _MTP_DETACH_PATHS
    assert args.mtp_num_layers is None
    assert args.only_train_params_name_list is None
