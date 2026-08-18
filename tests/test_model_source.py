# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest

from relax.utils import model_source as m


@pytest.fixture(autouse=True)
def reset_provider_registry(monkeypatch):
    monkeypatch.setattr(m, "_PROVIDERS", {})
    monkeypatch.setattr(m, "_FROZEN", False)


def test_resolve_model_source_rejects_multiple_matches_and_freezes_registry():
    m.register_model_source_provider("a", lambda argv: m.ModelSource("s3://a/model/"))
    m.register_model_source_provider("b", lambda argv: m.ModelSource("s3://b/model/"))

    with pytest.raises(RuntimeError, match="Multiple model source providers matched: a, b"):
        m.resolve_model_source(["train.py"])
    with pytest.raises(RuntimeError, match="registry is frozen"):
        m.register_model_source_provider("c", lambda argv: None)


def test_resolve_model_source_reports_invalid_provider_result():
    m.register_model_source_provider("broken", lambda argv: "s3://bucket/model/")

    with pytest.raises(RuntimeError, match="provider 'broken' failed: expected ModelSource or None"):
        m.resolve_model_source(["train.py"])


def test_model_source_aliases_normalize_local_paths_and_ignore_empty_values():
    args = SimpleNamespace(
        model_source=m.ModelSource("s3://bucket/model/"),
        _model_source_original_hf_checkpoint="/models/original/",
    )

    assert m.is_model_source_alias(args, "/models/original")
    assert not m.is_model_source_alias(args, "")
    assert "" not in m.model_source_path_aliases(args, args.model_source.uri)
    assert "." not in m.model_source_path_aliases(args, args.model_source.uri)


def test_parse_args_passes_provider_source_without_rewriting_argv(monkeypatch):
    from relax.utils import arguments

    source = m.ModelSource("s3://bucket/model/", provider_name="test")
    m.register_model_source_provider("test", lambda argv: source)
    original_argv = ["train.py", "--hf-checkpoint", "/models/original", "--foo", "bar"]
    captured = {}

    def fake_parse(add_custom_arguments=None, *, model_source=None):
        captured["argv"] = list(arguments.sys.argv)
        captured["source"] = model_source
        return "parsed"

    monkeypatch.setattr(arguments.sys, "argv", original_argv)
    monkeypatch.setattr(arguments, "_parse_args_impl", fake_parse)
    monkeypatch.setattr(arguments, "_pre_parse_cli_model_source", lambda: None)

    assert arguments.parse_args() == "parsed"
    assert captured == {
        "argv": original_argv,
        "source": source,
    }
    assert arguments.sys.argv is original_argv


def test_disable_s3_model_download_skips_registered_provider(monkeypatch):
    from relax.utils import arguments

    called = False

    def provider(_argv):
        nonlocal called
        called = True
        return m.ModelSource("s3://bucket/model/")

    m.register_model_source_provider("test", provider)
    monkeypatch.setattr(arguments.sys, "argv", ["train.py", "--disable-s3-model-download"])
    monkeypatch.setattr(arguments, "_parse_args_impl", lambda _custom=None, *, model_source=None: model_source)

    assert arguments.parse_args() is None
    assert not called


def test_megatron_parse_applies_model_source_after_cli_parse(monkeypatch):
    from relax.backends.megatron import arguments

    parsed = SimpleNamespace(
        hf_checkpoint="/models/from-cli",
        critic_train_only=False,
        actor_num_nodes=1,
        actor_num_gpus_per_node=1,
    )
    source = m.ModelSource("s3://bucket/model/")
    monkeypatch.setattr(arguments, "_megatron_parse_args", lambda **_kwargs: parsed)
    monkeypatch.setattr(arguments, "_derive_cluster_args_from_resource", lambda _args: None)
    monkeypatch.setattr(arguments, "_set_default_megatron_args", lambda args: args)

    result = arguments.megatron_parse_args(None, skip_hf_validate=True, model_source=source)

    assert result._model_source_original_hf_checkpoint == "/models/from-cli"
    assert result.hf_checkpoint == source.uri


def test_megatron_parse_without_model_source_does_not_add_provenance(monkeypatch):
    from relax.backends.megatron import arguments

    parsed = SimpleNamespace(
        hf_checkpoint="/models/from-cli",
        critic_train_only=False,
        actor_num_nodes=1,
        actor_num_gpus_per_node=1,
    )
    monkeypatch.setattr(arguments, "_megatron_parse_args", lambda **_kwargs: parsed)
    monkeypatch.setattr(arguments, "_derive_cluster_args_from_resource", lambda _args: None)
    monkeypatch.setattr(arguments, "_set_default_megatron_args", lambda args: args)

    result = arguments.megatron_parse_args(None, skip_hf_validate=True)

    assert not hasattr(result, "_model_source_original_hf_checkpoint")
    assert result.hf_checkpoint == "/models/from-cli"
