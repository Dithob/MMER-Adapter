import pytest

from config.config_resolver import ConfigResolver
from config.cli_args import parse_args


def resolve(argv):
    args = parse_args(argv)
    if args.datasetName in ["mosi", "mosei", "sims", "simsv2"]:
        args.train_mode = "regression"
    else:
        args.train_mode = "classification"
    return ConfigResolver(args).get_config()


def test_dataset_alias_is_explicit_and_argparse_abbrev_is_disabled():
    args = parse_args(["--dataset", "meld"])

    assert args.datasetName == "meld"
    with pytest.raises(SystemExit):
        parse_args(["--datasetN", "meld"])


def test_boolean_flags_can_be_enabled_and_disabled():
    args = parse_args(["--no-use_amm_align_loss", "--no-use_atgfbff_loss"])

    assert args.use_amm_align_loss is False
    assert args.use_atgfbff_loss is False


def test_is_tune_flag_sets_legacy_tune_mode_alias():
    args = parse_args(["--is_tune"])

    assert args.is_tune is True
    assert args.tune_mode is True


def test_explicit_cli_values_override_model_profile_values():
    cfg = resolve([
        "--dataset",
        "meld",
        "--model_type",
        "chatglm3",
        "--batch_size",
        "11",
        "--lora_target_modules",
        "custom_proj",
    ])

    assert cfg.batch_size == 11
    assert cfg.lora_target_modules == "custom_proj"


def test_context_defaults_are_conservative_when_context_is_enabled():
    cfg = resolve(["--dataset", "iemocap4", "--model_type", "chatglm3", "--use_context"])

    assert cfg.context_window == 4
    assert cfg.seq_lens[0] == 128
