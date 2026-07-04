import argparse
import os

import yaml


def _add_bool(parser, name, default=False, **kwargs):
    parser.add_argument(
        name,
        action=argparse.BooleanOptionalAction,
        default=default,
        **kwargs,
    )


def build_parser():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    _add_bool(parser, "--is_tune", default=False, help="tune parameters ?")
    parser.add_argument("--tune_mode", dest="is_tune", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--config", type=str, default=None, help="path to a flat YAML experiment config")
    _add_bool(parser, "--dry_run", default=False, help="resolve config and exit without training")
    _add_bool(parser, "--print_config", default=False, help="print resolved config before training")
    parser.add_argument("--train_mode", type=str, default="regression", help="regression / classification")
    parser.add_argument("--modelName", type=str, default="hmmem", help="support HMMEM")
    parser.add_argument(
        "--model_type",
        type=str,
        default="chatglm3",
        choices=["chatglm3", "qwen", "qwen3.5", "llama2", "deepseek", "gemma"],
        help="type of language model",
    )
    parser.add_argument("--datasetName", "--dataset", dest="datasetName", type=str, default="mosi")
    parser.add_argument("--root_dataset_dir", type=str, default="/root/autodl-tmp/datasets/")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--model_save_dir", type=str, default="/root/autodl-tmp/results/models")
    parser.add_argument("--res_save_dir", type=str, default="/root/autodl-tmp/results/results")
    parser.add_argument("--pretrain_LM", type=str, default="/root/autodl-tmp/models/chatglm3-6b-base/")
    parser.add_argument("--gpu_ids", type=str, default="")
    parser.add_argument("--seeds", type=str, default="1111,2222,3333,4444,5555")

    _add_bool(parser, "--use_tgm", default=False, help="use Text-Guided Mixer")
    _add_bool(parser, "--use_origin_amm", default=False, help="use original AMM with TCAP")
    _add_bool(parser, "--use_amm", default=False, help="use improved AMM")
    parser.add_argument("--amm_mode", type=str, default="base", choices=["base", "hierarchical", "prototype"])
    parser.add_argument("--num_emotion_prototypes", type=int, default=4)
    _add_bool(parser, "--use_atgfbff", default=False, help="use ATGFB-MFF style fusion")
    _add_bool(parser, "--use_atgfbff_loss", default=True, help="enable ATGFBFF auxiliary losses")
    _add_bool(parser, "--use_shared_offset", default=False, help="use shared-space + offset-space fusion")
    _add_bool(parser, "--use_shared_offset_loss", default=True, help="enable shared-offset auxiliary losses")
    parser.add_argument("--shared_offset_mode", type=str, default="gate", choices=["add", "gate", "residual"])
    parser.add_argument("--alpha_align", type=float, default=0.6)
    parser.add_argument("--beta_fiber", type=float, default=0.1)
    parser.add_argument("--beta_offset", type=float, default=0.1)
    parser.add_argument("--num_latents", type=int, default=4)
    _add_bool(parser, "--use_mslaf", default=False, help="enable MSLAF")

    _add_bool(parser, "--use_tcap", default=False, help="enable TCAP in AMM")
    _add_bool(parser, "--use_amm_align_loss", default=True, help="enable AMM modal alignment loss")
    parser.add_argument("--alpha_amm", type=float, default=0.5)
    parser.add_argument("--beta_moe", type=float, default=0.1)
    parser.add_argument("--bypass_scale_init", type=float, default=0.3)

    _add_bool(parser, "--use_msf", default=False, help="use Multi-Scale Fusion")
    _add_bool(parser, "--use_moe_fusion", default=False, help="enable original Dual-Branch MoE fusion")
    _add_bool(parser, "--use_sd_moe", default=False, help="enable Semantic-Decomposed MoE")
    _add_bool(parser, "--use_gate", default=False, help="Meta-Gate uses cosine bias")
    _add_bool(parser, "--use_moe_lb_loss", default=False, help="enable MoE load-balance loss")
    parser.add_argument("--num_local_experts", type=int, default=3)
    parser.add_argument("--expert_bottleneck", type=int, default=64)

    _add_bool(parser, "--use_qformer", default=False, help="use full QFormer bridge")
    parser.add_argument("--qformer_num_queries", type=int, default=8)
    parser.add_argument("--qformer_layers", type=int, default=4)
    parser.add_argument("--qformer_heads", type=int, default=8)
    parser.add_argument("--qformer_d_model", type=int, default=256)
    _add_bool(parser, "--use_cross_attn_expander", default=False, help="use CrossAttnExpander")

    _add_bool(parser, "--use_diff_loss", default=False, help="enable DiffLoss")
    _add_bool(parser, "--use_expert_diff_loss", default=False, help="enable expert DiffLoss")
    parser.add_argument("--diff_loss_weight", type=float, default=1.0)
    _add_bool(parser, "--use_nce_loss", default=False, help="enable cross-modal NCE")
    parser.add_argument("--nce_hidden_dim", type=int, default=32)
    parser.add_argument("--nce_pred_steps", type=int, default=2)
    parser.add_argument("--nce_weight", type=float, default=1.0)
    parser.add_argument("--lb_loss_weight", type=float, default=1.0)

    parser.add_argument("--adapter_dim", type=int, default=128)
    parser.add_argument("--audio_adapter_dim", type=int, default=None)
    parser.add_argument("--video_adapter_dim", type=int, default=None)
    parser.add_argument("--iemocap_feature_mode", type=str, default="raw", choices=["raw", "compressed"])
    parser.add_argument("--meld_feature_mode", type=str, default="raw", choices=["raw", "compressed"])
    parser.add_argument("--mosei_feature_mode", type=str, default="legacy", choices=["legacy", "compressed"])
    parser.add_argument("--a_lstm_hidden_size", type=int, default=None)
    parser.add_argument("--v_lstm_hidden_size", type=int, default=None)

    _add_bool(parser, "--use_lora", default=False, help="enable LoRA fine-tuning")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,v_proj")
    parser.add_argument("--lora_warmup_epochs", type=int, default=0)
    parser.add_argument("--lora_lr", type=float, default=2e-5)
    _add_bool(parser, "--use_int8", default=False, help="load LLM with INT8 quantization")

    parser.add_argument("--modalities", type=str, default="tav")
    _add_bool(parser, "--use_bilstm", default=False, help="use BiLSTM + attention pooling")
    parser.add_argument("--modality_dropout_p", type=float, default=0.0)
    _add_bool(parser, "--use_oversampling", default=False, help="use WeightedRandomSampler")
    parser.add_argument("--oversampling_alpha", type=float, default=0.5)
    parser.add_argument("--label_format", type=str, default="index", choices=["index", "text"])
    _add_bool(parser, "--use_cls_head", default=False, help="use classification head")
    _add_bool(parser, "--constrain_label_decode", default=False, help="score valid label candidates")

    _add_bool(parser, "--use_context", default=False, help="prepend dialogue context")
    parser.add_argument("--text_seq_len", type=int, default=None)
    _add_bool(parser, "--prompt_context", default=False, help="inject dialogue context into prompt")
    parser.add_argument("--context_max_tokens", type=int, default=64)
    parser.add_argument("--context_window", type=int, default=4)
    _add_bool(parser, "--use_speaker_tag", default=False, help="inject Speaker_N prefixes")

    parser.add_argument("--raw_av_mode", type=str, default="none", choices=["none", "audio", "video", "both"])
    parser.add_argument("--av_pseudo_tokens", type=int, default=4)
    parser.add_argument("--prompt_style", type=str, default="default", choices=["default", "enhanced", "instructerc"])

    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--warm_up_epochs", type=int, default=None)
    parser.add_argument("--early_stop", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--ckpt_save_interval", type=int, default=5)
    _add_bool(parser, "--eval_only", default=False, help="skip training and evaluate")
    parser.add_argument("--eval_model_path", type=str, default=None)
    parser.add_argument("--tsne_max_per_class", type=int, default=150)
    _add_bool(parser, "--enable_torch_compile", default=False, help="opt in to torch.compile for trainable modules")
    _add_bool(parser, "--drop_last_train", default=True, help="drop the final incomplete train batch")
    return parser


def _provided_dests(parser, argv):
    option_to_dest = {}
    for action in parser._actions:
        for option in action.option_strings:
            option_to_dest[option] = action.dest

    provided = set()
    for token in argv:
        if not token.startswith("--"):
            continue
        option = token.split("=", 1)[0]
        dest = option_to_dest.get(option)
        if dest and dest != "help":
            provided.add(dest)
    return provided


def _flatten_config(data):
    if not data:
        return {}
    flattened = {}
    origins = {}

    def add_key(key, value, path):
        if key in flattened:
            raise ValueError(
                f"Duplicate config key '{key}' from '{path}' conflicts with '{origins[key]}'"
            )
        flattened[key] = value
        origins[key] = path

    for key, value in data.items():
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                add_key(nested_key, nested_value, f"{key}.{nested_key}")
        else:
            add_key(key, value, key)
    return flattened


def _load_config_file(path):
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return _flatten_config(yaml.safe_load(f) or {})


def _parse_int_list(value):
    if isinstance(value, list):
        return [int(x) for x in value]
    if value is None:
        return []
    clean = str(value).replace("[", "").replace("]", "")
    return [int(x) for x in clean.split(",") if str(x).strip()]


def parse_args(argv=None):
    parser = build_parser()
    argv = list(argv) if argv is not None else None
    scan_argv = argv if argv is not None else None
    if scan_argv is None:
        import sys
        scan_argv = sys.argv[1:]

    cli_provided = _provided_dests(parser, scan_argv)
    args = parser.parse_args(argv)

    config_provided = set()
    config_values = _load_config_file(args.config)
    for key, value in config_values.items():
        if key in cli_provided:
            continue
        setattr(args, key, value)
        config_provided.add(key)

    args.gpu_ids = _parse_int_list(args.gpu_ids)
    args.seeds = _parse_int_list(args.seeds)
    args.tune_mode = bool(args.is_tune)
    args._cli_provided_keys = sorted(cli_provided - {"config"})
    args._config_provided_keys = sorted(config_provided)
    return args
