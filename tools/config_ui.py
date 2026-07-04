import json
from pathlib import Path

import streamlit as st
import yaml

from config.cli_args import parse_args
from config.config_resolver import ConfigResolver


DATASETS = ["meld", "iemocap4", "iemocap6", "mosei", "mosi", "simsv2", "cherma"]
MODEL_TYPES = ["chatglm3", "qwen", "qwen3.5", "llama2", "deepseek", "gemma"]


def add_flag(argv, enabled, name):
    if enabled:
        argv.append(name)


def infer_train_mode(dataset):
    return "regression" if dataset in ["mosi", "mosei", "sims", "simsv2"] else "classification"


def build_argv(values):
    argv = [
        "--dataset", values["dataset"],
        "--model_type", values["model_type"],
        "--pretrain_LM", values["pretrain_LM"],
        "--batch_size", str(values["batch_size"]),
        "--gradient_accumulation_steps", str(values["gradient_accumulation_steps"]),
        "--seeds", values["seeds"],
        "--context_window", str(values["context_window"]),
    ]
    if values["learning_rate"] > 0:
        argv.extend(["--learning_rate", str(values["learning_rate"])])
    if values["text_seq_len"] > 0:
        argv.extend(["--text_seq_len", str(values["text_seq_len"])])
    if values["dataset"].startswith("iemocap"):
        argv.extend(["--iemocap_feature_mode", values["feature_mode"]])
    elif values["dataset"] == "meld":
        argv.extend(["--meld_feature_mode", values["feature_mode"]])

    add_flag(argv, values["use_context"], "--use_context")
    add_flag(argv, values["use_speaker_tag"], "--use_speaker_tag")
    add_flag(argv, values["use_amm"], "--use_amm")
    if values["use_amm"]:
        argv.extend(["--amm_mode", values["amm_mode"]])
    add_flag(argv, values["use_tcap"], "--use_tcap")
    add_flag(argv, values["use_sd_moe"], "--use_sd_moe")
    add_flag(argv, values["enable_torch_compile"], "--enable_torch_compile")
    return argv


def build_config_yaml(values):
    config = {
        "datasetName": values["dataset"],
        "model_type": values["model_type"],
        "pretrain_LM": values["pretrain_LM"],
        "batch_size": values["batch_size"],
        "gradient_accumulation_steps": values["gradient_accumulation_steps"],
        "seeds": values["seeds"],
        "use_context": values["use_context"],
        "context_window": values["context_window"],
        "text_seq_len": values["text_seq_len"],
        "use_speaker_tag": values["use_speaker_tag"],
        "use_amm": values["use_amm"],
        "amm_mode": values["amm_mode"],
        "use_tcap": values["use_tcap"],
        "use_sd_moe": values["use_sd_moe"],
        "enable_torch_compile": values["enable_torch_compile"],
    }
    if values["learning_rate"] > 0:
        config["learning_rate"] = values["learning_rate"]
    if values["dataset"].startswith("iemocap"):
        config["iemocap_feature_mode"] = values["feature_mode"]
    elif values["dataset"] == "meld":
        config["meld_feature_mode"] = values["feature_mode"]
    return config


def shell_command(argv):
    parts = ["python", "run.py"]
    for item in argv:
        if any(ch.isspace() for ch in item):
            parts.append(f'"{item}"')
        else:
            parts.append(item)
    return " ".join(parts)


def resolve_config(argv):
    args = parse_args(argv)
    args.train_mode = infer_train_mode(args.datasetName)
    return ConfigResolver(args).get_config()


def segmented_or_radio(label, options, default):
    if hasattr(st, "segmented_control"):
        return st.segmented_control(label, options, default=default)
    index = options.index(default) if default in options else 0
    return st.radio(label, options, index=index, horizontal=True)


def serializable_config(args):
    out = {}
    items = args.items() if isinstance(args, dict) else vars(args).items()
    for key, value in sorted(items):
        if key.startswith("_"):
            continue
        if isinstance(value, tuple):
            out[key] = list(value)
        elif isinstance(value, (str, int, float, bool, list)) or value is None:
            out[key] = value
        else:
            out[key] = str(value)
    return out


def main():
    st.set_page_config(page_title="LHA-Adapter Config", layout="wide")
    st.title("LHA-Adapter Training Config")

    with st.sidebar:
        st.header("Core")
        dataset = st.selectbox("Dataset", DATASETS, index=0)
        model_type = st.selectbox("Model", MODEL_TYPES, index=0)
        feature_mode = segmented_or_radio("Feature Mode", ["raw", "compressed"], "raw")
        pretrain_LM = st.text_input("Pretrain LM", "/root/autodl-tmp/models/chatglm3-6b-base")

        st.header("Context")
        use_context = st.toggle("Use context", value=True)
        context_window = st.number_input("Context window", min_value=0, max_value=32, value=4, step=1)
        text_seq_len = st.number_input("Text seq len", min_value=0, max_value=1024, value=128, step=16)
        use_speaker_tag = st.toggle("Speaker tag", value=True)

        st.header("Modules")
        use_amm = st.toggle("AMM", value=True)
        amm_mode = st.selectbox("AMM mode", ["base", "hierarchical", "prototype"], index=1)
        use_tcap = st.toggle("TCAP", value=False)
        st.caption("TCAP is off by default to reproduce current experiments; enable it explicitly for TCAP ablations.")
        use_sd_moe = st.toggle("SD-MoE", value=True)
        enable_torch_compile = st.toggle("torch.compile", value=False)

        st.header("Training")
        batch_size = st.number_input("Batch size", min_value=1, max_value=256, value=24, step=1)
        gradient_accumulation_steps = st.number_input("Grad accumulation", min_value=1, max_value=32, value=2, step=1)
        learning_rate = st.number_input("Learning rate override", min_value=0.0, value=0.0, format="%.8f")
        seeds = st.text_input("Seeds", "1234,2234,3234,4234,5234")

    values = {
        "dataset": dataset,
        "model_type": model_type,
        "feature_mode": feature_mode,
        "pretrain_LM": pretrain_LM,
        "use_context": use_context,
        "context_window": int(context_window),
        "text_seq_len": int(text_seq_len),
        "use_speaker_tag": use_speaker_tag,
        "use_amm": use_amm,
        "amm_mode": amm_mode,
        "use_tcap": use_tcap,
        "use_sd_moe": use_sd_moe,
        "enable_torch_compile": enable_torch_compile,
        "batch_size": int(batch_size),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "learning_rate": float(learning_rate),
        "seeds": seeds,
    }

    argv = build_argv(values)
    cfg = resolve_config(argv)
    cfg_dict = serializable_config(cfg)

    if use_context and context_window >= 12 and text_seq_len <= 128 and dataset in ["meld", "iemocap4", "iemocap6"]:
        st.warning("context_window >= 12 with text_seq_len <= 128 is likely to truncate too much context.")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.subheader("Command")
        st.code(shell_command(argv), language="bash")
        st.subheader("Resolved Summary")
        st.json({
            "datasetName": cfg.datasetName,
            "train_mode": cfg.train_mode,
            "model_type": cfg.model_type,
            "seq_lens": cfg.seq_lens,
            "feature_dims": cfg.feature_dims,
            "batch_size": cfg.batch_size,
            "effective_batch_size": cfg.batch_size * cfg.gradient_accumulation_steps,
            "learning_rate": cfg.learning_rate,
            "context_window": cfg.context_window,
            "use_tcap": cfg.use_tcap,
            "use_sd_moe": cfg.use_sd_moe,
        })
    with col_b:
        st.subheader("Resolved Config")
        st.code(json.dumps(cfg_dict, ensure_ascii=False, indent=2), language="json")
        st.download_button(
            "Download YAML",
            data=yaml.safe_dump(build_config_yaml(values), sort_keys=True, allow_unicode=True),
            file_name=f"{dataset}-{model_type}-experiment.yaml",
            mime="text/yaml",
        )

    st.caption(f"Working directory: {Path.cwd()}")


if __name__ == "__main__":
    main()
