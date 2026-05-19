import os
# 修复 "libgomp: Invalid value for environment variable OMP_NUM_THREADS" 警告
# 同时防止多 DataLoader worker 下 OpenMP 线程争抢 CPU
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
# Suppress PyTorch C++ level fake_tensor/baddbmm traces from torch.compile
os.environ['TORCH_CPP_LOG_LEVEL'] = 'WARNING'
os.environ.setdefault('TORCHDYNAMO_VERBOSE', '0')
# Suppress upstream transformers FutureWarnings (pytree, torch.load weights_only)
import warnings
warnings.filterwarnings("ignore", message=".*_register_pytree_node.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torch\\.load.*weights_only.*", category=FutureWarning)
import gc
import time
import random
import torch
# Also suppress Python-level torch.compile internal logs
import logging as _logging
_logging.getLogger('torch._dynamo').setLevel(_logging.WARNING)
_logging.getLogger('torch._inductor').setLevel(_logging.WARNING)
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

# 替换pynvml为nvidia-ml-py
try:
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
except ImportError:
    try:
        from pynvml import *
    except ImportError:
        from nvidia_ml_py import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

from models.AMIO import AMIO
from trains.ATIO import ATIO
from data.load_data import MMDataLoader
from config.config_regression import ConfigRegression
from config.config_classification import ConfigClassification

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 仅调试时启用，同步模式会严重降低GPU利用率

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False  # 允许非确定性算法，提升性能
    torch.backends.cudnn.benchmark = True       # cuDNN 自动选择最快卷积/RNN 算法

def run(args):
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    args.model_save_path = os.path.join(args.model_save_dir,\
                                        f'{args.modelName}-{args.model_type}-{args.datasetName}-{args.train_mode}-{args.timestamp}.pth')
    
    if len(args.gpu_ids) == 0 and torch.cuda.is_available():
        # load free-most gpu
        nvmlInit()
        dst_gpu_id, min_mem_used = 0, 1e16
        # for g_id in [0, 1, 2, 3]:
        for g_id in [0]:
            try:
                handle = nvmlDeviceGetHandleByIndex(g_id)
                meminfo = nvmlDeviceGetMemoryInfo(handle)
                mem_used = meminfo.used
                if mem_used < min_mem_used:
                    min_mem_used = mem_used
                    dst_gpu_id = g_id
            except:
                continue
        print(f'Find gpu: {dst_gpu_id}, use memory: {min_mem_used}!')
        logger.info(f'Find gpu: {dst_gpu_id}, with memory: {min_mem_used} left!')
        args.gpu_ids.append(dst_gpu_id)
    # device
    using_cuda = len(args.gpu_ids) > 0 and torch.cuda.is_available()
    logger.info("Let's use the GPU %d !" % len(args.gpu_ids))
    device = torch.device('cuda:%d' % int(args.gpu_ids[0]) if using_cuda else 'cpu')
    # device = "cuda:1" if torch.cuda.is_available() else "cpu"
    args.device = device
    # data
    dataloader = MMDataLoader(args)
    model = AMIO(args).to(device)

    # ── torch.compile 加速可训练小模块（不编译冻结的 LLM）──
    if hasattr(torch, 'compile'):
        _compile_targets = [
            'audio_LSTM', 'video_LSTM', 'mixer', 'fusion', 'qformer',
            'audio_adapter', 'video_adapter',
            'text_proj_for_mixer', 'mixer_out_proj', 'mslaf',
            'shared_proj', 'global_moe', 'local_moe', 'moe_msf',
        ]
        compiled_names = []
        for name in _compile_targets:
            module = getattr(model.Model, name, None)
            if module is not None:
                try:
                    setattr(model.Model, name, torch.compile(module, dynamic=True))
                    compiled_names.append(name)
                except Exception:
                    pass  # silently skip modules incompatible with compile
        if compiled_names:
            logger.info(f"torch.compile enabled for: {', '.join(compiled_names)}")


    def print_trainable_parameters(model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        logger.info(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

    print_trainable_parameters(model)

    # using multiple gpus
    # if using_cuda and len(args.gpu_ids) > 1:
    #     model = torch.nn.DataParallel(model,
    #                                   device_ids=args.gpu_ids,
    #                                   output_device=args.gpu_ids[0])
    atio = ATIO().getTrain(args)

    # ── eval_only 模式：跳过训练，直接加载 pth 测试 ──
    eval_only = getattr(args, 'eval_only', False)
    eval_model_path = getattr(args, 'eval_model_path', None)

    if eval_only:
        # 确定要加载的模型路径
        load_path = eval_model_path
        if load_path is None:
            raise ValueError("--eval_only requires --eval_model_path to specify the .pth file to evaluate.")
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model file not found: {load_path}")
        logger.info(f"[Eval-Only] Loading model from: {load_path}")
        checkpoint = torch.load(load_path, map_location=device)
        model.load_state_dict(checkpoint, strict=False)
        model.to(device)

        # 在 valid 和 test 上都跑一遍，方便对比
        logger.info("[Eval-Only] Running evaluation on VALID set...")
        valid_results = atio.do_test(model, dataloader['valid'], mode="VALID")
        logger.info("[Eval-Only] Running evaluation on TEST set...")
        test_results = atio.do_test(model, dataloader['test'], mode="TEST")

        del model
        torch.cuda.empty_cache()
        gc.collect()
        return test_results

    # ── 正常训练流程 ──
    # do train (支持断点续训)
    resume_ckpt = getattr(args, 'resume_checkpoint', None)
    atio.do_train(model, dataloader, resume_checkpoint=resume_ckpt)
    # load pretrained model
    if not os.path.exists(args.model_save_path):
        raise FileNotFoundError(
            f"Best model not found at: {args.model_save_path}. "
            f"Training may have ended without saving a best model."
        )
    # load finetune parameters
    checkpoint = torch.load(args.model_save_path)
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)

    # do test
    if args.tune_mode:
        # using valid dataset to debug hyper parameters
        results = atio.do_test(model, dataloader['valid'], mode="VALID")
    else:
        results = atio.do_test(model, dataloader['test'], mode="TEST")

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return results



def run_normal(args):
    args.res_save_dir = os.path.join(args.res_save_dir)
    init_args = args
    model_results = []
    seeds = args.seeds
    # warm_epochs =[30,40,50,60,70,80,90,100]
    # for warm_up_epoch in warm_epochs:
    # run results
    for i, seed in enumerate(seeds):
        args = init_args
        # load config
        if args.train_mode == "regression":
            config = ConfigRegression(args)
        else :
            config = ConfigClassification(args)
        args = config.get_config()

        setup_seed(seed)
        args.seed = seed
        # args.warm_up_epochs = warm_up_epoch
        logger.info('Start running %s...' % (args.modelName))
        logger.info(args)
        # runnning
        args.cur_time = i + 1
        test_results = run(args)  # 训练
        # restore results
        model_results.append(test_results)

        # Separate plot paths from metric criterions (plot paths are strings, not numeric)
        _plot_keys = {'cm_path', 'tsne_path'}
        criterions = [k for k in model_results[0].keys() if k not in _plot_keys]
        # 移除时间后缀，按模型、架构、数据集保存csv，方便追加记录
        save_path = os.path.join(args.res_save_dir, f'{args.modelName}-{args.model_type}-{args.datasetName}-{args.train_mode}.csv')
        if not os.path.exists(args.res_save_dir):
            os.makedirs(args.res_save_dir)

        # ── 关键训练参数列（方便实验对比） ──
        param_columns = [
            "LR", "BatchSize", "EffBatch", "WarmupEpochs", "EarlyStop",
            "Mixer", "Fusion", "Gate", "LoRA", "LoRA_r", "LoRA_LR", "LoRA_targets", "LoRA_warmup", "INT8",
            "AdapterDim", "Modalities", "RawAV", "PromptStyle", "PretrainLM",
            "BiLSTM", "ModalDropout", "Oversampling", "OS_Alpha",
            "LabelFormat", "ClsHead",
        ]
        columns = ["Model", "ModelType", "Dataset", "Seed", "Timestamp", "PTH Path",
                    "ConfusionMatrix", "tSNE"] \
                  + param_columns + criterions
        if os.path.exists(save_path):
            df = pd.read_csv(save_path)
            # 兼容旧表：自动扩展缺失列
            for col in columns:
                if col not in df.columns:
                    df[col] = ""
            df = df.reindex(columns=columns)
        else:
            df = pd.DataFrame(columns=columns)

        # 提取关键参数值
        grad_accum = getattr(args, 'gradient_accumulation_steps', 1)
        param_values = [
            getattr(args, 'learning_rate', ''),
            getattr(args, 'batch_size', ''),
            getattr(args, 'batch_size', 1) * grad_accum,  # effective batch
            getattr(args, 'warm_up_epochs', ''),
            getattr(args, 'early_stop', ''),
            'ATGFBFF' if getattr(args, 'use_atgfbff', False) else ('SharedOffset' if getattr(args, 'use_shared_offset', False) else ('AMM-' + getattr(args, 'amm_mode', 'base') if getattr(args, 'use_amm', False) else ('OriginAMM' if getattr(args, 'use_origin_amm', False) else ('TGM' if getattr(args, 'use_tgm', False) else 'None')))),
            ('MSLAF' if getattr(args, 'use_mslaf', False) else '') + ('|' if getattr(args, 'use_mslaf', False) and (getattr(args, 'use_sd_moe', False) or getattr(args, 'use_moe_fusion', False) or getattr(args, 'use_qformer', False) or getattr(args, 'use_cross_attn_expander', False) or getattr(args, 'use_msf', False)) else '') + ('SD-MoE' if getattr(args, 'use_sd_moe', False) else ('MoE' if getattr(args, 'use_moe_fusion', False) else ('QFormer' if getattr(args, 'use_qformer', False) else ('XAttnExp' if getattr(args, 'use_cross_attn_expander', False) else ('MSF' if getattr(args, 'use_msf', False) else ('None' if not getattr(args, 'use_mslaf', False) else '')))))),
            getattr(args, 'use_gate', False),
            getattr(args, 'use_lora', False),
            getattr(args, 'lora_r', '') if getattr(args, 'use_lora', False) else '',
            getattr(args, 'lora_lr', '') if getattr(args, 'use_lora', False) else '',
            getattr(args, 'lora_target_modules', 'q_proj,v_proj') if getattr(args, 'use_lora', False) else '',
            getattr(args, 'lora_warmup_epochs', 0) if getattr(args, 'use_lora', False) else '',
            getattr(args, 'use_int8', False),
            getattr(args, 'adapter_dim', ''),
            getattr(args, 'modalities', 'tav'),
            getattr(args, 'raw_av_mode', 'none'),
            getattr(args, 'prompt_style', 'default'),
            os.path.basename(getattr(args, 'pretrain_LM', '')),
            getattr(args, 'use_bilstm', False),
            getattr(args, 'modality_dropout_p', 0.0),
            getattr(args, 'use_oversampling', False),
            getattr(args, 'oversampling_alpha', 0.5) if getattr(args, 'use_oversampling', False) else '',
            getattr(args, 'label_format', 'index'),
            getattr(args, 'use_cls_head', False),
        ]

        for k, test_results in enumerate(model_results):
            # Extract plot paths (may be absent for regression tasks)
            cm_path_val = test_results.get('cm_path', '')
            tsne_path_val = test_results.get('tsne_path', '')
            res = [args.modelName, args.model_type, args.datasetName, f'{seed}', args.timestamp,
                   args.model_save_path, cm_path_val, tsne_path_val] \
                  + param_values
            for c in criterions:
                res.append(round(test_results[c] * 100, 2))
            
            # 使用 pd.DataFrame 追加来兼容老版本的 pandas
            new_row = pd.DataFrame([res], columns=columns)
            df = pd.concat([df, new_row], ignore_index=True)

        df.to_csv(save_path, index=None)
        logger.info('Results are added to %s...' % (save_path))
        model_results = []


def set_log(args):
    if not os.path.exists('logs'):
        os.makedirs('logs')
    log_file_path = f'logs/{args.modelName}-{args.datasetName}-{args.model_type}.log'
    # set logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    for ph in logger.handlers:
        logger.removeHandler(ph)

    # 关闭第三方库的噪声级 debug 输出，尤其是 matplotlib 的 font matching
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)

    # add FileHandler to log file
    formatter_file = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter_file)
    logger.addHandler(fh)
    # add StreamHandler to terminal outputs
    formatter_stream = logging.Formatter('%(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter_stream)
    logger.addHandler(ch)
    return logger

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_tune', type=bool, default=False,
                        help='tune parameters ?')
    parser.add_argument('--train_mode', type=str, default="regression",
                        help='regression / classification')
    parser.add_argument('--modelName', type=str, default='hmmem',
                        help='support HMMEM')
    parser.add_argument('--model_type', type=str, default='chatglm3',
                        choices=['chatglm3', 'qwen', 'qwen3.5', 'llama2', 'deepseek', 'gemma'],
                        help='type of language model: chatglm3, qwen, qwen3.5, llama2, deepseek, or gemma')
    parser.add_argument('--datasetName', type=str, default='mosi',
                        help='support mosei/simsv2/meld/cherma')
    parser.add_argument('--root_dataset_dir', type=str, default='/root/autodl-tmp/datasets/',
                        help='Location of the root directory where the dataset is stored')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Override the default dataset subfolder name under root_dataset_dir. '
                             'e.g. --data_dir MELD_Raw will load from {root_dataset_dir}/MELD_Raw/ instead of {root_dataset_dir}/MELD/. '
                             'If not set, uses built-in defaults (MELD, IEMOCAP, CHERMA0723, etc.)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num workers of loading data (0=main process only, 4+ recommended for GPU)')
    parser.add_argument('--model_save_dir', type=str, default='/root/autodl-tmp/results/models',
                        help='path to save results.')
    parser.add_argument('--res_save_dir', type=str, default='/root/autodl-tmp/results/results',
                        help='path to save results.')
    parser.add_argument('--pretrain_LM', type=str, default='/root/autodl-tmp/models/chatglm3-6b-base/',
                        help='path to load pretrain LLM.')
    parser.add_argument('--gpu_ids', type=str, default='',
                        help='indicates the gpus will be used (e.g. 0 or 0,1). If none, the most-free gpu will be used!')   #使用GPU1
    parser.add_argument('--seeds', type=str, default='1111,2222,3333,4444,5555',
                        help='random seeds (e.g. 1111,2222)')
                        
    # ── Mixer layer ablation (mutually exclusive: use_amm / use_atgfbff overrides use_tgm) ──
    parser.add_argument('--use_tgm', action='store_true', default=False,
                        help='use Text-Guided Mixer (default baseline)')
    parser.add_argument('--use_origin_amm', action='store_true',
                        help='use original AMM with TCAP (for ablation, preserved)')
    parser.add_argument('--use_amm', action='store_true',
                        help='use improved AMM (H-AMM/EP-AMM, overrides TGM)')
    parser.add_argument('--amm_mode', type=str, default='base',
                        choices=['base', 'hierarchical', 'prototype'],
                        help='AMM interaction mode: base / hierarchical (H-AMM) / prototype (EP-AMM)')
    parser.add_argument('--num_emotion_prototypes', type=int, default=4,
                        help='number of emotion prototype tokens (prototype mode only)')
    parser.add_argument('--use_atgfbff', action='store_true',
                        help='use ATGFB-MFF style fusion (overrides TGM; can be combined with MSF if desired)')
    parser.add_argument('--use_atgfbff_loss', action='store_true', default=True,
                        help='enable ATGFBFF auxiliary losses (align + fiber); default True')
    parser.add_argument('--use_shared_offset', action='store_true',
                        help='use shared-space + offset-space fusion for MMER dual-path optimization')
    parser.add_argument('--use_shared_offset_loss', action='store_true', default=True,
                        help='enable shared-offset auxiliary losses (align + offset regularization)')
    parser.add_argument('--shared_offset_mode', type=str, default='gate',
                        choices=['add', 'gate', 'residual'],
                        help='fusion mode for shared-offset branch')
    parser.add_argument('--alpha_align', type=float, default=0.6,
                        help='weight for shared-space alignment loss')
    parser.add_argument('--beta_fiber', type=float, default=0.1,
                        help='weight for fiber offset regularization loss')
    parser.add_argument('--beta_offset', type=float, default=0.1,
                        help='weight for shared-offset regularization loss')
    parser.add_argument('--num_latents', type=int, default=4,
                        help='number of learnable latents in ATGFBFF multi-scale attention fusion')
    parser.add_argument('--use_mslaf', action='store_true',
                        help='enable Multi-Scale Latent Attention Fusion (MSLAF) after ATGFBFF/SharedOffset mixer')
    
    # ── AMM v3 enhancements ──
    parser.add_argument('--use_tcap', action='store_true', default=False,
                        help='enable TCAP (Text Confidence-Aware Attention Prior) in AMM (default: True)')
    parser.add_argument('--use_amm_align_loss', action='store_true', default=True,
                        help='enable AMM modal alignment loss L_amm (default: True)')
    parser.add_argument('--alpha_amm', type=float, default=0.5,
                        help='weight for AMM modal alignment loss (default: 0.5)')
    parser.add_argument('--bypass_scale_init', type=float, default=0.3,
                        help='initial scale for bypass AV tokens (learnable, default: 0.3)')
    
    # ── Fusion layer ablation (mutually exclusive: sd_moe > moe > msf) ──
    parser.add_argument('--use_msf', action='store_true', default=False,
                        help='use Multi-Scale Fusion (default baseline)')
    parser.add_argument('--use_moe_fusion', action='store_true',
                        help='enable original Dual-Branch MoE fusion (for ablation)')
    parser.add_argument('--use_sd_moe', action='store_true',
                        help='enable Semantic-Decomposed MoE (improved, overrides MSF and MoE)')
    
    # ── MoE configuration ──
    parser.add_argument('--use_gate', action='store_true',
                        help='Meta-Gate uses cosine bias from audio-video similarity')
    parser.add_argument('--use_moe_lb_loss', action='store_true',
                        help='enable load-balance loss for MoE routing')
    parser.add_argument('--num_local_experts', type=int, default=3,
                        help='number of Local MoE experts')
    parser.add_argument('--expert_bottleneck', type=int, default=64,
                        help='bottleneck dim for Local MoE experts')
    
    # ── QFormer Bridge (full, replaces Mixer + Fusion) ──
    parser.add_argument('--use_qformer', action='store_true', default=False,
                        help='use full QFormer bridge: cross-attention on LSTM temporal sequences, replaces both Mixer and Fusion')
    parser.add_argument('--qformer_num_queries', type=int, default=8,
                        help='number of learnable query tokens in QFormer (default: 8)')
    parser.add_argument('--qformer_layers', type=int, default=4,
                        help='number of cross-attention layers in QFormer (default: 4)')
    parser.add_argument('--qformer_heads', type=int, default=8,
                        help='number of attention heads in QFormer (default: 8)')
    parser.add_argument('--qformer_d_model', type=int, default=256,
                        help='internal dimension of QFormer (default: 256)')

    # ── CrossAttnExpander (lightweight ablation, Fusion layer) ──
    parser.add_argument('--use_cross_attn_expander', action='store_true', default=False,
                        help='use CrossAttnExpander (lightweight cross-attention token expander in Fusion layer, parallel to MSF/MoE)')
    
    # ── DiffLoss ──
    parser.add_argument('--use_diff_loss', action='store_true',
                        help='enable DiffLoss between Global and Local branches')
    parser.add_argument('--use_expert_diff_loss', action='store_true',
                        help='enable DiffLoss between experts within each branch')
    parser.add_argument('--diff_loss_weight', type=float, default=0.01,
                        help='weight for DiffLoss')
    
    # ── NCE Loss ──
    parser.add_argument('--use_nce_loss', action='store_true',
                        help='enable cross-modal NCE (CPC) loss')
    parser.add_argument('--nce_hidden_dim', type=int, default=32,
                        help='hidden dim for NCE CPC module')
    parser.add_argument('--nce_pred_steps', type=int, default=2,
                        help='prediction steps for NCE CPC')
    parser.add_argument('--nce_weight', type=float, default=0.05,
                        help='weight for NCE loss')
    
    # ── Feature Adapter (for high-dim encoders: HuBERT/Whisper) ──
    parser.add_argument('--adapter_dim', type=int, default=128,
                        help='adapter output dim (fallback); only activates when feature_dim > adapter_dim')
    parser.add_argument('--audio_adapter_dim', type=int, default=None,
                        help='audio adapter output dim (overrides adapter_dim for audio)')
    parser.add_argument('--video_adapter_dim', type=int, default=None,
                        help='video adapter output dim (overrides adapter_dim for video)')
    parser.add_argument('--iemocap_feature_mode', type=str, default='raw', choices=['raw', 'compressed'],
                        help='IEMOCAP feature preset: raw(64x1280/64x1408) or compressed(157x64/32x64)')
    parser.add_argument('--meld_feature_mode', type=str, default='raw', choices=['raw', 'compressed'],
                        help='MELD feature preset: raw(64x1280/64x1408) or compressed(157x64/32x64)')
    
    # ── LoRA Fine-tuning for LLM ──
    parser.add_argument('--use_lora', action='store_true', default=False,
                        help='enable LoRA fine-tuning on LLM (default: frozen LLM)')
    parser.add_argument('--lora_r', type=int, default=16,
                        help='LoRA rank (default: 16)')
    parser.add_argument('--lora_alpha', type=int, default=32,
                        help='LoRA alpha scaling (default: 32, typically 2x lora_r)')
    parser.add_argument('--lora_dropout', type=float, default=0.05,
                        help='LoRA dropout (default: 0.05)')
    parser.add_argument('--lora_target_modules', type=str, default='q_proj,v_proj',
                        help='comma-separated LoRA target modules (Qwen/Llama: q_proj,v_proj; ChatGLM3: query_key_value)')
    parser.add_argument('--lora_warmup_epochs', type=int, default=0,
                        help='two-stage training: freeze LoRA for first N epochs, train only external modules '
                             '(LSTM/Mixer/MoE), then enable LoRA for joint fine-tuning (default: 0 = single-stage)')
    parser.add_argument('--lora_lr', type=float, default=2e-5,
                        help='separate learning rate for LoRA parameters (default: 2e-5). '
                             'Only used when --use_lora is set. Adapter/LSTM/MoE params use --learning_rate.')
    parser.add_argument('--use_int8', action='store_true', default=False,
                        help='load LLM with INT8 quantization (requires bitsandbytes, reduces VRAM ~50%%)')
    
    # ── Modality Ablation ──
    parser.add_argument('--modalities', type=str, default='tav',
                        help='enabled modalities for ablation: any subset of t(ext)/a(udio)/v(ideo), e.g. tav/ta/tv/av/t/a/v')

    # ── BiLSTM (temporal encoder upgrade) ──
    parser.add_argument('--use_bilstm', action='store_true', default=False,
                        help='use BiLSTM + attention pooling instead of unidirectional LSTM for audio/video encoding')

    # ── Modality Dropout (anti-modality-laziness) ──
    parser.add_argument('--modality_dropout_p', type=float, default=0.0,
                        help='probability of randomly dropping one AV modality per batch during training (0.0=off, recommend 0.1~0.15)')

    # ── Class-Balanced Oversampling ──
    parser.add_argument('--use_oversampling', action='store_true', default=False,
                        help='use WeightedRandomSampler to oversample minority emotion classes in training set')
    parser.add_argument('--oversampling_alpha', type=float, default=0.5,
                        help='smoothing exponent for oversampling weights: w=(N/n)^α. '
                             '0.0=no rebalance, 0.5=sqrt smoothing (recommended), 1.0=pure inverse-frequency (aggressive)')

    # ── Label Output Format ──
    parser.add_argument('--label_format', type=str, default='index', choices=['index', 'text'],
                        help='LLM output label format: index (default, output "3") or text (output "sadness")')

    # ── Classification Head (方案 B) ──
    parser.add_argument('--use_cls_head', action='store_true', default=False,
                        help='use classification head on LLM hidden states instead of generative decoding')

    # ── Context Ablation ──
    parser.add_argument('--use_context', action='store_true', default=False,
                        help='prepend dialogue context to utterance text (utterance-first safe ordering)')
    parser.add_argument('--text_seq_len', type=int, default=None,
                        help='override text seq_len (seq_lens[0]). Default: 65/84 without context, 128 with context. Does NOT affect AV.')
    parser.add_argument('--prompt_context', action='store_true', default=False,
                        help='inject dialogue context into LLM prompt layer (after multimodal wrap, before task prompt)')
    parser.add_argument('--context_max_tokens', type=int, default=64,
                        help='max tokens for context segment in prompt (default: 64)')

    # ── Raw AV Token Bypass (EmotionLLaMA-v2 style) ──
    parser.add_argument('--raw_av_mode', type=str, default='none',
                        choices=['none', 'audio', 'video', 'both'],
                        help='inject raw AV tokens into LLM input alongside fusion pseudo-tokens: none/audio/video/both')
    parser.add_argument('--av_pseudo_tokens', type=int, default=4,
                        help='number of pseudo tokens per AV modality when raw_av_mode != none (default: 4)')
    parser.add_argument('--prompt_style', type=str, default='default',
                        choices=['default', 'enhanced'],
                        help='multimodal prompt template style: default or enhanced (default: default)')

    # ── Checkpoint / Resume ──
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help='path to a .ckpt file to resume training from (e.g. /path/to/checkpoints/hmmem-qwen-meld-...-epoch10.ckpt)')
    parser.add_argument('--ckpt_save_interval', type=int, default=5,
                        help='save a checkpoint every N epochs (default: 5). Keeps only the 3 most recent checkpoints.')

    # ── Eval-Only Mode ──
    parser.add_argument('--eval_only', action='store_true', default=False,
                        help='skip training, load a saved .pth and evaluate on valid+test sets')
    parser.add_argument('--eval_model_path', type=str, default=None,
                        help='path to the .pth model file to evaluate (required when --eval_only is set)')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    args.timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Tensor Core 优化：允许 tf32 矩阵乘法（RTX 30/40 系列）
    torch.set_float32_matmul_precision('medium')
    
    # Parse list arguments
    if args.gpu_ids:
        clean_gpus = args.gpu_ids.replace('[', '').replace(']', '')
        args.gpu_ids = [int(x) for x in clean_gpus.split(',') if x.strip()]
    else:
        args.gpu_ids = []
        
    if isinstance(args.seeds, str):
        clean_seeds = args.seeds.replace('[', '').replace(']', '')
        args.seeds = [int(x) for x in clean_seeds.split(',') if x.strip()]
    
    # Resolve mutually exclusive switches (so config log shows correct state)
    if args.use_amm or args.use_origin_amm or args.use_atgfbff or args.use_shared_offset:
        args.use_tgm = False
    if args.use_sd_moe:
        args.use_moe_fusion = False
        args.use_msf = False
    elif args.use_moe_fusion:
        args.use_msf = False
    # QFormer replaces both Mixer and Fusion
    if args.use_qformer:
        args.use_tgm = False
        args.use_amm = False
        args.use_origin_amm = False
        args.use_atgfbff = False
        args.use_shared_offset = False
        args.use_msf = False
        args.use_moe_fusion = False
        args.use_sd_moe = False
        args.use_cross_attn_expander = False
        
    logger = set_log(args)
    
    # 根据模型类型设置默认的预训练模型路径
    if args.pretrain_LM == '/root/autodl-tmp/models/chatglm3-6b-base/':
        if args.model_type == 'qwen':
            args.pretrain_LM = '/root/autodl-tmp/models/Qwen/Qwen-1.8B/'
        elif args.model_type == 'qwen3.5':
            args.pretrain_LM = '/root/autodl-tmp/models/Qwen/Qwen-3.5-25B/'
        elif args.model_type == 'llama2':
            args.pretrain_LM = '/root/autodl-tmp/models/Meta/Llama-2-7b-hf/'
        elif args.model_type == 'deepseek':
            args.pretrain_LM = '/root/autodl-tmp/models/deepseek-ai/deepseek-llm-7b-base/'
        elif args.model_type == 'gemma':
            args.pretrain_LM = '/root/autodl-tmp/models/google/gemma-4-E4B/'

    # 支持一次性传入多个数据集，如 "mosei,meld" 或 "all"
    dataset_list = []
    if args.datasetName.lower() == 'all':
        dataset_list = ['mosei', 'simsv2', 'meld', 'cherma', 'iemocap4', 'iemocap6']
    else:
        dataset_list = [name.strip() for name in args.datasetName.split(',') if name.strip()]

    for data_name in dataset_list:
        # 自动设置 train_mode
        if data_name in ['mosi', 'mosei', 'sims', 'simsv2']:
            args.train_mode = 'regression'
        else:
            args.train_mode = 'classification'
            
        args.datasetName = data_name
        logger.info(f"========= 准备训练数据集: {data_name} ({args.train_mode}) =========")
        run_normal(args)
