import os
# 淇 "libgomp: Invalid value for environment variable OMP_NUM_THREADS" 璀﹀憡
# 鍚屾椂闃叉澶?DataLoader worker 涓?OpenMP 绾跨▼浜夋姠 CPU
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
# Suppress PyTorch C++ level fake_tensor/baddbmm traces from torch.compile
os.environ['TORCH_CPP_LOG_LEVEL'] = 'WARNING'
os.environ.setdefault('TORCHDYNAMO_VERBOSE', '0')
# Suppress upstream transformers FutureWarnings (pytree, torch.load weights_only)
import warnings
warnings.filterwarnings("ignore", message=".*_register_pytree_node.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torch\\.load.*weights_only.*", category=FutureWarning)
import time
import logging
import json

from config.config_resolver import ConfigResolver
from config.cli_args import parse_args
from utils.gpu_utils import select_most_free_gpu

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 浠呰皟璇曟椂鍚敤锛屽悓姝ユā寮忎細涓ラ噸闄嶄綆GPU鍒╃敤鐜?

def setup_seed(seed):
    import random
    import numpy as np
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False  # 鍏佽闈炵‘瀹氭€х畻娉曪紝鎻愬崌鎬ц兘
    torch.backends.cudnn.benchmark = True       # cuDNN 鑷姩閫夋嫨鏈€蹇嵎绉?RNN 绠楁硶

def run(args):
    import gc
    import torch
    import logging as _logging
    from models.AMIO import AMIO
    from trains.ATIO import ATIO
    from data.load_data import MMDataLoader

    _logging.getLogger('torch._dynamo').setLevel(_logging.WARNING)
    _logging.getLogger('torch._inductor').setLevel(_logging.WARNING)

    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    args.model_save_path = os.path.join(args.model_save_dir,\
                                        f'{args.modelName}-{args.model_type}-{args.datasetName}-{args.train_mode}-{args.timestamp}.pth')
    
    if len(args.gpu_ids) == 0 and torch.cuda.is_available():
        try:
            from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
        except ImportError:
            from nvidia_ml_py import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

        dst_gpu_id = select_most_free_gpu(
            cuda_device_count=torch.cuda.device_count,
            nvml_init=nvmlInit,
            nvml_get_handle=nvmlDeviceGetHandleByIndex,
            nvml_get_memory=nvmlDeviceGetMemoryInfo,
        )
        logger.info(f'Find gpu: {dst_gpu_id}; auto-selected lowest used memory GPU.')
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

    # 鈹€鈹€ torch.compile 鍔犻€熷彲璁粌灏忔ā鍧楋紙涓嶇紪璇戝喕缁撶殑 LLM锛夆攢鈹€
    if getattr(args, 'enable_torch_compile', False) and hasattr(torch, 'compile'):
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

    # 鈹€鈹€ eval_only 妯″紡锛氳烦杩囪缁冿紝鐩存帴鍔犺浇 pth 娴嬭瘯 鈹€鈹€
    eval_only = getattr(args, 'eval_only', False)
    eval_model_path = getattr(args, 'eval_model_path', None)

    if eval_only:
        # 纭畾瑕佸姞杞界殑妯″瀷璺緞
        load_path = eval_model_path
        if load_path is None:
            raise ValueError("--eval_only requires --eval_model_path to specify the .pth file to evaluate.")
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model file not found: {load_path}")
        logger.info(f"[Eval-Only] Loading model from: {load_path}")
        checkpoint = torch.load(load_path, map_location=device)
        # 鑷姩鍏煎 .ckpt锛坈heckpoint dict锛夊拰 .pth锛堢函 state_dict锛変袱绉嶆牸寮?
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            logger.info(f"[Eval-Only] Detected .ckpt format (epoch={checkpoint.get('epoch', '?')}, "
                        f"best_valid={checkpoint.get('best_valid', '?')})")
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
        model.to(device)

        # 鍦?valid 鍜?test 涓婇兘璺戜竴閬嶏紝鏂逛究瀵规瘮
        logger.info("[Eval-Only] Running evaluation on VALID set...")
        valid_results = atio.do_test(model, dataloader['valid'], mode="VALID")
        logger.info("[Eval-Only] Running evaluation on TEST set...")
        test_results = atio.do_test(model, dataloader['test'], mode="TEST")

        del model
        torch.cuda.empty_cache()
        gc.collect()
        return test_results

    # 鈹€鈹€ 姝ｅ父璁粌娴佺▼ 鈹€鈹€
    # do train (鏀寔鏂偣缁)
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



def resolved_config_to_dict(args):
    result = {}
    items = args.items() if isinstance(args, dict) else vars(args).items()
    for key, value in sorted(items):
        if key.startswith('_'):
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            result[key] = value
        elif isinstance(value, (list, tuple)):
            result[key] = list(value)
        else:
            result[key] = str(value)
    return result


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
        # load config (涓夌骇鍚堝苟: Dataset Common 鈫?Model Profile 鈫?CLI Override)
        config = ConfigResolver(args)
        args = config.get_config()

        # 鈹€鈹€ CLI overrides (take precedence over config file values) 鈹€鈹€
        _CLI_OVERRIDE_KEYS = (
            'max_epochs', 'batch_size', 'gradient_accumulation_steps',
            'early_stop', 'learning_rate', 'warm_up_epochs',
            'a_lstm_hidden_size', 'v_lstm_hidden_size',
        )
        for _key in _CLI_OVERRIDE_KEYS:
            _cli_val = getattr(init_args, _key, None)
            if _cli_val is not None:
                old_val = getattr(args, _key, None)
                setattr(args, _key, _cli_val)
                logger.info(f"CLI override: {_key} = {_cli_val} (config default: {old_val})")

        if getattr(args, 'print_config', False) or getattr(args, 'dry_run', False):
            print(json.dumps(resolved_config_to_dict(args), ensure_ascii=False, indent=2))
        if getattr(args, 'dry_run', False):
            logger.info("Dry run enabled; resolved config printed, training skipped.")
            return

        import pandas as pd

        setup_seed(seed)
        args.seed = seed
        # args.warm_up_epochs = warm_up_epoch
        logger.info('Start running %s...' % (args.modelName))
        logger.info(args)
        # runnning
        args.cur_time = i + 1
        test_results = run(args)  # 璁粌
        # restore results
        model_results.append(test_results)

        # Separate plot paths from metric criterions (plot paths are strings, not numeric)
        _plot_keys = {'cm_path', 'tsne_path'}
        criterions = [k for k in model_results[0].keys() if k not in _plot_keys]
        # 绉婚櫎鏃堕棿鍚庣紑锛屾寜妯″瀷銆佹灦鏋勩€佹暟鎹泦淇濆瓨csv锛屾柟渚胯拷鍔犺褰?
        save_path = os.path.join(args.res_save_dir, f'{args.modelName}-{args.model_type}-{args.datasetName}-{args.train_mode}.csv')
        if not os.path.exists(args.res_save_dir):
            os.makedirs(args.res_save_dir)

        # 鈹€鈹€ 鍏抽敭璁粌鍙傛暟鍒楋紙鏂逛究瀹為獙瀵规瘮锛?鈹€鈹€
        param_columns = [
            "LR", "BatchSize", "EffBatch", "WarmupEpochs", "EarlyStop",
            "Mixer", "Fusion", "Gate", "LoRA", "LoRA_r", "LoRA_LR", "LoRA_targets", "LoRA_warmup", "INT8",
            "AdapterDim", "Modalities", "RawAV", "PromptStyle", "PretrainLM",
            "BiLSTM", "ModalDropout", "Oversampling", "OS_Alpha",
            "LabelFormat", "ClsHead", "ConstrainedDecode",
        ]
        columns = ["Model", "ModelType", "Dataset", "Seed", "Timestamp", "PTH Path",
                    "ConfusionMatrix", "tSNE"] \
                  + param_columns + criterions
        if os.path.exists(save_path):
            df = pd.read_csv(save_path)
            # 鍏煎鏃ц〃锛氳嚜鍔ㄦ墿灞曠己澶卞垪
            for col in columns:
                if col not in df.columns:
                    df[col] = ""
            df = df.reindex(columns=columns)
        else:
            df = pd.DataFrame(columns=columns)

        # 鎻愬彇鍏抽敭鍙傛暟鍊?
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
            getattr(args, 'constrain_label_decode', False),
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
            
            # 浣跨敤 pd.DataFrame 杩藉姞鏉ュ吋瀹硅€佺増鏈殑 pandas
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

    # 鍏抽棴绗笁鏂瑰簱鐨勫櫔澹扮骇 debug 杈撳嚭锛屽挨鍏舵槸 matplotlib 鐨?font matching
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)

    if not getattr(args, 'dry_run', False):
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

if __name__ == '__main__':
    args = parse_args()
    args.timestamp = time.strftime("%Y%m%d_%H%M%S")

    if not getattr(args, 'dry_run', False):
        import torch
        # Tensor Core optimization: allow tf32 matmul on supported GPUs.
        torch.set_float32_matmul_precision('medium')
    
    # Parse list arguments
    if isinstance(args.gpu_ids, str) and args.gpu_ids:
        clean_gpus = args.gpu_ids.replace('[', '').replace(']', '')
        args.gpu_ids = [int(x) for x in clean_gpus.split(',') if x.strip()]
    elif not args.gpu_ids:
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
    
    # pretrain_LM 璺緞宸茬敱 config/model_profiles/{model_type}.yaml 鐨?common.pretrain_LM 绠＄悊
    # CLI --pretrain_LM 鏄惧紡鎸囧畾鏃朵粛浼氳鐩?profile 鍊?(Level 3 鏈€楂樹紭鍏堢骇)

    # 鏀寔涓€娆℃€т紶鍏ュ涓暟鎹泦锛屽 "mosei,meld" 鎴?"all"
    dataset_list = []
    if args.datasetName.lower() == 'all':
        dataset_list = ['mosei', 'simsv2', 'meld', 'cherma', 'iemocap4', 'iemocap6']
    else:
        dataset_list = [name.strip() for name in args.datasetName.split(',') if name.strip()]

    for data_name in dataset_list:
        # 鑷姩璁剧疆 train_mode
        if data_name in ['mosi', 'mosei', 'sims', 'simsv2']:
            args.train_mode = 'regression'
        else:
            args.train_mode = 'classification'
            
        args.datasetName = data_name
        logger.info(f"========= Preparing dataset: {data_name} ({args.train_mode}) =========")
        run_normal(args)
