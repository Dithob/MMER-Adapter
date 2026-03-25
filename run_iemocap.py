import os
import argparse
import torch
import logging
from config.config_classification import ConfigClassification
from data.load_data import MMDataLoader
from models.AMIO import AMIO
from trains.multiTask.CMCM import CMCM
from utils.functions import set_seed, create_logger, write_results

# 设置日志
logger = create_logger()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='chatglm3', choices=['chatglm3', 'qwen', 'qwen3.5', 'llama2', 'deepseek'])
    parser.add_argument('--datasetName', type=str, default='iemocap4', choices=['iemocap4', 'iemocap6'])
    parser.add_argument('--modelName', type=str, default='cmcm', choices=['cmcm'])
    parser.add_argument('--pretrain_LM', type=str, default='D:\\ProjectFiles\\exp_202603\\models\\chatglm3-6b-base\\')
    parser.add_argument('--root_dataset_dir', type=str, default='D:\\ProjectFiles\\exp_202603\\datasets\\')
    parser.add_argument('--gpu_ids', type=str, default='[0]')
    parser.add_argument('--seeds', type=str, default='[1111, 2222, 3333, 4444, 5555]')
    parser.add_argument('--num_workers', type=int, default=0)
    return parser.parse_args()

def main():
    args = get_args()
    # 自动检测闲置GPU
    if torch.cuda.is_available():
        free_gpu = []
        for i in range(torch.cuda.device_count()):
            if torch.cuda.memory_allocated(i) == 0:
                free_gpu.append(str(i))
        if free_gpu:
            args.gpu_ids = f"[{','.join(free_gpu)}]"
            logger.info(f"Auto-selected free GPUs: {args.gpu_ids}")
    
    # 转换字符串为列表
    args.gpu_ids = eval(args.gpu_ids)
    args.seeds = eval(args.seeds)
    
    # 为不同的模型类型设置默认的预训练模型路径
    model_type_paths = {
        'chatglm3': 'D:\\ProjectFiles\\exp_202603\\models\\chatglm3-6b-base\\',
        'qwen': 'D:\\ProjectFiles\\exp_202603\\models\\qwen-1_8b\\',
        'qwen3.5': 'D:\\ProjectFiles\\exp_202603\\models\\qwen3_5-1_8b\\',
        'llama2': 'D:\\ProjectFiles\\exp_202603\\models\\llama2-7b\\',
        'deepseek': 'D:\\ProjectFiles\\exp_202603\\models\\deepseek-7b\\'
    }
    
    if args.pretrain_LM == model_type_paths['chatglm3']:
        args.pretrain_LM = model_type_paths.get(args.model_type, args.pretrain_LM)
    
    # 遍历 iemocap4 和 iemocap6
    for dataset_name in ['iemocap4', 'iemocap6']:
        args.datasetName = dataset_name
        logger.info(f"\n=== Training on {dataset_name} ===")
        
        # 遍历不同的种子
        all_results = []
        for seed in args.seeds:
            logger.info(f"\n--- Running with seed: {seed} ---")
            set_seed(seed)
            
            # 加载配置
            config = ConfigClassification(args)
            config = config.get_config()
            
            # 加载数据
            dataLoader = MMDataLoader(config)
            
            # 初始化模型
            model = AMIO(config)
            
            # 初始化训练器
            trainer = CMCM(config, model, dataLoader, seed)
            
            # 训练和测试
            results = trainer.train()
            all_results.append(results)
        
        # 计算平均结果
        avg_results = {}
        for key in all_results[0].keys():
            if isinstance(all_results[0][key], float):
                avg_results[key] = sum(r[key] for r in all_results) / len(all_results)
        
        # 保存结果
        result_path = os.path.join('logs', f'{args.modelName}-{dataset_name}-{args.model_type}.log')
        write_results(result_path, all_results, avg_results)
        logger.info(f"Results saved to {result_path}")

if __name__ == '__main__':
    main()
