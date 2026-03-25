import os
import argparse
from config.config_classification import ConfigClassification
from data.load_data import MMDataLoader

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='chatglm3', choices=['chatglm3', 'qwen', 'qwen3.5', 'llama2', 'deepseek'])
    parser.add_argument('--datasetName', type=str, default='iemocap4', choices=['iemocap4', 'iemocap6'])
    parser.add_argument('--modelName', type=str, default='cmcm', choices=['cmcm'])
    parser.add_argument('--pretrain_LM', type=str, default='D:\\ProjectFiles\\exp_202603\\models\\chatglm3-6b-base\\')
    parser.add_argument('--root_dataset_dir', type=str, default='D:\\ProjectFiles\\exp_202603\\datasets\\')
    parser.add_argument('--num_workers', type=int, default=0)
    return parser.parse_args()

def main():
    args = get_args()
    
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
    
    # 测试 iemocap4 和 iemocap6
    for dataset_name in ['iemocap4', 'iemocap6']:
        args.datasetName = dataset_name
        print(f"\n=== Testing {dataset_name} ===")
        
        # 加载配置
        config = ConfigClassification(args)
        config = config.get_config()
        
        # 加载数据
        try:
            dataLoader = MMDataLoader(config)
            print(f"✓ Data loader created successfully for {dataset_name}")
            
            # 测试训练集
            train_dataset = dataLoader['train'].dataset
            print(f"  Train set size: {len(train_dataset)}")
            print(f"  Text shape: {train_dataset.text.shape}")
            print(f"  Audio shape: {train_dataset.audio.shape}")
            print(f"  Vision shape: {train_dataset.vision.shape}")
            print(f"  Labels shape: {len(train_dataset.labels['M'])}")
            
            # 测试验证集
            valid_dataset = dataLoader['valid'].dataset
            print(f"  Valid set size: {len(valid_dataset)}")
            
            # 测试测试集
            test_dataset = dataLoader['test'].dataset
            print(f"  Test set size: {len(test_dataset)}")
            
            # 测试数据加载
            for i, batch in enumerate(dataLoader['train']):
                print(f"  Batch {i} loaded successfully")
                print(f"    Batch text shape: {batch['text'].shape}")
                print(f"    Batch audio shape: {batch['audio'].shape}")
                print(f"    Batch vision shape: {batch['vision'].shape}")
                print(f"    Batch labels shape: {batch['labels']['M'].shape}")
                break  # 只测试一个批次
                
        except Exception as e:
            print(f"✗ Error loading {dataset_name}: {e}")

if __name__ == '__main__':
    main()
