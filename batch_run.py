#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
batch_run.py — 批量消融实验调度器

用法:
  1. 直接运行预定义的消融组:
     python batch_run.py --group mixer_ablation

  2. 只运行指定编号的实验 (0-indexed, 逗号分隔):
     python batch_run.py --group full_ablation --only 0,3,5

  3. 从某个编号开始继续 (跳过已完成的):
     python batch_run.py --group full_ablation --start_from 3

  4. 预览所有命令但不执行:
     python batch_run.py --group full_ablation --dry_run

  5. 自定义基础参数:
     python batch_run.py --group mixer_ablation --model_type qwen3.5 --dataset meld --seeds 1111,2222

所有实验结果自动记录到同一 CSV 文件中，通过 param_columns 区分不同配置。
"""

import os
import sys
import time
import subprocess
import argparse
from collections import OrderedDict


# ═══════════════════════════════════════════════════════════════════
# 消融实验组定义
# 每个实验是一个 dict，key = run.py 参数名，value = 参数值
# 布尔开关（action='store_true'）用 True/False 表示是否添加
# ═══════════════════════════════════════════════════════════════════

EXPERIMENT_GROUPS = OrderedDict()

# ── 1. Mixer 消融: None vs TGM vs AMM ──
EXPERIMENT_GROUPS['mixer_ablation'] = {
    'description': 'Mixer层消融: None(Direct) vs TGM vs AMM',
    'experiments': [
        {'name': 'mixer_none',   'use_tgm': False, 'use_amm': False},
        {'name': 'mixer_tgm',    'use_tgm': True,  'use_amm': False},
        {'name': 'mixer_amm',    'use_tgm': False, 'use_amm': True},
    ]
}

# ── 2. Fusion 消融: Direct vs MSF vs MoE ──
EXPERIMENT_GROUPS['fusion_ablation'] = {
    'description': 'Fusion层消融: Direct vs MSF vs Dual-MoE',
    'experiments': [
        {'name': 'fusion_direct',  'use_msf': False, 'use_moe_fusion': False},
        {'name': 'fusion_msf',     'use_msf': True,  'use_moe_fusion': False},
        {'name': 'fusion_moe',     'use_msf': False, 'use_moe_fusion': True},
    ]
}

# ── 3. 模态消融: tav/ta/tv/av/t/a/v ──
EXPERIMENT_GROUPS['modality_ablation'] = {
    'description': '模态消融: 7种模态组合',
    'experiments': [
        {'name': 'mod_tav', 'modalities': 'tav'},
        {'name': 'mod_ta',  'modalities': 'ta'},
        {'name': 'mod_tv',  'modalities': 'tv'},
        {'name': 'mod_av',  'modalities': 'av'},
        {'name': 'mod_t',   'modalities': 't'},
        {'name': 'mod_a',   'modalities': 'a'},
        {'name': 'mod_v',   'modalities': 'v'},
    ]
}

# ── 4. LoRA 消融 ──
EXPERIMENT_GROUPS['lora_ablation'] = {
    'description': 'LoRA消融: 冻结LLM vs LoRA r=8/16/32',
    'experiments': [
        {'name': 'no_lora',     'use_lora': False},
        {'name': 'lora_r8',     'use_lora': True, 'lora_r': 8,  'lora_alpha': 16},
        {'name': 'lora_r16',    'use_lora': True, 'lora_r': 16, 'lora_alpha': 32},
        {'name': 'lora_r32',    'use_lora': True, 'lora_r': 32, 'lora_alpha': 64},
    ]
}

# ── 5. Raw AV Token Bypass 消融 ──
EXPERIMENT_GROUPS['raw_av_ablation'] = {
    'description': 'AV Token旁路消融: none vs audio vs video vs both',
    'experiments': [
        {'name': 'av_none',    'raw_av_mode': 'none'},
        {'name': 'av_audio',   'raw_av_mode': 'audio', 'av_pseudo_tokens': 4},
        {'name': 'av_video',   'raw_av_mode': 'video', 'av_pseudo_tokens': 4},
        {'name': 'av_both_4',  'raw_av_mode': 'both',  'av_pseudo_tokens': 4},
        {'name': 'av_both_8',  'raw_av_mode': 'both',  'av_pseudo_tokens': 8},
    ]
}

# ── 6. Prompt Style 消融 ──
EXPERIMENT_GROUPS['prompt_ablation'] = {
    'description': '提示词模板消融: default vs enhanced',
    'experiments': [
        {'name': 'prompt_default',   'prompt_style': 'default'},
        {'name': 'prompt_enhanced',  'prompt_style': 'enhanced'},
    ]
}

# ── 7. 辅助损失消融 ──
EXPERIMENT_GROUPS['aux_loss_ablation'] = {
    'description': '辅助损失消融: DiffLoss / NCE / 全部',
    'experiments': [
        {'name': 'no_aux',       'use_diff_loss': False, 'use_nce_loss': False},
        {'name': 'diff_only',    'use_diff_loss': True,  'use_nce_loss': False},
        {'name': 'nce_only',     'use_diff_loss': False, 'use_nce_loss': True},
        {'name': 'diff_and_nce', 'use_diff_loss': True,  'use_nce_loss': True},
    ]
}

# ── 8. MoE Gate & Expert配置消融 ──
EXPERIMENT_GROUPS['moe_detail_ablation'] = {
    'description': 'MoE内部配置消融 (需先启用 --use_moe_fusion)',
    'experiments': [
        {'name': 'moe_no_gate',        'use_moe_fusion': True, 'use_gate': False, 'num_local_experts': 3},
        {'name': 'moe_with_gate',       'use_moe_fusion': True, 'use_gate': True,  'num_local_experts': 3},
        {'name': 'moe_5experts',        'use_moe_fusion': True, 'use_gate': True,  'num_local_experts': 5},
        {'name': 'moe_expert_diff',     'use_moe_fusion': True, 'use_gate': True,  'use_expert_diff_loss': True},
    ]
}

# ── 9. 完整消融（最佳配置搜索）──
EXPERIMENT_GROUPS['full_ablation'] = {
    'description': '完整消融: Mixer × Fusion 组合',
    'experiments': [
        # Direct + Direct
        {'name': 'direct_direct', 'use_tgm': False, 'use_amm': False, 'use_msf': False, 'use_moe_fusion': False},
        # TGM + MSF
        {'name': 'tgm_msf',      'use_tgm': True,  'use_amm': False, 'use_msf': True,  'use_moe_fusion': False},
        # TGM + MoE
        {'name': 'tgm_moe',      'use_tgm': True,  'use_amm': False, 'use_msf': False, 'use_moe_fusion': True},
        # AMM + MSF
        {'name': 'amm_msf',      'use_tgm': False, 'use_amm': True,  'use_msf': True,  'use_moe_fusion': False},
        # AMM + MoE
        {'name': 'amm_moe',      'use_tgm': False, 'use_amm': True,  'use_msf': False, 'use_moe_fusion': True},
        # AMM + MoE + LoRA
        {'name': 'amm_moe_lora', 'use_tgm': False, 'use_amm': True,  'use_msf': False, 'use_moe_fusion': True, 'use_lora': True},
    ]
}

# ── 10. 跨数据集验证 ──
EXPERIMENT_GROUPS['cross_dataset'] = {
    'description': '跨数据集验证: 使用相同配置在多个数据集上测试',
    'experiments': [
        {'name': 'ds_meld',     'datasetName': 'meld'},
        {'name': 'ds_iemocap4', 'datasetName': 'iemocap4'},
        {'name': 'ds_cherma',   'datasetName': 'cherma'},
        {'name': 'ds_mosei',    'datasetName': 'mosei'},
        {'name': 'ds_simsv2',   'datasetName': 'simsv2'},
    ]
}


# ═══════════════════════════════════════════════════════════════════
# 命令构建
# ═══════════════════════════════════════════════════════════════════

# run.py 中 action='store_true' 的参数列表
BOOLEAN_FLAGS = {
    'use_tgm', 'use_amm', 'use_msf', 'use_moe_fusion',
    'use_gate', 'use_moe_lb_loss',
    'use_diff_loss', 'use_expert_diff_loss',
    'use_nce_loss',
    'use_lora',
    'eval_only',
}


def build_command(exp_config, base_args):
    """将实验配置 + 基础参数合并成完整的命令行"""
    cmd = [sys.executable, 'run.py']

    # 合并基础参数（base_args 被实验覆盖）
    merged = {**base_args}
    for k, v in exp_config.items():
        if k == 'name':
            continue  # 'name' 只是标识，不是参数
        merged[k] = v

    # 构建命令行
    for key, value in merged.items():
        arg_name = f'--{key}'
        if key in BOOLEAN_FLAGS:
            if value:  # True → 添加 flag
                cmd.append(arg_name)
            # False → 不添加 (使用 argparse 默认)
        else:
            cmd.append(arg_name)
            cmd.append(str(value))

    return cmd


# ═══════════════════════════════════════════════════════════════════
# 主调度逻辑
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='HMMEM 批量消融实验调度器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='\n'.join([
            '可用的实验组:',
            *[f'  {name:25s} — {group["description"]}'
              for name, group in EXPERIMENT_GROUPS.items()],
            '',
            '示例:',
            '  python batch_run.py --group mixer_ablation --dataset meld',
            '  python batch_run.py --group full_ablation --only 0,2,4 --dry_run',
            '  python batch_run.py --group modality_ablation --start_from 3',
        ])
    )

    # 调度参数
    parser.add_argument('--group', type=str, required=True,
                        choices=list(EXPERIMENT_GROUPS.keys()),
                        help='选择实验组')
    parser.add_argument('--dry_run', action='store_true',
                        help='只打印命令，不执行')
    parser.add_argument('--only', type=str, default=None,
                        help='只运行指定编号的实验 (0-indexed, 逗号分隔, e.g. 0,2,4)')
    parser.add_argument('--start_from', type=int, default=0,
                        help='从第N个实验开始 (跳过之前的)')

    # 基础参数（可覆盖 run.py 默认值）
    parser.add_argument('--model_type', type=str, default='qwen3.5',
                        help='LLM类型 (default: qwen3.5)')
    parser.add_argument('--dataset', type=str, default='meld',
                        help='数据集名称 (default: meld)')
    parser.add_argument('--seeds', type=str, default='1111',
                        help='随机种子 (default: 1111，多个用逗号分隔)')
    parser.add_argument('--pretrain_LM', type=str, default=None,
                        help='LLM路径 (默认根据 model_type 自动选择)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='batch size (默认使用 run.py 的配置文件值)')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='学习率 (默认使用 run.py 的配置文件值)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers (default: 4)')
    parser.add_argument('--gpu_ids', type=str, default='',
                        help='GPU ID (default: 自动选择)')

    args = parser.parse_args()

    # 构建基础参数字典
    base_args = OrderedDict()
    base_args['modelName'] = 'hmmem'
    base_args['model_type'] = args.model_type
    base_args['datasetName'] = args.dataset
    base_args['seeds'] = args.seeds
    base_args['num_workers'] = args.num_workers
    if args.gpu_ids:
        base_args['gpu_ids'] = args.gpu_ids
    if args.pretrain_LM:
        base_args['pretrain_LM'] = args.pretrain_LM
    if args.batch_size:
        base_args['batch_size'] = args.batch_size
    if args.learning_rate:
        base_args['learning_rate'] = args.learning_rate

    # 获取实验组
    group = EXPERIMENT_GROUPS[args.group]
    experiments = group['experiments']

    # 筛选要运行的实验
    if args.only is not None:
        indices = [int(x.strip()) for x in args.only.split(',')]
        experiments = [(i, experiments[i]) for i in indices if i < len(experiments)]
    else:
        experiments = [(i, exp) for i, exp in enumerate(experiments) if i >= args.start_from]

    # 打印计划
    total = len(experiments)
    print('=' * 72)
    print(f'  HMMEM 批量消融实验')
    print(f'  实验组: {args.group} — {group["description"]}')
    print(f'  共 {total} 个实验 | 基础模型: {args.model_type} | 数据集: {args.dataset}')
    print(f'  种子: {args.seeds}')
    print('=' * 72)

    for idx, (orig_idx, exp) in enumerate(experiments):
        cmd = build_command(exp, base_args)
        cmd_str = ' '.join(cmd)
        marker = '[DRY RUN] ' if args.dry_run else ''
        print(f'\n  [{orig_idx}] {marker}{exp["name"]}')
        print(f'      {cmd_str}')

    if args.dry_run:
        print(f'\n{"=" * 72}')
        print(f'  DRY RUN 完成 — 以上命令未执行')
        print(f'{"=" * 72}')
        return

    print(f'\n{"─" * 72}')
    print(f'  即将开始执行...')
    print(f'{"─" * 72}\n')

    # 执行
    results = []
    for idx, (orig_idx, exp) in enumerate(experiments):
        exp_name = exp['name']
        cmd = build_command(exp, base_args)
        cmd_str = ' '.join(cmd)

        print(f'\n{"█" * 72}')
        print(f'  [{idx + 1}/{total}] 实验: {exp_name}')
        print(f'  命令: {cmd_str}')
        print(f'{"█" * 72}\n')

        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                cwd=os.path.dirname(os.path.abspath(__file__)),
                check=True,
            )
            elapsed = time.time() - start_time
            status = '✅ 成功'
            results.append((exp_name, status, elapsed))
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - start_time
            status = f'❌ 失败 (exit code {e.returncode})'
            results.append((exp_name, status, elapsed))
            print(f'\n⚠️  实验 {exp_name} 失败，继续下一个...\n')
        except KeyboardInterrupt:
            elapsed = time.time() - start_time
            results.append((exp_name, '⏹ 中断', elapsed))
            print(f'\n\n⏹  用户中断，终止批量执行。')
            break

    # 汇总
    print(f'\n\n{"═" * 72}')
    print(f'  批量实验完成汇总')
    print(f'{"═" * 72}')
    print(f'  {"编号":<6} {"实验名称":<25} {"状态":<15} {"耗时":>10}')
    print(f'  {"─" * 60}')
    total_time = 0
    for i, (name, status, elapsed) in enumerate(results):
        total_time += elapsed
        t_str = f'{elapsed / 60:.1f} min' if elapsed >= 60 else f'{elapsed:.0f} sec'
        print(f'  {i:<6} {name:<25} {status:<15} {t_str:>10}')
    print(f'  {"─" * 60}')
    total_t = f'{total_time / 60:.1f} min' if total_time >= 60 else f'{total_time:.0f} sec'
    print(f'  {"总计":>31} {len(results)} 个实验 {total_t:>12}')
    print(f'{"═" * 72}\n')


if __name__ == '__main__':
    main()
