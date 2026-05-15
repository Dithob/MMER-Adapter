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

# ── 1. Mixer 消融: None vs TGM vs AMM vs ATGFBFF vs SharedOffset ──
EXPERIMENT_GROUPS['mixer_ablation'] = {
    'description': 'Mixer层消融: None(Direct) vs TGM vs AMM vs ATGFBFF vs SharedOffset',
    'experiments': [
        {'name': 'mixer_none',       'use_tgm': False, 'use_amm': False, 'use_atgfbff': False, 'use_shared_offset': False},
        {'name': 'mixer_tgm',        'use_tgm': True,  'use_amm': False, 'use_atgfbff': False, 'use_shared_offset': False},
        {'name': 'mixer_amm',        'use_tgm': False, 'use_amm': True,  'use_atgfbff': False, 'use_shared_offset': False},
        {'name': 'mixer_atgfbff',    'use_tgm': False, 'use_amm': False, 'use_atgfbff': True,  'use_shared_offset': False},
        {'name': 'mixer_sharedoff',  'use_tgm': False, 'use_amm': False, 'use_atgfbff': False, 'use_shared_offset': True},
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

# ── 4b. LoRA target_modules 消融 ──
EXPERIMENT_GROUPS['lora_targets_ablation'] = {
    'description': 'LoRA target_modules扩展消融: qv / qkv / qkvo / qkvo+ffn',
    'experiments': [
        {'name': 'qv_only',    'use_lora': True, 'lora_r': 16, 'lora_alpha': 32,
         'lora_target_modules': 'q_proj,v_proj'},
        {'name': 'qkv',        'use_lora': True, 'lora_r': 16, 'lora_alpha': 32,
         'lora_target_modules': 'q_proj,k_proj,v_proj'},
        {'name': 'qkvo',       'use_lora': True, 'lora_r': 16, 'lora_alpha': 32,
         'lora_target_modules': 'q_proj,k_proj,v_proj,o_proj'},
        {'name': 'qkvo_ffn',   'use_lora': True, 'lora_r': 8,  'lora_alpha': 16,
         'lora_target_modules': 'q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj'},
    ]
}

# ── 4c. 两阶段训练消融 (LoRA warmup) ──
EXPERIMENT_GROUPS['two_stage_ablation'] = {
    'description': '两阶段训练 vs 单阶段 (LoRA warmup)',
    'experiments': [
        {'name': 'single_stage',  'use_lora': True, 'lora_r': 16, 'lora_alpha': 32,
         'lora_warmup_epochs': 0},
        {'name': 'warmup_3ep',    'use_lora': True, 'lora_r': 16, 'lora_alpha': 32,
         'lora_warmup_epochs': 3},
        {'name': 'warmup_5ep',    'use_lora': True, 'lora_r': 16, 'lora_alpha': 32,
         'lora_warmup_epochs': 5},
        {'name': 'warmup_8ep',    'use_lora': True, 'lora_r': 16, 'lora_alpha': 32,
         'lora_warmup_epochs': 8},
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
    'description': '完整消融: Mixer × Fusion 组合（含 ATGFBFF + MSLAF）',
    'experiments': [
        # Direct + Direct
        {'name': 'direct_direct', 'use_tgm': False, 'use_amm': False, 'use_atgfbff': False, 'use_msf': False, 'use_moe_fusion': False},
        # TGM + MSF
        {'name': 'tgm_msf',      'use_tgm': True,  'use_amm': False, 'use_atgfbff': False, 'use_msf': True,  'use_moe_fusion': False},
        # TGM + MoE
        {'name': 'tgm_moe',      'use_tgm': True,  'use_amm': False, 'use_atgfbff': False, 'use_msf': False, 'use_moe_fusion': True},
        # AMM + MSF
        {'name': 'amm_msf',      'use_tgm': False, 'use_amm': True,  'use_atgfbff': False, 'use_msf': True,  'use_moe_fusion': False},
        # AMM + MoE
        {'name': 'amm_moe',      'use_tgm': False, 'use_amm': True,  'use_atgfbff': False, 'use_msf': False, 'use_moe_fusion': True},
        # ATGFBFF (no MSLAF)
        {'name': 'atgfbff_base', 'use_tgm': False, 'use_amm': False, 'use_atgfbff': True,  'use_mslaf': False, 'use_msf': False, 'use_moe_fusion': False},
        # ATGFBFF + MSLAF (论文完整)
        {'name': 'atgfbff_mslaf', 'use_tgm': False, 'use_amm': False, 'use_atgfbff': True,  'use_mslaf': True, 'use_msf': False, 'use_moe_fusion': False},
    ]
}

# ── 10. ATGFB-MFF 组件逐步消融 ──
# 验证每个新增模块的独立贡献，从基线到论文完整复现
EXPERIMENT_GROUPS['atgfbff_ablation'] = {
    'description': 'ATGFB-MFF 逐步消融: 基线 → ATGFBFF → +MSLAF → +损失调优',
    'experiments': [
        # A0: 原始基线 (TGM + MSF，无 ATGFB 组件)
        {'name': 'A0_baseline_tgm_msf',
         'use_tgm': True, 'use_msf': True},

        # A1: ATGFBFF 不带辅助损失 (纯结构替换)
        {'name': 'A1_atgfbff_no_loss',
         'use_atgfbff': True, 'use_atgfbff_loss': False},

        # A2: ATGFBFF + 辅助损失 (α=0.6, β=0.2 — 之前的配置)
        {'name': 'A2_atgfbff_beta02',
         'use_atgfbff': True, 'use_atgfbff_loss': True,
         'alpha_align': 0.6, 'beta_fiber': 0.2},

        # A3: ATGFBFF + 辅助损失 (α=0.6, β=0.1 — 论文最优)
        {'name': 'A3_atgfbff_beta01',
         'use_atgfbff': True, 'use_atgfbff_loss': True,
         'alpha_align': 0.6, 'beta_fiber': 0.1},

        # A4: ATGFBFF + MSLAF (α=0.6, β=0.1 — 论文完整流程)
        {'name': 'A4_atgfbff_mslaf',
         'use_atgfbff': True, 'use_mslaf': True, 'use_atgfbff_loss': True,
         'alpha_align': 0.6, 'beta_fiber': 0.1},

        # A5: SharedOffset + MSLAF (对比替代方案)
        {'name': 'A5_sharedoff_mslaf',
         'use_shared_offset': True, 'use_mslaf': True,
         'use_shared_offset_loss': True, 'shared_offset_mode': 'gate',
         'alpha_align': 0.6, 'beta_offset': 0.1},
    ]
}

# ── 11. ATGFB-MFF 超参数敏感性分析 ──
# 参照论文 §5.3 对 α 和 β 进行网格搜索
EXPERIMENT_GROUPS['atgfbff_hyperparams'] = {
    'description': 'ATGFB-MFF 超参数分析: α∈{0.5,0.6,0.7,0.8} × β∈{0.1,0.15,0.2}',
    'experiments': [
        # α 扫描 (固定 β=0.1)
        {'name': 'hp_a05_b01', 'use_atgfbff': True, 'use_mslaf': True, 'use_atgfbff_loss': True,
         'alpha_align': 0.5, 'beta_fiber': 0.1},
        {'name': 'hp_a06_b01', 'use_atgfbff': True, 'use_mslaf': True, 'use_atgfbff_loss': True,
         'alpha_align': 0.6, 'beta_fiber': 0.1},
        {'name': 'hp_a07_b01', 'use_atgfbff': True, 'use_mslaf': True, 'use_atgfbff_loss': True,
         'alpha_align': 0.7, 'beta_fiber': 0.1},
        {'name': 'hp_a08_b01', 'use_atgfbff': True, 'use_mslaf': True, 'use_atgfbff_loss': True,
         'alpha_align': 0.8, 'beta_fiber': 0.1},
        # β 扫描 (固定 α=0.6)
        {'name': 'hp_a06_b015', 'use_atgfbff': True, 'use_mslaf': True, 'use_atgfbff_loss': True,
         'alpha_align': 0.6, 'beta_fiber': 0.15},
        {'name': 'hp_a06_b02', 'use_atgfbff': True, 'use_mslaf': True, 'use_atgfbff_loss': True,
         'alpha_align': 0.6, 'beta_fiber': 0.2},
    ]
}

# ── 12. ATGFB-MFF 完整对比 (带多 seed 验证) ──
# 用于论文级结果复现: 最佳配置 × 三 seed
EXPERIMENT_GROUPS['atgfbff_final'] = {
    'description': 'ATGFB-MFF 最终对比: 论文完整流程 vs 各 baseline（建议配合 --seeds 1234,2314,4321 使用）',
    'experiments': [
        # MSE-Adapter 原始基线
        {'name': 'final_tgm_msf',
         'use_tgm': True, 'use_msf': True},
        # ATGFBFF only (无 MSLAF)
        {'name': 'final_atgfbff_only',
         'use_atgfbff': True, 'use_atgfbff_loss': True,
         'alpha_align': 0.6, 'beta_fiber': 0.1},
        # 论文完整复现: ATGFBFF + MSLAF
        {'name': 'final_atgfbff_mslaf',
         'use_atgfbff': True, 'use_mslaf': True, 'use_atgfbff_loss': True,
         'alpha_align': 0.6, 'beta_fiber': 0.1},
    ]
}

# ── 13. 跨数据集验证 ──
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

# ── 14. 训练优化消融 (BiLSTM / Modality Dropout / Oversampling) ──
EXPERIMENT_GROUPS['training_tricks_ablation'] = {
    'description': '训练优化消融: BiLSTM × Modality Dropout × Oversampling',
    'experiments': [
        # T0: 纯基线
        {'name': 'T0_baseline',
         'use_tgm': True, 'use_msf': True},

        # T1: BiLSTM only
        {'name': 'T1_bilstm_only',
         'use_tgm': True, 'use_msf': True, 'use_bilstm': True},

        # T2~T4: Modality Dropout sweep
        {'name': 'T2_md_010',
         'use_tgm': True, 'use_msf': True, 'modality_dropout_p': 0.10},
        {'name': 'T3_md_015',
         'use_tgm': True, 'use_msf': True, 'modality_dropout_p': 0.15},
        {'name': 'T4_md_020',
         'use_tgm': True, 'use_msf': True, 'modality_dropout_p': 0.20},

        # T5a~T5c: Oversampling α sweep
        {'name': 'T5a_os_a05',
         'use_tgm': True, 'use_msf': True,
         'use_oversampling': True, 'oversampling_alpha': 0.5},
        {'name': 'T5b_os_a07',
         'use_tgm': True, 'use_msf': True,
         'use_oversampling': True, 'oversampling_alpha': 0.7},
        {'name': 'T5c_os_a10',
         'use_tgm': True, 'use_msf': True,
         'use_oversampling': True, 'oversampling_alpha': 1.0},

        # T6~T8: 两两组合 (oversampling 使用推荐 α=0.5)
        {'name': 'T6_md010_os',
         'use_tgm': True, 'use_msf': True,
         'modality_dropout_p': 0.10, 'use_oversampling': True, 'oversampling_alpha': 0.5},
        {'name': 'T7_bilstm_md010',
         'use_tgm': True, 'use_msf': True,
         'use_bilstm': True, 'modality_dropout_p': 0.10},
        {'name': 'T8_bilstm_os',
         'use_tgm': True, 'use_msf': True,
         'use_bilstm': True, 'use_oversampling': True, 'oversampling_alpha': 0.5},

        # T9: 全部组合
        {'name': 'T9_all_combined',
         'use_tgm': True, 'use_msf': True,
         'use_bilstm': True, 'modality_dropout_p': 0.10,
         'use_oversampling': True, 'oversampling_alpha': 0.5},
    ]
}

# ── 15. 训练优化 × Mixer 交叉验证 ──
EXPERIMENT_GROUPS['training_tricks_x_mixer'] = {
    'description': '训练优化最优组合 × Mixer 交叉验证',
    'experiments': [
        # TGM baseline vs best
        {'name': 'X0_tgm_baseline',
         'use_tgm': True, 'use_msf': True},
        {'name': 'X1_tgm_best',
         'use_tgm': True, 'use_msf': True,
         'use_bilstm': True, 'modality_dropout_p': 0.10, 'use_oversampling': True},

        # AMM baseline vs best
        {'name': 'X2_amm_baseline',
         'use_amm': True, 'use_msf': True},
        {'name': 'X3_amm_best',
         'use_amm': True, 'use_msf': True,
         'use_bilstm': True, 'modality_dropout_p': 0.10, 'use_oversampling': True},

        # ATGFBFF baseline vs best
        {'name': 'X4_atgfbff_baseline',
         'use_atgfbff': True, 'use_mslaf': True},
        {'name': 'X5_atgfbff_best',
         'use_atgfbff': True, 'use_mslaf': True,
         'use_bilstm': True, 'modality_dropout_p': 0.10, 'use_oversampling': True},
    ]
}

# ── 16. LLM 输出方式消融 (label_format + cls_head) ──
EXPERIMENT_GROUPS['label_mode_ablation'] = {
    'description': 'LLM输出方式消融: gen+index vs gen+text vs cls_head',
    'experiments': [
        # L0: 基线 (生成式 + 数字索引)
        {'name': 'L0_gen_index',
         'use_tgm': True, 'use_msf': True, 'label_format': 'index'},

        # L1: 生成式 + 文本标签
        {'name': 'L1_gen_text',
         'use_tgm': True, 'use_msf': True, 'label_format': 'text'},

        # L2: 分类头
        {'name': 'L2_cls_head',
         'use_tgm': True, 'use_msf': True, 'use_cls_head': True},
    ]
}

# ── 17. SD-MoE / AMM 模式消融 ──
# 固定 Fusion (SD-MoE)，验证 Mixer 层改进
EXPERIMENT_GROUPS['fusion_ablation'] = {
    'description': 'Mixer 消融: OriginAMM vs AMM-base vs H-AMM vs EP-AMM (Fusion 固定为 SD-MoE)',
    'experiments': [
        # M0: OriginAMM + SD-MoE ★
        {'name': 'M0_oamm_sdmoe',
         'use_origin_amm': True, 'use_msf': False, 'use_moe_fusion': False, 'use_sd_moe': True},

        # M1: 改进 AMM (base 模式) + SD-MoE
        {'name': 'M1_amm_base_sdmoe',
         'use_amm': True, 'amm_mode': 'base', 'use_sd_moe': True},

        # M2: H-AMM + SD-MoE ★ 推荐
        {'name': 'M2_hamm_sdmoe',
         'use_amm': True, 'amm_mode': 'hierarchical', 'use_sd_moe': True},

        # M3: EP-AMM (4 proto) + SD-MoE
        {'name': 'M3_epamm4_sdmoe',
         'use_amm': True, 'amm_mode': 'prototype', 'num_emotion_prototypes': 4, 'use_sd_moe': True},

        # M4: H-AMM + SD-MoE + TCAP 关闭 (验证 TCAP 必要性)
        {'name': 'M4_hamm_sdmoe_notcap',
         'use_amm': True, 'amm_mode': 'hierarchical', 'use_sd_moe': True, 'use_tcap': False},
    ]
}

# ── 19. v3 辅助 Loss 消融 ──
# 固定架构为 H-AMM + SD-MoE，逐项验证各辅助 Loss 的贡献
EXPERIMENT_GROUPS['v3_loss_ablation'] = {
    'description': 'v3 辅助 Loss 消融: H-AMM + SD-MoE 下逐项添加辅助损失',
    'experiments': [
        # L0: 纯 CE (关闭所有辅助 loss)
        {'name': 'L0_ce_only',
         'use_amm': True, 'amm_mode': 'hierarchical', 'use_sd_moe': True,
         'use_amm_align_loss': False},

        # L1: CE + AMM Align Loss (默认 α=0.5)
        {'name': 'L1_ce_align',
         'use_amm': True, 'amm_mode': 'hierarchical', 'use_sd_moe': True,
         'use_amm_align_loss': True, 'alpha_amm': 0.5},

        # L2: CE + AMM Align (α=0.3, 较弱)
        {'name': 'L2_ce_align_a03',
         'use_amm': True, 'amm_mode': 'hierarchical', 'use_sd_moe': True,
         'use_amm_align_loss': True, 'alpha_amm': 0.3},

        # L3: CE + AMM Align (α=0.8, 较强)
        {'name': 'L3_ce_align_a08',
         'use_amm': True, 'amm_mode': 'hierarchical', 'use_sd_moe': True,
         'use_amm_align_loss': True, 'alpha_amm': 0.8},

        # L4: CE + Align + MoE LB Loss
        {'name': 'L4_align_lb',
         'use_amm': True, 'amm_mode': 'hierarchical', 'use_sd_moe': True,
         'use_amm_align_loss': True, 'use_moe_lb_loss': True},

        # L5: CE + Align + DiffLoss (Global/Local 正交)
        {'name': 'L5_align_diff',
         'use_amm': True, 'amm_mode': 'hierarchical', 'use_sd_moe': True,
         'use_amm_align_loss': True, 'use_diff_loss': True},

        # L6: CE + Align + ExpertDiffLoss (Expert 分化)
        {'name': 'L6_align_expert_diff',
         'use_amm': True, 'amm_mode': 'hierarchical', 'use_sd_moe': True,
         'use_amm_align_loss': True, 'use_expert_diff_loss': True},

        # L7: CE + Align + NCE (跨模态对比)
        {'name': 'L7_align_nce',
         'use_amm': True, 'amm_mode': 'hierarchical', 'use_sd_moe': True,
         'use_amm_align_loss': True, 'use_nce_loss': True},

        # L8: CE + Align + LB + Diff (组合)
        {'name': 'L8_align_lb_diff',
         'use_amm': True, 'amm_mode': 'hierarchical', 'use_sd_moe': True,
         'use_amm_align_loss': True, 'use_moe_lb_loss': True, 'use_diff_loss': True},

        # L9: 全部辅助 Loss
        {'name': 'L9_all_aux',
         'use_amm': True, 'amm_mode': 'hierarchical', 'use_sd_moe': True,
         'use_amm_align_loss': True, 'use_moe_lb_loss': True,
         'use_diff_loss': True, 'use_expert_diff_loss': True, 'use_nce_loss': True},
    ]
}

# ── 20. v3 完整交叉验证 (Mixer × Fusion × Loss) ──
# 最终论文级结果: 最佳配置 + 各关键 baseline
EXPERIMENT_GROUPS['v3_final'] = {
    'description': 'v3 完整对比: 新旧架构 × Loss 最优组合 (建议配合 --seeds 1234,2314,3124)',
    'experiments': [
        # F0: 旧基线 — TGM + MSF
        {'name': 'F0_tgm_msf',
         'use_tgm': True, 'use_msf': True},

        # F1: 旧最佳 — OriginAMM + MSF
        {'name': 'F1_oamm_msf',
         'use_origin_amm': True, 'use_msf': True},

        # F2: OriginAMM + Origin MoE (旧 MoE 对照)
        {'name': 'F2_oamm_origin_moe',
         'use_origin_amm': True, 'use_moe_fusion': True},

        # F3: OriginAMM + SD-MoE (仅 Fusion 改进)
        {'name': 'F3_oamm_sdmoe',
         'use_origin_amm': True, 'use_sd_moe': True},

        # F4: H-AMM + MSF (仅 Mixer 改进)
        {'name': 'F4_hamm_msf',
         'use_amm': True, 'amm_mode': 'hierarchical', 'use_msf': True},

        # F5: H-AMM + SD-MoE (完整 v3) ★
        {'name': 'F5_hamm_sdmoe',
         'use_amm': True, 'amm_mode': 'hierarchical', 'use_sd_moe': True},

        # F6: EP-AMM + SD-MoE
        {'name': 'F6_epamm_sdmoe',
         'use_amm': True, 'amm_mode': 'prototype', 'use_sd_moe': True},

        # F7: H-AMM + SD-MoE + Gate
        {'name': 'F7_hamm_sdmoe_gate',
         'use_amm': True, 'amm_mode': 'hierarchical', 'use_sd_moe': True, 'use_gate': True},
    ]
}

# ═══════════════════════════════════════════════════════════════════
# 命令构建
# ═══════════════════════════════════════════════════════════════════

# run.py 中 action='store_true' 的参数列表
BOOLEAN_FLAGS = {
    'use_tgm', 'use_amm', 'use_origin_amm',
    'use_atgfbff', 'use_atgfbff_loss', 'use_shared_offset', 'use_shared_offset_loss',
    'use_mslaf',
    'use_msf', 'use_moe_fusion', 'use_sd_moe',
    'use_gate', 'use_moe_lb_loss',
    'use_diff_loss', 'use_expert_diff_loss',
    'use_nce_loss',
    'use_amm_align_loss', 'use_tcap',
    'use_lora', 'use_int8',
    'use_bilstm', 'use_oversampling',
    'use_cls_head',
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
    parser.add_argument('--meld_feature_mode', type=str, default=None,
                        help='MELD 特征模式: raw / processed (默认使用 run.py 配置)')

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
    if args.meld_feature_mode:
        base_args['meld_feature_mode'] = args.meld_feature_mode

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
