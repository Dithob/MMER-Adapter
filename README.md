# HMMEM: Hierarchical MoE Multimodal ERC Model

## 项目概述

HMMEM 是一个统一的多模态情感识别框架，支持多种大语言模型后端适配。该项目通过可插拔的 Mixer-Fusion 两阶段架构，实现了对文本、音频和视频三种模态信息的灵活整合与消融实验。

## 项目结构

```
MMER-Adapter/
├── config/                        # 配置文件
│   ├── config_classification.py   # 分类任务配置
│   └── config_regression.py       # 回归任务配置
├── data/                          # 数据处理
│   ├── DataPre.py                 # 数据预处理
│   ├── TextPre.py                 # 文本预处理
│   └── load_data.py               # 数据加载器
├── models/                        # 模型定义
│   ├── AMIO.py                    # 模型入口包装器
│   ├── text_modules/              # 统一的语言模型加载器
│   └── multiTask/                 # HMMEM 核心模块
│       ├── HMMEM.py               # 主模型 (forward/generate)
│       ├── HMMEM_mixer.py         # Mixer 层: AMM (Adaptive Modal Mixer)
│       ├── HMMEM_moe.py           # Fusion 层: GlobalMoE / LocalMoE
│       ├── HMMEM_loss.py          # 辅助损失: DiffLoss / NCE (CPC)
│       └── HMMEM_modules.py       # 基础组件: LSTM / TGM / MSF / FeatureAdapter / ATGFBFF / SharedOffset
├── trains/                        # 训练逻辑
│   └── ATIO.py                    # 训练 / 测试路由
├── run.py                         # 主入口脚本
└── README.md
```

## 重构亮点

### 1. 统一的语言模型加载器

在 `models/text_modules/` 中，我们创建了一个统一的语言模型加载器，支持：

- **ChatGLM3-6B**
- **Qwen-1.8B / Qwen3.5**
- **Llama2-7B**
- **DeepSeek**
- **Gemma**

### 2. 模型与配置统一入口

通过 `--model_type` 指定语言模型，通过配置文件统一管理回归 / 分类任务：

- 回归任务：MOSEI、SIMSV2、MOSI、SIMS
- 分类任务：MELD、CHERMA、IEMOCAP（4 分类和 6 分类）

### 3. 代码结构统一

当前项目保留 HMMEM 作为架构名，并将多模态编码、融合、损失与训练流程统一到一套可插拔实现中，便于维护和消融实验。

## 新增模块说明

在原始 HMMEM 的 Mixer-Fusion 两阶段架构基础上，当前版本新增了以下模块：

- **ATGFBFF**：共享语义与偏移建模的融合模块，输出 pseudo tokens 供 LLM 使用
- **SharedOffsetFusion**：ATGFBFF 的简化过渡版本，支持 `add / gate / residual`
- **Dual-Branch MoE**：GlobalMoE + LocalMoE + Meta-Gate 的双分支融合
- **FeatureAdapter**：用于高维音频 / 视频特征的降维适配
- **Raw AV Token Bypass**：支持 `raw_av_mode`，可将原始音频 / 视频 token 直接注入 LLM 输入

这些模块都支持单独开关，方便进行消融对比。

## 使用方法

### 基本用法

```bash
# 使用ChatGLM3-6B模型
python run.py --model_type chatglm3 --datasetName mosei --pretrain_LM /path/to/chatglm3-6b-base/

# 使用Qwen-1.8B模型
python run.py --model_type qwen --datasetName mosei --pretrain_LM /path/to/qwen-1.8b/

# 使用Llama2-7B模型
python run.py --model_type llama2 --datasetName mosei --pretrain_LM /path/to/llama2-7b/
```

### 参数说明

**基础参数：**

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--model_type` | 语言模型类型 | `chatglm3` |
| `--datasetName` | 数据集名称 | `mosi` |
| `--pretrain_LM` | 预训练语言模型路径 | 自动推断 |
| `--seeds` | 随机种子列表 | `1111,2222,3333,4444,5555` |

**Mixer 层消融（互斥）：**

| 参数 | 说明 | 默认 |
|---|---|---|
| `--use_tgm` | 使用 Text-Guided Mixer（原版基线） | True |
| `--use_amm` | 使用 Adaptive Modal Mixer（覆盖 TGM） | False |
| `--use_atgfbff` | 使用 ATGFBFF（覆盖 TGM） | False |
| `--use_shared_offset` | 使用 SharedOffsetFusion（覆盖 TGM） | False |
| `--shared_offset_mode` | SharedOffset 的融合方式：`add` / `gate` / `residual` | `gate` |

**Fusion 层消融（互斥）：**

| 参数 | 说明 | 默认 |
|---|---|---|
| `--use_msf` | 使用 Multi-Scale Fusion（原版基线） | True |
| `--use_moe_fusion` | 使用 Dual-Branch MoE（覆盖 MSF） | False |

**ATGFBFF / SharedOffset 辅助损失：**

| 参数 | 说明 | 默认 |
|---|---|---|
| `--use_atgfbff_loss` | 启用 ATGFBFF 的对齐 / fiber 辅助损失 | True |
| `--use_shared_offset_loss` | 启用 SharedOffset 的对齐 / 偏移正则 | True |
| `--alpha_align` | 对齐损失权重 | 0.6 |
| `--beta_fiber` | fiber 正则权重 | 0.1 |
| `--beta_offset` | offset 正则权重 | 0.1 |
| `--num_latents` | ATGFBFF 多尺度 latent 数量 | 4 |

**MoE 配置：**

| 参数 | 说明 | 默认 |
|---|---|---|
| `--use_gate` | Meta-Gate 使用 cosine bias | False |
| `--use_moe_lb_loss` | MoE 负载均衡损失 | False |
| `--num_local_experts` | Local MoE 专家数量 | 3 |
| `--expert_bottleneck` | Local Expert 瓶颈维度 | 64 |

**辅助损失：**

| 参数 | 说明 | 默认 |
|---|---|---|
| `--use_diff_loss` | 分支间 DiffLoss | False |
| `--use_expert_diff_loss` | Expert 间 DiffLoss | False |
| `--use_nce_loss` | 跨模态 NCE 损失 | False |
| `--nce_weight` | NCE 损失权重 | 0.05 |

**高维特征适配：**

| 参数 | 说明 | 默认 |
|---|---|---|
| `--adapter_dim` | FeatureAdapter 输出维度（仅当特征维度 > adapter_dim 时激活） | 128 |
| `--audio_adapter_dim` | 音频单独适配维度 | None |
| `--video_adapter_dim` | 视频单独适配维度 | None |

**LLM / 输入增强：**

| 参数 | 说明 | 默认 |
|---|---|---|
| `--raw_av_mode` | 原始 AV token 注入模式 | `none` |
| `--av_pseudo_tokens` | 原始 AV pseudo token 数量 | `4` |
| `--prompt_style` | 多模态 prompt 风格 | `default` |

### 示例

```bash
# 在MELD数据集上训练 (默认: TGM + MSF)
python run.py --model_type chatglm3 --datasetName meld --seeds 1111,2222,3333

# 在IEMOCAP数据集上训练（6分类）- 启用MoE
python run.py --model_type chatglm3 --datasetName iemocap6 --use_moe_fusion

# 启用 ATGFBFF 融合
python run.py --model_type qwen3.5 --datasetName meld --use_atgfbff

# 启用 SharedOffset 过渡版本
python run.py --model_type qwen3.5 --datasetName meld --use_shared_offset --shared_offset_mode gate
```

## 支持的数据集

### 回归任务

- **MOSEI**: 多模态情感强度预测，范围\[-3.0, 3.0]
- **SIMSV2**: 中文多模态情感强度预测，范围\[-1.0, 1.0]

### 分类任务

- **MELD**: 英文多模态情感分类，7类情感
- **CHERMA**: 中文多模态情感分类，7类情感
- **IEMOCAP**: 英文多模态情感分类
  - **iemocap4**: 4类情感（angry, happy, sad, neutral）
  - **iemocap6**: 6类情感（angry, happy, excited, sad, neutral, frustrated）

## 模型架构 (HMMEM)

### 信号流（通用路径）

```
audio → [FeatureAdapter] → LSTM → audio_h ─┐
video → [FeatureAdapter] → LSTM → video_h ─┤──→ 【Mixer 层】──→ feature_f ──→ 【Fusion 层】──→ fusion_h ──→ LLM
text  → LLM Embedding → text_embed ────────┘
```

### 信号流（ATGFBFF / SharedOffset 专用路径）

```
audio → [FeatureAdapter] → LSTM → audio_h ─────────────────────┐
video → [FeatureAdapter] → LSTM → video_h ─────────────────────┤
text  → LLM Embed → GAP Pool → Linear(text_in→256) → text_h ──┘
                                                                ↓
                                              ATGFBFF / SharedOffset (Mixer)
                                                                ↓
                                              [B, pt, 256] → Linear(256→text_in)
                                                                ↓
                                              fusion_h [B, pt, text_in] → LLM
```

### Mixer 层（模态交互）

| 模块 | 说明 |
|---|---|
| **TGM** (Text-Guided Mixer) | 文本主导：GAP 池化文本 → 调制音频/视频 → 融合。基线方案。 |
| **AMM** (Adaptive Modal Mixer) | 三模态对等：音频↔视频直接交互，再进行自适应加权。 |
| **ATGFBFF** | 共享语义 + 偏移建模的融合模块，文本池化后与音频 / 视频共同参与融合。 |
| **SharedOffsetFusion** | 共享空间 + 偏移空间的过渡版本，可通过 `add/gate/residual` 调整融合方式。 |
| **Lightweight_mixer** | 简化版 AV 融合，仅在缺少文本或轻量场景下使用。 |

### Fusion 层（特征映射）

| 模块 | 说明 |
|---|---|
| **MSF** (Multi-Scale Fusion) | 原始多尺度融合模块，基线方案。 |
| **Dual-Branch MoE** | Global MoE + Local MoE 的双分支融合，Meta-Gate 根据分支差异进行加权。 |
| **Direct Projection** | 当未启用 MSF / MoE 时的兜底映射方式。 |

### 辅助优化机制

| 模块 | 说明 | 依赖 |
|---|---|---|
| **LB Loss** | Expert 负载均衡 | `use_moe_fusion` |
| **DiffLoss** | 分支 / Expert 间正交互补 | `use_moe_fusion` |
| **NCE Loss** | 跨模态时序对比学习 | `use_nce_loss` |
| **ATGFBFF Align/Fiber Loss** | 对齐共享语义并约束偏移 | `use_atgfbff` |
| **SharedOffset Align/Offset Loss** | 对齐共享表征并约束 offset 分量 | `use_shared_offset` |

### 高维特征适配

| 模块 | 说明 |
|---|---|
| **FeatureAdapter** | 当输入特征维度 > `adapter_dim` 时自动启用的轻量降维层，支持 HuBERT / Whisper 等高维 Encoder。 |

## 消融实验命令参考

以 MELD 数据集为例，基础命令前缀：

```bash
BASE="python run.py --model_type chatglm3 --datasetName meld \
  --pretrain_LM /root/autodl-tmp/models/chatglm3-6b-base \
  --seeds 1234,2314,4321"
```

### 1. Mixer 层消融

```bash
# 基线：TGM + MSF
$BASE

# AMM 替换 TGM
$BASE --use_amm

# ATGFBFF 替换 TGM
$BASE --use_atgfbff

# SharedOffset 替换 TGM
$BASE --use_shared_offset --shared_offset_mode gate
```

### 2. Fusion 层消融

```bash
# 基线：MSF
$BASE

# MoE 替换 MSF
$BASE --use_moe_fusion

# MoE + Gate
$BASE --use_moe_fusion --use_gate

# MoE + DiffLoss
$BASE --use_moe_fusion --use_diff_loss
```

### 3. ATGFBFF / SharedOffset 相关消融

```bash
# ATGFBFF 不启用辅助损失
$BASE --use_atgfbff

# ATGFBFF + 辅助损失
$BASE --use_atgfbff --use_atgfbff_loss --alpha_align 0.6 --beta_fiber 0.2

# SharedOffset + 辅助损失
$BASE --use_shared_offset --use_shared_offset_loss --shared_offset_mode gate
```

### 4. 模态与输入增强消融

```bash
# 仅文本
$BASE --modalities t

# 文本 + 音频
$BASE --modalities ta

# 文本 + 视频
$BASE --modalities tv

# 原始 AV bypass
$BASE --raw_av_mode both
```

## 技术特点

1. **两阶段可插拔架构**: Mixer 与 Fusion 解耦，每层都支持多种实现，便于替换和消融。
2. **双分支 MoE 设计**: Global / Local 两路专家协同建模，Meta-Gate 根据分支差异进行加权。
3. **多种 Mixer 方案**: 支持 TGM、AMM、ATGFBFF、SharedOffset 等不同模态交互方式。
4. **高维 Encoder 适配**: FeatureAdapter 自动按需启用，支持高维音频 / 视频特征降维。
5. **原始 AV 输入增强**: 支持 raw AV bypass，保留更多局部模态信息。
6. **实验工程化完整**: 支持多 seed、多数据集、断点续训、仅评估模式和结果汇总。

## 依赖项

- PyTorch
- Transformers
- ModelScope
- scikit-learn
- pandas
- numpy
- tqdm

## 结果保存

训练结果将保存在以下位置：

- **模型文件**: `{model_save_dir}/{modelName}-{model_type}-{datasetName}-{train_mode}-{timestamp}.pth`
- **实验结果**: `{res_save_dir}/{modelName}-{model_type}-{datasetName}-{train_mode}.csv`
- **日志文件**: `logs/{modelName}-{datasetName}-{model_type}.log`

## 注意事项

1. 确保已安装所需的依赖项
2. 根据使用的语言模型，正确设置 `--pretrain_LM` 路径
3. 确保数据集路径正确配置
4. 根据GPU内存情况调整 `batch_size` 参数
5. `--use_amm`、`--use_atgfbff`、`--use_shared_offset` 会覆盖默认的 TGM
6. `--use_moe_fusion` 会覆盖 MSF

## 迁移指南

如果您之前使用的是原始 HMMEM 版本，迁移到当前框架主要有以下变化：

1. 使用新的统一项目结构
2. 在运行命令中通过 `--model_type` 指定 LLM 类型
3. 如需启用新增模块，再按需添加 `--use_atgfbff`、`--use_shared_offset`、`--use_moe_fusion` 等开关
4. 其余基础训练参数保持兼容

例如：

```bash
python run.py --model_type chatglm3 --datasetName mosei
```

## IEMOCAP数据集动态分类配置

HMMEM 框架支持 IEMOCAP 数据集的 4 分类和 6 分类任务，通过以下方式实现动态配置：

### 1. 数据集名称映射

- `iemocap4`: 4类情感分类（angry, happy, sad, neutral）
- `iemocap6`: 6类情感分类（angry, happy, excited, sad, neutral, frustrated）

### 2. 动态配置实现

框架会根据数据集名称自动：

- 设置相应的分类数量（4或6）
- 选择对应的提示词（task\_specific\_prompt）
- 选择对应的标签映射（label\_index\_mapping）
- 选择对应的情感映射（emo\_map4或emo\_map6）

### 3. 使用方法

使用 `run.py` 脚本训练IEMOCAP数据集：

```bash
# 训练IEMOCAP 4分类
python run.py --model_type chatglm3 --datasetName iemocap4

# 训练IEMOCAP 6分类
python run.py --model_type qwen --datasetName iemocap6

# 查看脚本帮助
python run.py --help
```

### 4. 数据加载

框架支持从以下位置加载IEMOCAP数据集的文本标签：

- `d:\ProjectFiles\exp_202603\datasets\text_data\iemocap_text`（优先）
- 其他常见路径（自动检测）

如果找不到文本标签，框架会自动生成虚拟数据用于测试。

## 许可证

请参考原项目的LICENSE文件。

## 贡献

欢迎提交问题和改进建议！
