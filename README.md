# HMMEM: Hierarchical MoE Multimodal ERC Model

## 项目概述

HMMEM (Hierarchical MoE Multimodal Emotion Recognition in Conversation) 是一个统一的多模态情感识别框架，支持多种大语言模型后端适配。该项目通过可插拔的 Mixer-Fusion 两阶段架构，实现了对文本、音频和视频三种模态信息的灵活整合与消融实验。

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
│   ├── AMIO.py                    # 模型包装器
│   ├── ChatGLM3/                  # ChatGLM3模型文件
│   ├── multiTask/                 # HMMEM 核心模块
│   │   ├── HMMEM.py               # 主模型 (forward/generate)
│   │   ├── HMMEM_mixer.py         # Mixer 层: AMM (Adaptive Modal Mixer)
│   │   ├── HMMEM_moe.py           # Fusion 层: GlobalMoE / LocalMoE
│   │   ├── HMMEM_loss.py          # 辅助损失: DiffLoss / NCE (CPC)
│   │   └── HMMEM_modules.py       # 基础组件: LSTM / TGM / MSF / FeatureAdapter
│   └── subNets/
│       └── Textmodel.py           # 统一的语言模型加载器
├── trains/                        # 训练器
│   ├── ATIO.py                    # 训练器路由
│   └── multiTask/
│       └── HMMEM.py               # HMMEM 训练器
├── utils/                         # 工具函数
├── logs/                          # 日志文件
├── results/                       # 结果保存目录
├── run.py                         # 主运行脚本
└── test_load.py                   # 数据加载测试脚本
```

## 重构亮点

### 1. 统一的语言模型加载器

在 `models/subNets/Textmodel.py` 中，我们创建了一个统一的语言模型加载器，支持：

- **ChatGLM3-6B**: 使用原生ChatGLM3实现
- **Qwen-1.8B**: 使用ModelScope加载
- **Qwen3.5系列**: 使用ModelScope加载
- **Llama2-7B**: 使用ModelScope加载
- **DeepSeek模型**: 使用ModelScope加载

### 2. 模型类型参数

通过 `--model_type` 参数指定使用的语言模型：

```bash
python run.py --model_type chatglm3  # 使用ChatGLM3-6B
python run.py --model_type qwen      # 使用Qwen-1.8B
python run.py --model_type llama2    # 使用Llama2-7B
```

### 3. 自动路径配置

系统会根据模型类型自动设置默认的预训练模型路径，也可以通过 `--pretrain_LM` 参数自定义路径。

### 4. 统一的配置系统

配置系统支持：

- 回归任务：MOSEI、SIMSV2
- 分类任务：MELD、CHERMA、IEMOCAP（4分类和6分类）

### 5. 消除代码冗余

通过重构，我们：

- 将三个独立的项目合并为一个统一框架
- 提取公共代码，减少重复
- 统一接口，便于维护和扩展

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

**Fusion 层消融（互斥）：**

| 参数 | 说明 | 默认 |
|---|---|---|
| `--use_msf` | 使用 Multi-Scale Fusion（原版基线） | True |
| `--use_moe_fusion` | 使用 Dual-Branch MoE（覆盖 MSF） | False |

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

### 示例

```bash
# 在MELD数据集上训练 (默认: TGM + MSF)
python run.py --model_type chatglm3 --datasetName meld --seeds 1111,2222,3333

# 在IEMOCAP数据集上训练（6分类）- 启用MoE
python run.py --model_type chatglm3 --datasetName iemocap6 --use_moe_fusion
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

## 模型架构 (HMMEM v2)

### 信号流

```
audio → [FeatureAdapter] → LSTM → audio_h ─┐
video → [FeatureAdapter] → LSTM → video_h ─┤──→ 【Mixer 层】──→ feature_f ──→ 【Fusion 层】──→ fusion_h ──→ LLM
text  → LLM Embedding → text_embed ────────┘
```

### Mixer 层（模态交互）

| 模块 | 说明 |
|---|---|
| **TGM** (Text-Guided Mixer) | 文本主导：GAP 池化文本 → 逐元素调制音频/视频 → 相加。基线方案。 |
| **AMM** (Adaptive Modal Mixer) | 三模态对等：Self-Attention 让音频↔视频直接交互 → 自适应加权池化。 |

### Fusion 层（特征映射）

| 模块 | 说明 |
|---|---|
| **MSF** (Multi-Scale Fusion) | 3 条不同瓶颈的scale path + Conv1d 整合。基线方案。 |
| **Dual-Branch MoE** | Global MoE (异构 Expert: Bilinear/SE/Linear) + Local MoE (Bottleneck Expert，接收原始音视频拼接) + 后验感知 Meta-Gate。 |

### 辅助优化机制

| 模块 | 说明 | 依赖 |
|---|---|---|
| **LB Loss** | Expert 负载均衡 | `use_moe_fusion` |
| **DiffLoss** | 分支/Expert 间正交互补 | `use_moe_fusion` |
| **NCE Loss** | 跨模态时序对比学习 (text↔audio, text↔video) | 无（独立） |

### 高维特征适配

| 模块 | 说明 |
|---|---|
| **FeatureAdapter** | 当输入特征维度 > `adapter_dim` 时自动启用的轻量降维层，支持 HuBERT (768) / Whisper (1280) 等高维 Encoder。 |

## 消融实验命令参考

以 MELD 数据集为例，基础命令前缀：

```bash
BASE="python run.py --model_type chatglm3 --datasetName meld --pretrain_LM /path/to/chatglm3-6b --seeds 1111,2222,3333"
```

### 第一阶段：Mixer × Fusion 组合验证

```bash
# A0: TGM + MSF (Base 锚点)
$BASE

# A1: AMM + MSF — 单独验证 AMM 是否优于 TGM
$BASE --use_amm

# A2: TGM + MoE — 单独验证新 MoE 是否优于 MSF
$BASE --use_moe_fusion

# A3: AMM + MoE — 组合
$BASE --use_amm --use_moe_fusion
```

### 第二阶段：辅助机制消融（在 A3 基础上逐个加入）

```bash
# A4: + cosine bias 门控
$BASE --use_amm --use_moe_fusion --use_gate

# A5: + 分支间 DiffLoss
$BASE --use_amm --use_moe_fusion --use_diff_loss

# A6: + 跨模态 NCE
$BASE --use_amm --use_moe_fusion --use_nce_loss

# A7: 最优组合 (根据 A4-A6 结果选择)
$BASE --use_amm --use_moe_fusion --use_gate --use_diff_loss
```

### 第三阶段：特殊消融

```bash
# 无 Mixer (Lightweight: audio + video) + MSF
$BASE --no_use_tgm

# NCE 独立于 MoE 测试
$BASE --use_nce_loss
```

## 技术特点

1. **两阶段可插拔架构**: Mixer (模态交互) 与 Fusion (特征映射) 完全解耦，每层各有 2+ 种实现，可独立消融。
2. **异构 Expert 设计**: GlobalMoE 的 3 个 Expert 使用不同计算范式 (Bilinear 交互 / SE 通道注意力 / 线性残差)，让 Gate 有真正有意义的选择。
3. **差异化 MoE 输入**: Global 分支接收融合后语义特征，Local 分支接收原始音视频拼接，天然实现宏观/微观分化。
4. **后验感知 Meta-Gate**: 在分支路由中引入两分支输出差异作为后验信号，提升决策精度。
5. **三模态对等交互 (AMM)**: 通过 Self-Attention 让音频和视频直接对话，不再被文本单向束缚。
6. **高维 Encoder 无缝适配**: FeatureAdapter 自动按需启用，支持从 Librosa 64 维到 HuBERT/Whisper 1280 维的无缝切换。

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

- **模型文件**: `results/models/{modelName}-{datasetName}-{train_mode}.pth`
- **实验结果**: `results/results/{datasetName}-{train_mode}-{warm_up_epochs}.csv`
- **日志文件**: `logs/{modelName}-{datasetName}-{model_type}.log`

## 注意事项

1. 确保已安装所需的依赖项
2. 根据使用的语言模型，正确设置 `--pretrain_LM` 路径
3. 确保数据集路径正确配置
4. 根据GPU内存情况调整 `batch_size` 参数

## 迁移指南

如果您之前使用的是分离的项目（MSE-ChatGLM3-6B、MSE-Qwen-1.8B、MSE-Llama2-7B），迁移到新的统一框架很简单：

1. 使用新的统一项目结构
2. 在运行命令中添加 `--model_type` 参数指定模型类型
3. 其他参数保持不变

例如，之前的命令：

```bash
cd MSE-ChatGLM3-6B
python run.py --datasetName mosei
```

现在改为：

```bash
cd MSE-Adapter-main
python run.py --model_type chatglm3 --datasetName mosei
```

## IEMOCAP数据集动态分类配置

MSE-Adapter框架支持IEMOCAP数据集的4分类和6分类任务，通过以下方式实现动态配置：

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
