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
│   ├── llm_backends/              # LLM后端适配层
│   │   ├── chatglm3/             # ChatGLM3后端与原始实现
│   │   ├── gemma/                # Gemma后端
│   │   ├── modelscope/           # Qwen/Llama2/DeepSeek通用后端
│   │   ├── base.py               # 后端基类
│   │   └── factory.py            # 后端工厂
│   ├── multiTask/                 # HMMEM 核心模块
│   │   ├── HMMEM.py               # 主模型 (forward/generate)
│   │   ├── HMMEM_mixer.py         # Mixer 层: AMM (Adaptive Modal Mixer)
│   │   ├── HMMEM_moe.py           # Fusion 层: GlobalMoE / LocalMoE
│   │   ├── HMMEM_loss.py          # 辅助损失: DiffLoss / NCE (CPC)
│   │   └── HMMEM_modules.py       # 基础组件: LSTM / TGM / MSF / FeatureAdapter
│   └── text_modules/
│       └── model_text.py          # 统一的文本入口模块
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

### 1. 双层文本/后端架构

当前项目已经从“单一大文件”演进为两层结构：

- `models/text_modules/model_text.py`：文本入口模块，负责统一编排
- `models/llm_backends/`：LLM 后端适配层，负责模型加载与特殊 token 处理

支持的后端包括：

- **ChatGLM3**：使用原生 ChatGLM3 实现
- **Qwen / Qwen3.5**：使用 ModelScope 通用后端
- **Llama2**：使用 ModelScope 通用后端
- **DeepSeek**：使用 ModelScope 通用后端
- **Gemma**：使用 HuggingFace Transformers 后端

### 2. 模型类型参数

通过 `--model_type` 参数指定使用的语言模型：

```bash
python run.py --model_type chatglm3  # 使用 ChatGLM3
python run.py --model_type qwen      # 使用 Qwen
python run.py --model_type llama2    # 使用 Llama2
python run.py --model_type deepseek  # 使用 DeepSeek
python run.py --model_type gemma     # 使用 Gemma
```

### 3. 自动路径配置

系统会根据模型类型自动设置默认的预训练模型路径，也可以通过 `--pretrain_LM` 参数自定义路径。

### 4. 统一的配置系统

配置系统支持：

- 回归任务：MOSEI、SIMSV2
- 分类任务：MELD、CHERMA、IEMOCAP（4分类和6分类）

### 5. 消除代码冗余

通过重构，我们：

- 将 LLM 适配逻辑从文本入口中拆出
- 隔离了 LLaMA2 等模型的特殊处理，减少对其他模型训练路径的影响
- 统一接口，便于维护和扩展

## 使用方法

### 基本用法

```bash
# 使用 ChatGLM3
python run.py --model_type chatglm3 --datasetName mosei --pretrain_LM /path/to/chatglm3-6b-base/

# 使用 Qwen
python run.py --model_type qwen --datasetName mosei --pretrain_LM /path/to/qwen-1.8b/

# 使用 Llama2
python run.py --model_type llama2 --datasetName mosei --pretrain_LM /path/to/llama2-7b/

# 使用 DeepSeek
python run.py --model_type deepseek --datasetName mosei --pretrain_LM /path/to/deepseek-llm-7b-base/

# 使用 Gemma
python run.py --model_type gemma --datasetName mosei --pretrain_LM /path/to/gemma-4-e4b/
```

### 参数说明

**基础参数：**

| 参数 | 类型 | 说明 | 默认值 |
|---|---|---|---|
| `--model_type` | str | 语言模型类型 | `chatglm3` |
| `--modelName` | str | 模型名（仅 `hmmem`） | `hmmem` |
| `--datasetName` | str | 数据集名称，支持逗号分隔或 `all` | `mosi` |
| `--train_mode` | str | `regression` / `classification`（会自动推断） | `regression` |
| `--pretrain_LM` | str | 预训练语言模型路径（留默认则自动推断） | 自动推断 |
| `--root_dataset_dir` | str | 数据集根目录 | `/root/autodl-tmp/datasets/` |
| `--model_save_dir` | str | 模型保存目录 | `/root/autodl-tmp/results/models` |
| `--res_save_dir` | str | 结果 CSV 保存目录 | `/root/autodl-tmp/results/results` |
| `--gpu_ids` | str | 指定 GPU ID（如 `0` 或 `0,1`） | 空（自动选择） |
| `--seeds` | str | 随机种子列表 | `1111,2222,3333,4444,5555` |
| `--num_workers` | int | DataLoader 进程数 | `4` |

**Mixer 层消融（互斥）：**

| 参数 | 类型 | 说明 | 默认 |
|---|---|---|---|
| `--use_tgm` | flag | 使用 Text-Guided Mixer（原版基线） | False |
| `--use_amm` | flag | 使用 Adaptive Modal Mixer（覆盖 TGM） | False |

> 两者均不指定时使用 Lightweight_mixer (audio + video 直接相加)

**Fusion 层消融（互斥）：**

| 参数 | 类型 | 说明 | 默认 |
|---|---|---|---|
| `--use_msf` | flag | 使用 Multi-Scale Fusion（原版基线） | False |
| `--use_moe_fusion` | flag | 使用 Dual-Branch MoE（覆盖 MSF） | False |

> 两者均不指定时使用 Direct Projection fallback

**MoE 配置：**

| 参数 | 类型 | 说明 | 默认 |
|---|---|---|---|
| `--use_gate` | flag | Meta-Gate 使用 cosine bias | False |
| `--use_moe_lb_loss` | flag | MoE 负载均衡损失 | False |
| `--num_local_experts` | int | Local MoE 专家数量 | `3` |
| `--expert_bottleneck` | int | Local Expert 瓶颈维度 | `64` |

**辅助损失：**

| 参数 | 类型 | 说明 | 默认 |
|---|---|---|---|
| `--use_diff_loss` | flag | 分支间 DiffLoss | False |
| `--use_expert_diff_loss` | flag | Expert 间 DiffLoss | False |
| `--diff_loss_weight` | float | DiffLoss 权重 | `0.01` |
| `--use_nce_loss` | flag | 跨模态 NCE (CPC) 损失 | False |
| `--nce_hidden_dim` | int | NCE CPC 隐藏维度 | `32` |
| `--nce_pred_steps` | int | NCE CPC 预测步数 | `2` |
| `--nce_weight` | float | NCE 损失权重 | `0.05` |

**高维特征适配：**

| 参数 | 类型 | 说明 | 默认 |
|---|---|---|---|
| `--adapter_dim` | int | FeatureAdapter 输出维度（仅当 feature_dim > adapter_dim 时激活） | `128` |
| `--iemocap_feature_mode` | str | IEMOCAP 特征模式：`raw`(1280/1408 维) 或 `compressed`(64 维) | `raw` |

---

## 训练脚本

> 以下脚本基于 AutoDL 环境（RTX 4090D，路径 `/root/autodl-tmp/`），可根据实际环境修改路径。

### Qwen3.5 训练脚本

```bash
# ============================================
# Qwen3.5 + MELD (分类, 7类情感)
# ============================================
python run.py \
    --model_type qwen3.5 \
    --datasetName meld \
    --pretrain_LM /root/autodl-tmp/models/Qwen/Qwen-3.5-25B/ \
    --root_dataset_dir /root/autodl-tmp/datasets/ \
    --gpu_ids 0 \
    --seeds 1111,2222,3333 \
    --num_workers 4

# ============================================
# Qwen3.5 + MOSEI (回归, 情感强度)
# ============================================
python run.py \
    --model_type qwen3.5 \
    --datasetName mosei \
    --pretrain_LM /root/autodl-tmp/models/Qwen/Qwen-3.5-25B/ \
    --root_dataset_dir /root/autodl-tmp/datasets/ \
    --gpu_ids 0 \
    --seeds 1111,2222,3333

# ============================================
# Qwen3.5 + SIMSv2 (回归, 中文情感强度)
# ============================================
python run.py \
    --model_type qwen3.5 \
    --datasetName simsv2 \
    --pretrain_LM /root/autodl-tmp/models/Qwen/Qwen-3.5-25B/ \
    --root_dataset_dir /root/autodl-tmp/datasets/ \
    --gpu_ids 0 \
    --seeds 1111,2222,3333

# ============================================
# Qwen3.5 + IEMOCAP4 (分类, 4类情感)
# ============================================
python run.py \
    --model_type qwen3.5 \
    --datasetName iemocap4 \
    --pretrain_LM /root/autodl-tmp/models/Qwen/Qwen-3.5-25B/ \
    --root_dataset_dir /root/autodl-tmp/datasets/ \
    --gpu_ids 0 \
    --seeds 1111,2222,3333

# ============================================
# Qwen3.5 + CHERMA (分类, 中文7类情感)
# ============================================
python run.py \
    --model_type qwen3.5 \
    --datasetName cherma \
    --pretrain_LM /root/autodl-tmp/models/Qwen/Qwen-3.5-25B/ \
    --root_dataset_dir /root/autodl-tmp/datasets/ \
    --gpu_ids 0 \
    --seeds 1111,2222,3333

# ============================================
# Qwen3.5 + 全部数据集（一次性跑完）
# ============================================
python run.py \
    --model_type qwen3.5 \
    --datasetName all \
    --pretrain_LM /root/autodl-tmp/models/Qwen/Qwen-3.5-25B/ \
    --root_dataset_dir /root/autodl-tmp/datasets/ \
    --gpu_ids 0 \
    --seeds 1111
```

### ChatGLM3 训练脚本

```bash
# ============================================
# ChatGLM3 + MELD
# ============================================
python run.py \
    --model_type chatglm3 \
    --datasetName meld \
    --pretrain_LM /root/autodl-tmp/models/chatglm3-6b-base/ \
    --root_dataset_dir /root/autodl-tmp/datasets/ \
    --gpu_ids 0 \
    --seeds 1111,2222,3333

# ============================================
# ChatGLM3 + MOSEI
# ============================================
python run.py \
    --model_type chatglm3 \
    --datasetName mosei \
    --pretrain_LM /root/autodl-tmp/models/chatglm3-6b-base/ \
    --root_dataset_dir /root/autodl-tmp/datasets/ \
    --gpu_ids 0 \
    --seeds 1111,2222,3333
```

### 消融实验脚本（以 Qwen3.5 + MELD 为例）

```bash
BASE="python run.py --model_type qwen3.5 --datasetName meld --gpu_ids 0 --seeds 1111,2222,3333"

# ── 第一阶段：Mixer × Fusion 组合 ──

# A0: 基线 (Lightweight + Direct Projection)
$BASE

# A1: TGM + MSF
$BASE --use_tgm --use_msf

# A2: AMM + MSF — 验证 AMM 是否优于 TGM
$BASE --use_amm --use_msf

# A3: TGM + MoE — 验证 MoE 是否优于 MSF
$BASE --use_tgm --use_moe_fusion

# A4: AMM + MoE — 组合
$BASE --use_amm --use_moe_fusion

# ── 第二阶段：辅助机制消融（在最优组合基础上逐个加入）──

# A5: + cosine bias 门控
$BASE --use_amm --use_moe_fusion --use_gate

# A6: + 分支间 DiffLoss
$BASE --use_amm --use_moe_fusion --use_diff_loss

# A7: + Expert 间 DiffLoss
$BASE --use_amm --use_moe_fusion --use_expert_diff_loss

# A8: + 跨模态 NCE
$BASE --use_amm --use_moe_fusion --use_nce_loss

# A9: + 负载均衡
$BASE --use_amm --use_moe_fusion --use_moe_lb_loss

# A10: 最优组合 (根据 A5-A9 结果选择)
$BASE --use_amm --use_moe_fusion --use_gate --use_diff_loss --use_nce_loss
```

### 高维特征实验（IEMOCAP + HuBERT/Whisper）

```bash
# IEMOCAP raw 特征 (audio 1280维, video 1408维)
python run.py \
    --model_type qwen3.5 \
    --datasetName iemocap4 \
    --iemocap_feature_mode raw \
    --adapter_dim 128 \
    --gpu_ids 0 \
    --seeds 1111,2222,3333

# IEMOCAP compressed 特征 (audio 64维, video 64维)
python run.py \
    --model_type qwen3.5 \
    --datasetName iemocap4 \
    --iemocap_feature_mode compressed \
    --gpu_ids 0 \
    --seeds 1111,2222,3333
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

- **模型文件**: `{model_save_dir}/{modelName}-{model_type}-{datasetName}-{train_mode}-{timestamp}.pth`
- **实验结果**: `{res_save_dir}/{modelName}-{model_type}-{datasetName}-{train_mode}.csv`（追加模式，每次训练新增一行，包含 Timestamp 和 PTH Path）
- **日志文件**: `logs/{modelName}-{datasetName}-{model_type}.log`

## 训练性能优化

框架内置了以下训练性能优化，无需额外配置即可生效：

| 优化项 | 说明 | 影响 |
|---|---|---|
| **bf16 自动检测** | 支持 bf16 的 GPU 自动使用 bf16（无需 GradScaler），否则回退 fp16 | 训练稳定性 + 速度 |
| **Gradient Checkpointing** | LLM forward 中用时间换空间，减少 ~60% 激活内存 | 允许更大 batch |
| **cuDNN Benchmark** | 自动选择最快的卷积/RNN 算法 | LSTM 加速 5-15% |
| **tf32 Tensor Core** | `float32_matmul_precision='medium'`，利用 RTX 30/40 Tensor Core | 矩阵乘法加速 3-8% |
| **torch.compile** | 自动编译 LSTM/Mixer/Fusion 等小模块（失败自动 fallback） | 加速 10-20% |
| **梯度累积** | 由 config 中 `gradient_accumulation_steps` 控制，含残余步处理 | 等效更大 batch |
| **DataLoader 优化** | `pin_memory`, `persistent_workers`, `prefetch_factor=2`, train 专属 `drop_last` | 数据加载加速 |
| **高效梯度清零** | `optimizer.zero_grad(set_to_none=True)` | 微小加速 |

> 各数据集的 `batch_size` 和 `gradient_accumulation_steps` 在 `config/config_classification.py` 和 `config/config_regression.py` 中配置。如遇 OOM 可减小 `batch_size`。

## 注意事项

1. 确保已安装所需的依赖项
2. 根据使用的语言模型，正确设置 `--pretrain_LM` 路径
3. 确保数据集路径正确配置
4. 根据 GPU 内存情况调整 `batch_size` 参数（在 config 文件中）
5. `--datasetName all` 可一次性训练所有数据集（train_mode 自动推断）
6. `torch.compile` 首次运行有 1-2 分钟编译开销，之后每轮受益

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
