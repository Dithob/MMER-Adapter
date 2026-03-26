# MSE-Adapter: Unified Multimodal Sentiment Analysis Framework

## 项目概述

MSE-Adapter 是一个统一的多模态情感分析框架，支持多种大语言模型（ChatGLM3-6B、Qwen-1.8B、Llama2-7B）的后端适配。该项目通过文本引导的多模态融合技术，实现了对文本、音频和视频三种模态信息的有效整合。

## 项目结构

```
MSE-Adapter-main/
├── config/                    # 配置文件
│   ├── config_classification.py # 分类任务配置
│   └── config_regression.py    # 回归任务配置
├── data/                      # 数据处理
│   ├── DataPre.py             # 数据预处理
│   ├── TextPre.py            # 文本预处理
│   └── load_data.py          # 数据加载器
├── models/                    # 模型定义
│   ├── AMIO.py               # 模型包装器
│   ├── ChatGLM3/             # ChatGLM3模型文件
│   ├── multiTask/            # 多任务模型
│   │   └── CMCM.py          # 核心多模态融合模型
│   └── subNets/             # 子网络
│       └── Textmodel.py      # 统一的语言模型加载器
├── trains/                    # 训练器
│   ├── ATIO.py               # 训练器包装器
│   └── multiTask/            # 多任务训练器
│       └── CMCM.py          # CMCM训练器
├── utils/                     # 工具函数
│   ├── functions.py          # 辅助函数
│   └── metricsTop.py        # 评估指标
├── logs/                      # 日志文件
├── results/                   # 结果保存目录
│   ├── models/               # 模型保存
│   └── results/             # 实验结果
├── run.py                    # 主运行脚本
└── test_load.py              # 数据加载测试脚本
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

- `--model_type`: 语言模型类型，可选 `chatglm3`、`qwen`、`qwen3.5`、`llama2`、`deepseek`
- `--datasetName`: 数据集名称，支持 `mosei`、`simsv2`、`meld`、`cherma`、`iemocap4`、`iemocap6`
- `--pretrain_LM`: 预训练语言模型路径
- `--train_mode`: 训练模式，`regression`（回归）或 `classification`（分类）
- `--modelName`: 模型名称，目前支持 `cmcm`
- `--root_dataset_dir`: 数据集根目录
- `--gpu_ids`: 使用的GPU ID列表
- `--seeds`: 随机种子列表
- `--use_moe_fusion`: 是否启用MoE融合机制，默认为False
- `--use_gate`: 是否启用偏差感知门控机制，默认为False

### 示例

```bash
# 在MOSEI数据集上训练ChatGLM3-6B模型 (默认参数)
python run.py --model_type chatglm3 --datasetName mosei

# 在SIMSV2数据集上训练Qwen-1.8B模型，并指定使用 GPU 1
python run.py --model_type qwen --datasetName simsv2 --gpu_ids 1

# 在MELD数据集上训练ChatGLM3-6B模型
python run.py --model_type chatglm3 --datasetName meld

# 在IEMOCAP数据集上训练（4分类）
python run.py --model_type chatglm3 --datasetName iemocap4

# 在IEMOCAP数据集上训练（6分类）- 启用MoE和偏差感知门控
python run.py --model_type qwen --datasetName iemocap6 --use_moe_fusion --use_gate

# 一次性训练多个数据集（按顺序训练）
python run.py --model_type chatglm3 --datasetName mosei,simsv2,meld

# 一次性训练所有支持的数据集
python run.py --model_type qwen --datasetName all

# 自定义预训练路径和随机种子
python run.py \
    --model_type llama2 \
    --datasetName mosei \
    --pretrain_LM /custom/path/to/llama2-7b/ \
    --seeds 1111,2222 \
    --gpu_ids 0
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

## 模型架构

CMCM模型包含以下组件：

1. **文本编码器**: 使用预训练语言模型（ChatGLM3/Qwen/Llama2/DeepSeek）
2. **音频编码器**: LSTM网络
3. **视频编码器**: LSTM网络
4. **文本引导混合器**: 利用文本信息引导音频和视频特征融合
5. **多尺度融合器**: 通过不同尺度的特征提取和整合
6. **MoE融合机制**（可选）: 包含深度融合和轻量级融合两个专家网络
7. **门控机制**（可选）: 自适应选择最适合当前输入的融合策略

## 技术特点

1. **文本引导融合**: 利用文本语义信息指导多模态特征融合
2. **多尺度特征提取**: 捕获不同层次的特征信息
3. **低秩融合**: 减少计算复杂度，提高模型效率
4. **MoE自主融合策略**: 结合深度融合和轻量级融合两个专家网络，提高模型表达能力
5. **偏差感知门控机制**: 利用音频和视频特征的余弦相似度作为偏差指标，自适应选择最适合当前输入的融合策略
6. **大语言模型适配**: 支持多种主流大语言模型
7. **灵活的配置系统**: 支持多种数据集和任务类型

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
