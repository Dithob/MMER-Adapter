"""
config_resolver.py — 三级配置合并器

配置优先级 (高覆盖低):
  Level 3: CLI Override (--batch_size 32 等临时覆盖)
  Level 2: Model Profile = default.yaml ← {model_type}.yaml (深度合并覆盖)
  Level 1: Dataset Common (数据集公共参数: dataPath, seq_lens, feature_dims 等)

模型 YAML 文件只需写与 default.yaml 不同的参数即可，其余自动继承默认值。

使用方式:
  from config.config_resolver import ConfigResolver
  config = ConfigResolver(args)
  args = config.get_config()
"""

import os
import logging
import yaml

from utils.functions import Storage

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# Level 1: 数据集公共参数 (与模型无关)
# 只包含数据路径、特征维度、序列长度等物理属性
# ═══════════════════════════════════════════════════════════════════

def _dataset_common_classification(root_dataset_dir, data_dir_override=None):
    """分类任务的数据集公共参数"""
    iemocap_dir = data_dir_override or 'IEMOCAP'
    meld_dir    = data_dir_override or 'MELD'
    cherma_dir  = data_dir_override or 'CHERMA0723'

    return {
        'iemocap': {
            'unaligned_compressed': {
                'dataPath': os.path.join(root_dataset_dir, iemocap_dir, 'iemocap_data_0610.pkl'),
                'seq_lens': (84, 157, 32),
                'feature_dims': (0, 64, 64),
                'train_samples': 4290,
                'num_classes': 4,
                'language': 'en',
                'KeyEval': 'weight_F1'
            },
            'unaligned_raw': {
                'dataPath': os.path.join(root_dataset_dir, iemocap_dir, 'iemocap_data_0610.pkl'),
                'seq_lens': (84, 64, 64),
                'feature_dims': (0, 1280, 1408),
                'train_samples': 4290,
                'num_classes': 4,
                'language': 'en',
                'KeyEval': 'weight_F1'
            },
        },
        'meld': {
            'unaligned_compressed': {
                'dataPath': os.path.join(root_dataset_dir, meld_dir),
                'seq_lens': (65, 157, 32),
                'feature_dims': (0, 64, 64),
                'train_samples': 9988,
                'num_classes': 7,
                'language': 'en',
                'KeyEval': 'weight_F1'
            },
            'unaligned_raw': {
                'dataPath': os.path.join(root_dataset_dir, meld_dir, 'meld_data_0610.pkl'),
                'seq_lens': (65, 64, 64),
                'feature_dims': (0, 1280, 1408),
                'train_samples': 9988,
                'num_classes': 7,
                'language': 'en',
                'KeyEval': 'weight_F1'
            },
        },
        'cherma': {
            'unaligned': {
                'dataPath': os.path.join(root_dataset_dir, cherma_dir),
                'seq_lens': (78, 543, 16),
                'feature_dims': (0, 1024, 2048),
                'train_samples': 16326,
                'num_classes': 3,
                'language': 'cn',
                'KeyEval': 'weight_F1',
            }
        },
    }


def _dataset_common_regression(root_dataset_dir):
    """回归任务的数据集公共参数"""
    return {
        'mosi': {
            'unaligned': {
                'dataPath': os.path.join(root_dataset_dir, 'MOSI/Processed/unaligned_50.pkl'),
                'seq_lens': (50, 50, 50),
                'feature_dims': (0, 5, 20),
                'train_samples': 1284,
                'num_classes': 3,
                'language': 'en',
                'KeyEval': 'MAE'
            }
        },
        'mosei': {
            'unaligned': {
                'dataPath': os.path.join(root_dataset_dir, 'MOSEI/Processed/unaligned_50.pkl'),
                'seq_lens': (50, 500, 375),
                'feature_dims': (0, 74, 35),
                'train_samples': 16326,
                'num_classes': 3,
                'language': 'en',
                'KeyEval': 'MAE'
            },
            'unaligned_compressed': {
                'dataPath': os.path.join(root_dataset_dir, 'MOSEI', 'mosei_data_0610.pkl'),
                'seq_lens': (50, 157, 32),
                'feature_dims': (0, 64, 64),
                'train_samples': 16326,
                'num_classes': 3,
                'language': 'en',
                'KeyEval': 'MAE'
            },
        },
        'simsv2': {
            'unaligned': {
                'dataPath': os.path.join(root_dataset_dir, 'SIMS_V2/ch-simsv2s.pkl'),
                'seq_lens': (50, 925, 232),
                'feature_dims': (0, 25, 177),
                'train_samples': 2722,
                'num_classes': 3,
                'language': 'cn',
                'KeyEval': 'MAE',
            }
        }
    }


# ═══════════════════════════════════════════════════════════════════
# Model Common Parameters (所有模型共享的开关)
# ═══════════════════════════════════════════════════════════════════

_COMMON_PARAS = {
    'need_data_aligned': False,
    'need_model_aligned': False,
    'need_label_prefix': True,
    'need_normalized': False,
    'use_PLM': True,
    'save_labels': False,
}


# ═══════════════════════════════════════════════════════════════════
# Level 2: Model Profile Loader (default.yaml + 模型覆盖)
# ═══════════════════════════════════════════════════════════════════

_PROFILE_DIR = os.path.join(os.path.dirname(__file__), 'model_profiles')

# 缓存已加载的 YAML (同一进程内不重复读文件)
_yaml_cache = {}


def _load_yaml(filename):
    """加载 YAML 文件 (带缓存)"""
    if filename in _yaml_cache:
        return _yaml_cache[filename]

    filepath = os.path.join(_PROFILE_DIR, filename)
    if not os.path.exists(filepath):
        return None

    with open(filepath, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    _yaml_cache[filename] = data or {}
    return _yaml_cache[filename]


def _deep_merge(base, override):
    """
    递归深度合并两个 dict。override 中的值覆盖 base 中的同名 key。
    
    规则:
    - 如果 base[key] 和 override[key] 都是 dict → 递归合并
    - 否则 → override[key] 直接替换 base[key]
    - override 中存在但 base 中不存在的 key → 直接添加
    
    注意: 此函数不会修改原始 base/override dict，返回新 dict。
    """
    import copy
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _get_profile_params(model_type, task_type, dataset_name):
    """
    加载 default.yaml 作为基础，然后用 {model_type}.yaml 中的差异参数覆盖。
    最后提取 common + task_type.dataset_name 的参数。
    
    合并链: default.yaml ← {model_type}.yaml
    
    Returns:
        dict: 合并后的扁平参数 (common 被 task-specific 覆盖)
    """
    # 1. 加载 default.yaml (基础配置)
    default_profile = _load_yaml('default.yaml')
    if default_profile is None:
        raise FileNotFoundError(
            f"default.yaml not found in {_PROFILE_DIR}/\n"
            f"This file is required as the base configuration for all models."
        )

    # 2. 加载模型专属 YAML (覆盖配置)
    model_yaml = f'{model_type}.yaml'
    model_profile = _load_yaml(model_yaml)
    if model_profile is None:
        available = [f.replace('.yaml', '') for f in os.listdir(_PROFILE_DIR) if f.endswith('.yaml') and f != 'default.yaml']
        raise FileNotFoundError(
            f"Model profile not found: {model_yaml}\n"
            f"Please create it in {_PROFILE_DIR}/ (can be nearly empty, just override what differs from default.yaml)\n"
            f"Available model profiles: {available}"
        )

    # 3. 深度合并: default ← model_override
    merged_profile = _deep_merge(default_profile, model_profile)

    # 4. 提取 common + task-specific 参数
    result = {}
    if 'common' in merged_profile:
        result.update(merged_profile['common'])
    if task_type in merged_profile and dataset_name in merged_profile[task_type]:
        result.update(merged_profile[task_type][dataset_name])

    return result


# ═══════════════════════════════════════════════════════════════════
# ConfigResolver — 三级合并入口
# ═══════════════════════════════════════════════════════════════════

class ConfigResolver:
    """
    统一配置解析器，替代 ConfigClassification / ConfigRegression。
    
    合并优先级: CLI args > Model Profile (YAML) > Dataset Common > Model Common
    """

    def __init__(self, args):
        model_type = getattr(args, 'model_type', 'chatglm3')
        dataset_name = str.lower(args.datasetName)
        train_mode = str.lower(args.train_mode)   # 'classification' or 'regression'
        task_type = train_mode   # YAML 中的段名

        # ── 1. 确定 base_dataset_name (处理 iemocap4/iemocap6 别名) ──
        num_classes_override = None
        if dataset_name in ('iemocap4', 'iemocap6'):
            base_dataset_name = 'iemocap'
            num_classes_override = 4 if dataset_name == 'iemocap4' else 6
        else:
            base_dataset_name = dataset_name

        # ── 2. Level 1: 数据集公共参数 ──
        root_dir = args.root_dataset_dir
        data_dir_override = getattr(args, 'data_dir', None)

        if train_mode == 'classification':
            ds_map = _dataset_common_classification(root_dir, data_dir_override)
        else:
            ds_map = _dataset_common_regression(root_dir)

        if base_dataset_name not in ds_map:
            raise ValueError(
                f"Dataset '{base_dataset_name}' not found in {train_mode} dataset map. "
                f"Available: {list(ds_map.keys())}"
            )

        dataArgs = ds_map[base_dataset_name]

        # 选择 aligned/unaligned/raw/compressed 变体
        if base_dataset_name == 'iemocap' and train_mode == 'classification':
            mode = str(getattr(args, 'iemocap_feature_mode', 'raw')).lower()
            key = f'unaligned_{mode}'
            if key not in dataArgs:
                raise ValueError(f"Unsupported iemocap_feature_mode={mode}. Use 'raw' or 'compressed'.")
            dataArgs = dataArgs[key]
        elif base_dataset_name == 'meld' and train_mode == 'classification':
            mode = str(getattr(args, 'meld_feature_mode', 'raw')).lower()
            key = f'unaligned_{mode}'
            if key not in dataArgs:
                raise ValueError(f"Unsupported meld_feature_mode={mode}. Use 'raw' or 'compressed'.")
            dataArgs = dataArgs[key]
        elif base_dataset_name == 'mosei' and train_mode == 'regression':
            mode = str(getattr(args, 'mosei_feature_mode', 'legacy')).lower()
            if mode == 'legacy':
                key = 'unaligned'
            else:
                key = f'unaligned_{mode}'
            if key not in dataArgs:
                raise ValueError(f"Unsupported mosei_feature_mode={mode}. Use 'legacy' or 'compressed'.")
            dataArgs = dataArgs[key]
        else:
            # cherma, simsv2 等
            if 'unaligned' in dataArgs:
                dataArgs = dataArgs['unaligned']
            # 如果直接就是最终 dict (没有 aligned/unaligned 嵌套)，保持原样

        # Override num_classes for iemocap4/6
        if num_classes_override is not None:
            dataArgs = dict(dataArgs)  # shallow copy to avoid mutating the template
            dataArgs['num_classes'] = num_classes_override

        # ── 3. Level 2: 模型专属 Profile ──
        profile_params = _get_profile_params(model_type, task_type, base_dataset_name)

        # iemocap 6class 特殊处理: 动态选择 prompt 和 label mapping
        if base_dataset_name == 'iemocap' and num_classes_override == 6:
            if 'task_specific_prompt_6class' in profile_params:
                profile_params['task_specific_prompt'] = profile_params['task_specific_prompt_6class']
            if 'label_index_mapping_6class' in profile_params:
                profile_params['label_index_mapping'] = profile_params['label_index_mapping_6class']
        # 清理 6class 临时 key (避免污染 args)
        profile_params.pop('task_specific_prompt_6class', None)
        profile_params.pop('label_index_mapping_6class', None)

        # ── 4. 检测 pretrain_LM 是否被 CLI 显式覆盖 ──
        # 如果用户没有通过 CLI 显式设置 --pretrain_LM (仍是 argparse 默认值),
        # 则用 profile 中的 pretrain_LM
        _CLI_DEFAULT_PRETRAIN = '/root/autodl-tmp/models/chatglm3-6b-base/'
        cli_pretrain = getattr(args, 'pretrain_LM', _CLI_DEFAULT_PRETRAIN)
        if cli_pretrain != _CLI_DEFAULT_PRETRAIN:
            # 用户通过 CLI 显式指定了 LM 路径，覆盖 profile
            profile_params['pretrain_LM'] = cli_pretrain

        # ── 5. 合并: CLI args < Dataset Common < Model Common < Profile ──
        # 注意: dict 后面的 key 会覆盖前面的
        merged = dict(
            vars(args),           # CLI args (最低优先级的基础)
            **dataArgs,           # Level 1: 数据集路径、维度
            **_COMMON_PARAS,      # 模型公共开关
            **profile_params,     # Level 2: 模型×数据集 训练参数 (覆盖上面)
        )

        self.args = Storage(merged)

        # ── 6. Context Ablation: dynamic seq_lens[0] override ──
        if hasattr(self.args, 'text_seq_len') and self.args.text_seq_len is not None:
            old = list(self.args.seq_lens)
            old[0] = self.args.text_seq_len
            self.args.seq_lens = tuple(old)
        elif getattr(self.args, 'use_context', False):
            old = list(self.args.seq_lens)
            old[0] = max(old[0], 128)
            self.args.seq_lens = tuple(old)

        logger.info(f"[ConfigResolver] model_type={model_type}, dataset={dataset_name}, "
                     f"task={train_mode}, profile_keys={list(profile_params.keys())}")

    def get_config(self):
        return self.args
