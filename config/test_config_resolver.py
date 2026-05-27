"""验证 ConfigResolver 三级合并逻辑"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config.config_resolver import ConfigResolver

def make_args(**overrides):
    """构造模拟 CLI args"""
    defaults = {
        'modelName': 'hmmem',
        'model_type': 'chatglm3',
        'datasetName': 'meld',
        'train_mode': 'classification',
        'root_dataset_dir': '/root/autodl-tmp/datasets/',
        'data_dir': None,
        'pretrain_LM': '/root/autodl-tmp/models/chatglm3-6b-base/',
        'iemocap_feature_mode': 'raw',
        'meld_feature_mode': 'raw',
        'text_seq_len': None,
        'use_context': False,
        # CLI override 字段 (None = 不覆盖)
        'batch_size': None,
        'learning_rate': None,
        'max_epochs': None,
        'warm_up_epochs': None,
        'early_stop': None,
        'gradient_accumulation_steps': None,
        'a_lstm_hidden_size': None,
        'v_lstm_hidden_size': None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)

def test_chatglm3_meld():
    args = make_args(model_type='chatglm3', datasetName='meld')
    config = ConfigResolver(args)
    c = config.get_config()
    assert c.batch_size == 24, f"Expected 24, got {c.batch_size}"
    assert c.learning_rate == 3e-4, f"Expected 3e-4, got {c.learning_rate}"
    assert c.warm_up_epochs == 25, f"Expected 25, got {c.warm_up_epochs}"
    assert c.pretrain_LM == '/root/autodl-tmp/models/chatglm3-6b-base/'
    assert c.num_classes == 7, f"Expected 7, got {c.num_classes}"
    assert c.feature_dims == (0, 1280, 1408), f"Expected raw dims, got {c.feature_dims}"
    print("✓ chatglm3 + meld: OK")

def test_llama2_meld():
    args = make_args(model_type='llama2', datasetName='meld')
    config = ConfigResolver(args)
    c = config.get_config()
    assert c.batch_size == 6, f"Expected 6, got {c.batch_size}"
    assert c.learning_rate == 5e-4, f"Expected 5e-4, got {c.learning_rate}"
    assert c.pretrain_LM == '/root/autodl-tmp/models/Meta/Llama-2-7b-hf/'
    print("✓ llama2 + meld: OK (batch_size=6, lr=5e-4, pretrain_LM auto-set)")

def test_qwen_meld():
    args = make_args(model_type='qwen', datasetName='meld')
    config = ConfigResolver(args)
    c = config.get_config()
    assert c.batch_size == 16, f"Expected 16, got {c.batch_size}"
    assert c.learning_rate == 5e-4, f"Expected 5e-4, got {c.learning_rate}"
    assert c.warm_up_epochs == 50, f"Expected 50, got {c.warm_up_epochs}"
    assert c.pretrain_LM == '/root/autodl-tmp/models/Qwen1_8B/'
    print("✓ qwen + meld: OK (batch_size=16, lr=5e-4, warmup=50)")

def test_isolation():
    """验证修改一个模型不影响另一个"""
    args_chatglm = make_args(model_type='chatglm3', datasetName='meld')
    args_qwen = make_args(model_type='qwen', datasetName='meld')
    c1 = ConfigResolver(args_chatglm).get_config()
    c2 = ConfigResolver(args_qwen).get_config()
    assert c1.batch_size != c2.batch_size, "Should have different batch sizes"
    assert c1.learning_rate != c2.learning_rate, "Should have different LR"
    assert c1.pretrain_LM != c2.pretrain_LM, "Should have different pretrain_LM"
    print("✓ isolation: chatglm3 ≠ qwen (batch_size, lr, pretrain_LM all differ)")

def test_iemocap6():
    args = make_args(model_type='chatglm3', datasetName='iemocap6')
    config = ConfigResolver(args)
    c = config.get_config()
    assert c.num_classes == 6, f"Expected 6, got {c.num_classes}"
    assert 'excited' in c.label_index_mapping or 'frustrated' in c.label_index_mapping, \
        f"Expected 6class label mapping, got {c.label_index_mapping}"
    print("✓ iemocap6: OK (num_classes=6, 6class label mapping)")

def test_regression():
    args = make_args(model_type='chatglm3', datasetName='mosei', train_mode='regression')
    config = ConfigResolver(args)
    c = config.get_config()
    assert c.batch_size == 16, f"Expected 16, got {c.batch_size}"
    assert 'sentiment' in c.task_specific_prompt.lower()
    print("✓ chatglm3 + mosei (regression): OK")

def test_all_model_types():
    """确保所有模型类型都能成功加载"""
    for model_type in ['chatglm3', 'qwen', 'qwen3.5', 'llama2', 'deepseek', 'gemma']:
        args = make_args(model_type=model_type, datasetName='meld')
        config = ConfigResolver(args)
        c = config.get_config()
        assert c.pretrain_LM, f"{model_type}: pretrain_LM not set"
        assert c.batch_size > 0, f"{model_type}: batch_size not set"
        print(f"  ✓ {model_type:10s} → bs={c.batch_size}, lr={c.learning_rate}, LM={os.path.basename(c.pretrain_LM.rstrip('/'))}")
    print("✓ all model types: OK")

if __name__ == '__main__':
    print("=" * 60)
    print("ConfigResolver 验证测试")
    print("=" * 60)
    test_chatglm3_meld()
    test_llama2_meld()
    test_qwen_meld()
    test_isolation()
    test_iemocap6()
    test_regression()
    print()
    print("── 全模型类型加载测试 ──")
    test_all_model_types()
    print()
    print("✅ 全部测试通过！")
