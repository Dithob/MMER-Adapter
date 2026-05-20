import os
import argparse

from utils.functions import Storage

class ConfigClassification():
    def __init__(self, args):
        # hyper parameters for models
        HYPER_MODEL_MAP = {
            'hmmem': self.__HMMEM
        }
        # hyper parameters for datasets
        self.root_dataset_dir = args.root_dataset_dir
        self.data_dir_override = getattr(args, 'data_dir', None)
        HYPER_DATASET_MAP = self.__datasetCommonParams()

        # normalize
        model_name = str.lower(args.modelName)
        dataset_name = str.lower(args.datasetName)
        
        # load params
        commonArgs = HYPER_MODEL_MAP[model_name]()['commonParas']
        
        # Handle iemocap4 and iemocap6 dataset names
        if dataset_name in ['iemocap4', 'iemocap6']:
            # Use iemocap as the base dataset name for configuration
            base_dataset_name = 'iemocap'
            # Set num_classes based on dataset name
            num_classes = 4 if dataset_name == 'iemocap4' else 6
        else:
            base_dataset_name = dataset_name
            # num_classes will come from the dataset config (dataArgs), don't override
            num_classes = None
        
        # Get data parameters
        dataArgs = HYPER_DATASET_MAP[base_dataset_name]
        if base_dataset_name == 'iemocap':
            # Switchable IEMOCAP feature mode:
            # - raw: Emotion-LLaMA-v2 native dimensions (audio 1280 / video 1408)
            # - compressed: legacy MMER-Adapter dimensions (audio 64 / video 64)
            iemocap_feature_mode = str(getattr(args, 'iemocap_feature_mode', 'raw')).lower()
            if commonArgs['need_data_aligned'] and 'aligned' in dataArgs:
                dataArgs = dataArgs['aligned']
            else:
                if iemocap_feature_mode == 'raw':
                    dataArgs = dataArgs['unaligned_raw']
                elif iemocap_feature_mode == 'compressed':
                    dataArgs = dataArgs['unaligned_compressed']
                else:
                    raise ValueError(
                        f"Unsupported iemocap_feature_mode={iemocap_feature_mode}. "
                        f"Use 'raw' or 'compressed'."
                    )
        elif base_dataset_name == 'meld':
            # Switchable MELD feature mode (same pattern as IEMOCAP):
            meld_feature_mode = str(getattr(args, 'meld_feature_mode', 'raw')).lower()
            if commonArgs['need_data_aligned'] and 'aligned' in dataArgs:
                dataArgs = dataArgs['aligned']
            else:
                if meld_feature_mode == 'raw':
                    dataArgs = dataArgs['unaligned_raw']
                elif meld_feature_mode == 'compressed':
                    dataArgs = dataArgs['unaligned_compressed']
                else:
                    raise ValueError(
                        f"Unsupported meld_feature_mode={meld_feature_mode}. "
                        f"Use 'raw' or 'compressed'."
                    )
        else:
            dataArgs = dataArgs['aligned'] if (commonArgs['need_data_aligned'] and 'aligned' in dataArgs) else dataArgs['unaligned']
        
        # Only override num_classes for iemocap (where it's dynamically determined)
        if num_classes is not None:
            dataArgs['num_classes'] = num_classes
        
        # Get dataset parameters
        dataset_paras = HYPER_MODEL_MAP[model_name]()['datasetParas'][base_dataset_name]
        
        # For iemocap, dynamically select prompt and label mapping based on num_classes
        if base_dataset_name == 'iemocap':
            if num_classes == 6:
                dataset_paras['task_specific_prompt'] = dataset_paras['task_specific_prompt_6class']
                dataset_paras['label_index_mapping'] = dataset_paras['label_index_mapping_6class']
            # Remove the 6class-specific keys to avoid confusion
            dataset_paras.pop('task_specific_prompt_6class', None)
            dataset_paras.pop('label_index_mapping_6class', None)
        
        # integrate all parameters
        self.args = Storage(dict(vars(args),
                            **dataArgs,
                            **commonArgs,
                            **dataset_paras,
                            ))

        # ── Context Ablation: dynamic seq_lens[0] override ──
        # seq_lens = (text, audio, video) — only text dimension is modified here
        if hasattr(self.args, 'text_seq_len') and self.args.text_seq_len is not None:
            old = list(self.args.seq_lens)
            old[0] = self.args.text_seq_len
            self.args.seq_lens = tuple(old)
        elif getattr(self.args, 'use_context', False):
            # Context mode: auto-increase text seq_len to accommodate context
            old = list(self.args.seq_lens)
            old[0] = max(old[0], 128)
            self.args.seq_lens = tuple(old)
    
    def __datasetCommonParams(self):
        root_dataset_dir = self.root_dataset_dir
        data_dir = self.data_dir_override  # CLI override for dataset subfolder name

        # Default subfolder names per dataset (used when --data_dir is not set)
        iemocap_dir = data_dir if data_dir else 'IEMOCAP'
        meld_dir    = data_dir if data_dir else 'MELD'
        cherma_dir  = data_dir if data_dir else 'CHERMA0723'

        tmp = {
            'iemocap':{
                # Old compressed setting (kept here as reference):
                # 'seq_lens': (84, 157, 32)
                # 'feature_dims': (0, 64, 64)
                'unaligned_compressed': {
                    'dataPath': os.path.join(root_dataset_dir, iemocap_dir, 'iemocap_data_0610.pkl'),
                    'seq_lens': (84, 157, 32),
                    # (text, audio, video) text_dim=0 means auto-detect from LLM hidden_size
                    'feature_dims': (0, 64, 64),
                    'train_samples': 4290,
                    'num_classes': 4,
                    'language': 'en',
                    'KeyEval': 'weight_F1'
                },
                'unaligned_raw': {
                    'dataPath': os.path.join(root_dataset_dir, iemocap_dir, 'iemocap_data_0610.pkl'),
                    'seq_lens': (84, 64, 64),
                    # (text, audio, video) text_dim=0 means auto-detect from LLM hidden_size
                    'feature_dims': (0, 1280, 1408),
                    'train_samples': 4290,
                    'num_classes': 4,
                    'language': 'en',
                    'KeyEval': 'weight_F1'
                },
            },
            'meld':{
                'unaligned_compressed': {
                    'dataPath': os.path.join(root_dataset_dir, meld_dir),
                    'seq_lens': (65, 157, 32),
                    # (text, audio, video) text_dim=0 means auto-detect from LLM hidden_size
                    'feature_dims': (0, 64, 64),
                    'train_samples': 9988,
                    'num_classes': 7,
                    'language': 'en',
                    'KeyEval': 'weight_F1'
                },
                'unaligned_raw': {
                    'dataPath': os.path.join(root_dataset_dir, meld_dir, 'meld_data_0610.pkl'),
                    'seq_lens': (65, 64, 64),
                    # (text, audio, video) text_dim=0 means auto-detect from LLM hidden_size
                    'feature_dims': (0, 1280, 1408),
                    'train_samples': 9988,
                    'num_classes': 7,
                    'language': 'en',
                    'KeyEval': 'weight_F1'
                },
            },
            'cherma':{
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, cherma_dir),
                    # (batch_size, seq_lens, feature_dim)
                    'seq_lens': (78, 543, 16), # (text, audio, video)
                    'feature_dims': (0, 1024, 2048), # text_dim=0 means auto-detect from LLM hidden_size
                    'train_samples': 16326,
                    'num_classes': 3,
                    'language': 'cn',
                    'KeyEval': 'weight_F1',
                }
            },
        }
        return tmp

    def __HMMEM(self):
        tmp = {
            'commonParas':{
                'need_data_aligned': False,
                'need_model_aligned': False,
                'need_label_prefix':True,
                'need_normalized': False,
                'use_PLM': True,
                'save_labels': False,
            },
            # dataset
            'datasetParas':{
                'meld':{
                    # batch_size of each epoch is update_epochs * batch_size
                    'task_specific_prompt': 'Please recognize emotion of the above multimodal content from the target \
                                                set <neutral:0, surprise:1, fear:2, sadness:3, joy:4, disgust:5, anger:6>. response: The emotion is',
                    'max_new_tokens': 1,
                    'pseudo_tokens': 2,
                    'label_index_mapping': {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5,
                                           'anger': 6},
                    # 'batch_size': 8,
                    'batch_size': 24,
                    'gradient_accumulation_steps': 1,
                    'learning_rate': 2e-4,
                    # feature modules
                    'a_lstm_hidden_size': 32,
                    'v_lstm_hidden_size': 16,
                    'a_lstm_layers': 1,
                    'v_lstm_layers': 1,
                    'a_lstm_dropout': 0.0,
                    'v_lstm_dropout': 0.0,
                    'warm_up_epochs': 25,
                    #loss weight   best：1
                    'gamma': 1,
                    'update_epochs': 1,
                    'early_stop': 8,
                    'max_epochs': 50,
                    # res
                    'H': 3.0
                },
                'cherma':{
                    # batch_size of each epoch is update_epochs * batch_size
                    'task_specific_prompt': '请选择适用于上述多模态内容的情绪标签：<愤怒:0, 厌恶:1, 恐惧:2, 高兴:3, 平静:4, 悲伤:5, 惊奇:6>。响应: 情绪为',
                    'max_new_tokens': 2,
                    'pseudo_tokens': 4,
                    'label_index_mapping': {'愤怒': 0, '厌恶': 1, '恐惧': 2, '高兴': 3, '平静': 4, '悲伤': 5,
                                            '惊奇': 6},
                    'batch_size': 16,
                    'gradient_accumulation_steps': 2,
                    'learning_rate': 5e-5,
                    # feature modules
                    'a_lstm_hidden_size': 32,
                    'v_lstm_hidden_size': 16,
                    'a_lstm_layers': 1,
                    'v_lstm_layers': 1,
                    'a_lstm_dropout': 0.0,
                    'v_lstm_dropout': 0.0,
                    'warm_up_epochs': 30,
                    'update_epochs': 1,
                    'early_stop': 8,
                    'max_epochs': 50,
                    # loss weight
                    'gamma': 0,
                    # res
                    'H': 1.0
                },
                'iemocap':{
                    'task_specific_prompt': 'Please recognize the emotion of the above multimodal content from the target \
                                                set <angry:0, happy:1, sad:2, neutral:3>. response: The emotion is',
                    'task_specific_prompt_6class': 'Please recognize the emotion of the above multimodal content from the target \
                                                set <angry:0, happy:1, excited:2, sad:3, neutral:4, frustrated:5>. response: The emotion is',
                    'max_new_tokens': 2,
                    'pseudo_tokens': 4,
                    'label_index_mapping': {'angry': 0, 'happy': 1, 'sad': 2, 'neutral': 3},
                    'label_index_mapping_6class': {'angry': 0, 'happy': 1, 'excited': 2, 'sad': 3, 'neutral': 4, 'frustrated': 5},
                    'batch_size': 24,
                    'gradient_accumulation_steps': 2,
                    'learning_rate': 1e-4,
                    'a_lstm_hidden_size': 64,
                    'v_lstm_hidden_size': 32,
                    'a_lstm_layers': 1,
                    'v_lstm_layers': 1,
                    'a_lstm_dropout': 0.0,
                    'v_lstm_dropout': 0.0,
                    'warm_up_epochs': 30,
                    'gamma':1,
                    'update_epochs': 1,
                    'early_stop': 8,
                    'max_epochs': 50,
                    'H': 3.0
                },
            },
        }
        return tmp

    def get_config(self):
        return self.args
