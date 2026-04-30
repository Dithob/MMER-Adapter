import os
import logging
import pickle
import json
import numpy as np
import pandas as pd
import torch
import gzip
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from operator import itemgetter
from torch.nn.utils.rnn import pad_sequence

__all__ = ['MMDataLoader']

logger = logging.getLogger('MSA')

class MMDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.mode = mode
        self.args = args
        DATA_MAP = {
            'mosi': self.__init_mosi,
            'mosei': self.__init_mosei,
            'sims': self.__init_sims,
            'simsv2': self.__init_simsv2,
            'meld': self.__init_meld,
            'iemocap': self.__init_iemocap,
            'iemocap4': self.__init_iemocap,
            'iemocap6': self.__init_iemocap,
            'cherma': self.__init_cherma,

        }
        DATA_MAP[args.datasetName]()



    def __init_meld(self):
        """Auto-detect and load MELD data in either:
        - NEW format: single .pkl (split-indexed dicts) + meld_text/ CSV  (from preprocess_meld_emotionllama_v2.py)
        - OLD format: per-split .pkl files (list of dicts with embedded features+text)  (legacy)
        """
        import csv

        data_path = self.args.dataPath
        label_index_mapping = self.args.label_index_mapping

        # ── Format detection ──
        is_new_format = data_path.endswith('.pkl') and os.path.isfile(data_path)
        is_old_format = os.path.isdir(data_path)

        if not is_new_format and not is_old_format:
            # dataPath might be a .pkl path that points to a new-format file
            # or a directory for old-format; if neither exists, raise
            raise FileNotFoundError(
                f"MELD dataPath not found: {data_path}. "
                f"Expected a .pkl file (new format) or a directory (old format)."
            )

        if is_old_format:
            # ══════════════════════════════════════════════
            #  OLD FORMAT: per-split pkl (meld_train.pkl, etc.)
            # ══════════════════════════════════════════════
            split_pkl = os.path.join(data_path, self.args.datasetName + '_' + self.mode + '.pkl')
            if not os.path.exists(split_pkl):
                # Try alternate naming: meld_train.pkl
                split_pkl = os.path.join(data_path, 'meld_' + self.mode + '.pkl')
            logger.info(f"MELD loading OLD format: {split_pkl}")
            with open(split_pkl, 'rb') as f:
                data = pickle.load(f)
            self.vision = np.array(list(map(lambda item: item['features']['video'], data))).astype(np.float32)
            self.audio = np.array(list(map(lambda item: item['features']['audio'], data))).astype(np.float32)
            self.rawText = np.array(list(map(lambda item: item['features']['text'], data)))
            self.labels = {
                'M': list(map(lambda item: label_index_mapping.get(item['label'], -1), data))
            }
            if self.args.use_PLM:
                self.text = self.PLM_tokenizer(self.rawText)
            if not self.args.need_data_aligned:
                self.audio_lengths = np.array(list(map(lambda item: item['features']['audio_len'], data)))
                self.vision_lengths = np.array(list(map(lambda item: item['features']['video_len'], data)))
            return

        # ══════════════════════════════════════════════
        #  NEW FORMAT: single pkl (split-indexed dicts) + CSV text
        # ══════════════════════════════════════════════
        logger.info(f"MELD loading NEW format: {data_path}")

        def build_multimodal_text(meta):
            """Build text: utterance-first ordering ensures utterance is never truncated.
            
            If --use_context: "utterance [speaker] Context: ..."
            If no context:    "utterance"  (matching UniMSE's proven approach)
            
            Truncation is right-side, so only context tail gets clipped.
            """
            text = str(meta.get('text', '')).strip()
            if not getattr(self.args, 'use_context', False):
                return text
            # Utterance-first: truncation only cuts context tail
            parts = [text]
            speaker = str(meta.get('speaker', '')).strip()
            if speaker:
                parts.append(f'[{speaker}]')
            context = str(meta.get('context', '')).strip()
            if context and context != '[]':
                # Parse JSON list format if present
                if context.startswith('['):
                    try:
                        parsed = json.loads(context)
                        if isinstance(parsed, list):
                            context = ' '.join(str(t).strip() for t in parsed if str(t).strip())
                    except json.JSONDecodeError:
                        pass
                parts.append(f'Context: {context}')
            return ' '.join(parts)

        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        split_idx = 0 if self.mode == 'train' else (1 if self.mode == 'valid' else 2)
        if len(data['audio'][split_idx]) == 0:
            split_idx = 2

        audio_dict = data['audio'][split_idx]
        video_dict = data['video'][split_idx]

        # Load text/labels from CSV (meld_text/ directory)
        meld_text_dir = None
        data_dir = os.path.dirname(data_path)
        for p in [os.path.join(data_dir, 'meld_text'),
                  os.path.join(os.path.dirname(data_dir), 'meld_text'),
                  os.path.join(data_dir, '..', 'MELD', 'meld_text')]:
            if os.path.exists(p):
                meld_text_dir = p
                break

        meld_meta = {}
        if meld_text_dir:
            csv_path = os.path.join(meld_text_dir, f'meld_data_{self.mode}.csv')
            if os.path.exists(csv_path):
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if 'vid_cid' in row and 'label' in row:
                            meld_meta[row['vid_cid']] = {
                                'label': row['label'].strip(),
                                'text': row.get('text', '').strip(),
                                'context': row.get('context', '').strip(),
                                'speaker': row.get('speaker', '').strip(),
                            }
                logger.info(f"MELD text loaded from {csv_path}: {len(meld_meta)} entries")

        if not meld_meta:
            logger.warning(f"No MELD text CSV found near {data_dir}. Using keys from pkl only.")
            for k in audio_dict.keys():
                meld_meta[k] = {'label': 'neutral', 'text': '', 'context': '', 'speaker': ''}

        # Build padded arrays
        def pad_sequence_numpy(sequences, max_len, feature_dim):
            padded = []
            for seq in sequences:
                seq = np.atleast_2d(seq)
                if seq.shape[0] >= max_len:
                    padded.append(seq[:max_len, :])
                else:
                    pad = np.zeros((max_len - seq.shape[0], feature_dim), dtype=np.float32)
                    padded.append(np.vstack((seq, pad)))
            return np.array(padded, dtype=np.float32)

        raw_audio, raw_video, raw_text, raw_context, labels_m = [], [], [], [], []
        audio_lengths, video_lengths = [], []
        drop_stats = {'missing_meta': 0, 'mapping_missing': 0, 'missing_modal': 0}

        keys = list(audio_dict.keys())
        for k in keys:
            meta = meld_meta.get(k)
            if meta is None:
                drop_stats['missing_meta'] += 1
                continue

            mapped_label = meta['label']
            if mapped_label not in label_index_mapping:
                drop_stats['mapping_missing'] += 1
                continue

            aud_feat = audio_dict.get(k)
            vid_feat = video_dict.get(k)
            if aud_feat is None or vid_feat is None:
                drop_stats['missing_modal'] += 1
                continue

            labels_m.append(label_index_mapping[mapped_label])
            raw_text.append(build_multimodal_text(meta))
            # Collect context for prompt-level injection
            ctx = str(meta.get('context', '')).strip()
            if ctx.startswith('['):
                try:
                    parsed = json.loads(ctx)
                    if isinstance(parsed, list):
                        ctx = ' '.join(str(t).strip() for t in parsed if str(t).strip())
                except json.JSONDecodeError:
                    pass
            raw_context.append(ctx if ctx and ctx != '[]' else '')
            raw_audio.append(aud_feat)
            raw_video.append(vid_feat)
            audio_lengths.append(max(1, min(aud_feat.shape[0], self.args.seq_lens[1])))
            video_lengths.append(max(1, min(vid_feat.shape[0], self.args.seq_lens[2])))

        logger.info(
            "MELD %s split loaded %d/%d samples (drop: missing_meta=%d, mapping_missing=%d, missing_modal=%d)",
            self.mode, len(raw_text), len(keys),
            drop_stats['missing_meta'], drop_stats['mapping_missing'], drop_stats['missing_modal'],
        )

        if len(raw_text) == 0:
            raise ValueError(f"CRITICAL: No MELD samples loaded from {data_path}. drop_stats={drop_stats}")

        self.rawText = np.array(raw_text)
        self.rawContext = raw_context
        self.labels = {'M': labels_m}
        self.vision = pad_sequence_numpy(raw_video, self.args.seq_lens[2], self.args.feature_dims[2])
        self.audio = pad_sequence_numpy(raw_audio, self.args.seq_lens[1], self.args.feature_dims[1])

        if self.args.use_PLM:
            self.text = self.PLM_tokenizer(self.rawText)

        if not self.args.need_data_aligned:
            self.audio_lengths = np.array(audio_lengths)
            self.vision_lengths = np.array(video_lengths)

    def __init_iemocap(self):
        import re
        import glob
        
        # 1. Detect iemocap_text CSV directory
        iemocap_text_dir = None
        for p in [os.path.join(self.args.root_dataset_dir, 'iemocap_text'),
                  os.path.join(os.path.dirname(self.args.root_dataset_dir), 'iemocap_text')]:
            if os.path.exists(p):
                iemocap_text_dir = p
                break
                
        # 2. Robust path detection for IEMOCAP_full_release (fallback)
        search_paths = [
            os.path.join(self.args.root_dataset_dir, 'IEMOCAP', 'IEMOCAP_full_release'),
            os.path.join(self.args.root_dataset_dir, 'IEMOCAP_Origin', 'IEMOCAP_full_release'),
            os.path.join(self.args.root_dataset_dir, 'IEMOCAP_full_release'),
            os.path.join(os.path.dirname(self.args.root_dataset_dir), 'IEMOCAP', 'IEMOCAP_full_release'),
            os.path.join(os.path.dirname(self.args.root_dataset_dir), 'IEMOCAP_Origin', 'IEMOCAP_full_release'),
        ]
        iemocap_dir = None
        for p in search_paths:
            if os.path.exists(p):
                iemocap_dir = p
                break

        def build_multimodal_text(meta):
            """Build text: utterance-first ordering ensures utterance is never truncated.
            
            If --use_context: "utterance [speaker] Context: ..."
            If no context:    "utterance"  (matching UniMSE's proven approach)
            
            Truncation is right-side, so only context tail gets clipped.
            """
            text = str(meta.get('text', '')).strip()
            if not getattr(self.args, 'use_context', False):
                return text
            # Utterance-first: truncation only cuts context tail
            parts = [text]
            speaker = str(meta.get('speaker', '')).strip()
            if speaker:
                parts.append(f'[{speaker}]')
            context_raw = meta.get('context', '')
            context_text = ''
            if isinstance(context_raw, str):
                context_raw = context_raw.strip()
                if context_raw and context_raw.startswith('['):
                    try:
                        parsed = json.loads(context_raw)
                        if isinstance(parsed, list):
                            context_text = ' '.join(str(t).strip() for t in parsed if str(t).strip())
                        else:
                            context_text = str(parsed).strip()
                    except json.JSONDecodeError:
                        context_text = context_raw
                elif context_raw:
                    context_text = context_raw
            if context_text:
                parts.append(f'Context: {context_text}')
            return ' '.join(parts)
        
        # 1. Detect iemocap_text CSV directory
        iemocap_text_dir = None
        for p in [os.path.join(self.args.root_dataset_dir, 'iemocap_text'),
                  os.path.join(os.path.dirname(self.args.root_dataset_dir), 'iemocap_text'),
                  os.path.join(self.args.root_dataset_dir, 'IEMOCAP', 'iemocap_text'),
                  os.path.join(os.path.dirname(self.args.root_dataset_dir), 'text_data', 'iemocap_text'),
                  'd:\\ProjectFiles\\exp_202603\\datasets\\text_data\\iemocap_text']:
            if os.path.exists(p):
                iemocap_text_dir = p
                break
                
        # 2. Robust path detection for IEMOCAP_full_release (fallback)
        search_paths = [
            os.path.join(self.args.root_dataset_dir, 'IEMOCAP', 'IEMOCAP_full_release'),
            os.path.join(self.args.root_dataset_dir, 'IEMOCAP_Origin', 'IEMOCAP_full_release'),
            os.path.join(self.args.root_dataset_dir, 'IEMOCAP_full_release'),
            os.path.join(os.path.dirname(self.args.root_dataset_dir), 'IEMOCAP', 'IEMOCAP_full_release'),
            os.path.join(os.path.dirname(self.args.root_dataset_dir), 'IEMOCAP_Origin', 'IEMOCAP_full_release'),
        ]
        iemocap_dir = None
        for p in search_paths:
            if os.path.exists(p):
                iemocap_dir = p
                break
        
        def parse_labels_and_texts():
            info = {}
            # 1. Try iemocap_text CSV directory
            if iemocap_text_dir:
                import csv
                logger.info(f"Using pre-parsed CSV files from: {iemocap_text_dir}")
                for split in ['train', 'valid', 'test']:
                    csv_path = os.path.join(iemocap_text_dir, f'iemocap_data_{split}.csv')
                    if os.path.exists(csv_path):
                        with open(csv_path, 'r', encoding='utf-8') as f:
                            reader = csv.DictReader(f)
                            for row in reader:
                                if 'vid_cid' in row and 'label' in row and 'text' in row:
                                    info[row['vid_cid']] = {
                                        'emo': row['label'].strip().lower(),
                                        'text': row['text'].strip(),
                                        'context': row.get('context', '').strip(),
                                        'speaker': row.get('speaker', '').strip(),
                                    }
                if info:
                    return info
            
            # 2. Try IEMOCAP_full_release directory
            if iemocap_dir:
                logger.info(f"Fallback to raw data text matching at: {iemocap_dir}")
                for s in range(1, 6):
                    d_emo = os.path.join(iemocap_dir, f'Session{s}', 'dialog', 'EmoEvaluation')
                    if os.path.exists(d_emo):
                        for txt in glob.glob(os.path.join(d_emo, '*.txt')):
                            with open(txt, 'r', encoding='utf-8') as f:
                                for line in f:
                                    if line.startswith('['):
                                        t_match = re.match(r'\[(.*?) - (.*?)\]', line.split('\t')[0])
                                        parts = line.strip().split('\t')
                                        if t_match and len(parts) >= 3:
                                            info[parts[1]] = {
                                                'emo': parts[2].strip().lower(),
                                                'text': '',
                                                'context': '',
                                                'speaker': '',
                                            }
                    d_trans = os.path.join(iemocap_dir, f'Session{s}', 'dialog', 'transcriptions')
                    if os.path.exists(d_trans):
                        for txt in glob.glob(os.path.join(d_trans, '*.txt')):
                            with open(txt, 'r', encoding='utf-8', errors='ignore') as f:
                                for line in f:
                                    match = re.match(r'(Ses\w+) \[.*\]:\s*(.*)', line)
                                    if match:
                                        uid = match.group(1)
                                        if uid in info:
                                            info[uid]['text'] = match.group(2).strip()
                if info:
                    return info
            
            # 3. If no text found, create dummy text and labels for testing
            logger.warning("No text or labels found. Creating dummy data for testing.")
            # Get keys from audio_dict
            keys = list(audio_dict.keys())[:100]  # Use first 100 keys for testing
            for key in keys:
                info[key] = {
                    'emo': 'neu',
                    'text': 'This is a dummy text for testing.',
                    'context': '[]',
                    'speaker': '',
                }
            return info
            
        data_path = self.args.dataPath # 期望路径类似于 datasets/iemocap_data_0610.pkl
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        split_idx = 0 if self.mode == 'train' else (1 if self.mode == 'valid' else 2)
        if len(data['audio'][split_idx]) == 0:
            split_idx = 2
            
        audio_dict = data['audio'][split_idx]
        video_dict = data['video'][split_idx]
        
        def pad_sequence_numpy(sequences, max_len, feature_dim):
            padded = []
            for seq in sequences:
                seq = np.atleast_2d(seq)
                if seq.shape[0] >= max_len:
                    padded.append(seq[:max_len, :])
                else:
                    pad = np.zeros((max_len - seq.shape[0], feature_dim), dtype=np.float32)
                    padded.append(np.vstack((seq, pad)))
            return np.array(padded, dtype=np.float32)
        
        iemocap_meta = parse_labels_and_texts()
        
        raw_audio, raw_video, raw_text, labels_m = [], [], [], []
        audio_lengths, video_lengths = [], []
        
        # IEMOCAP Emotion Maps
        emo_map4 = {'ang': 'angry', 'hap': 'happy', 'exc': 'happy', 'sad': 'sad', 'neu': 'neutral'}
        emo_map6 = {'ang': 'angry', 'hap': 'happy', 'exc': 'excited', 'sad': 'sad', 'neu': 'neutral', 'fru': 'frustrated'}
        emo_map = emo_map4 if self.args.num_classes == 4 else emo_map6
        
        label_index_mapping = self.args.label_index_mapping
        drop_stats = {
            'missing_meta': 0,
            'unknown_emo': 0,
            'mapping_missing': 0,
            'missing_modal': 0,
        }
        
        keys = list(audio_dict.keys())
        for k in keys:
            meta = iemocap_meta.get(k)
            if meta is None:
                drop_stats['missing_meta'] += 1
                continue
            raw_emo = str(meta.get('emo', '')).strip().lower()
            if raw_emo not in emo_map:
                drop_stats['unknown_emo'] += 1
                continue 
                
            mapped_emo = emo_map[raw_emo]
            if mapped_emo not in label_index_mapping:
                drop_stats['mapping_missing'] += 1
                continue

            aud_feat = audio_dict.get(k)
            vid_feat = video_dict.get(k)
            if aud_feat is None or vid_feat is None:
                drop_stats['missing_modal'] += 1
                continue
                
            labels_m.append(label_index_mapping[mapped_emo])
            raw_text.append(build_multimodal_text(meta))
            
            raw_audio.append(aud_feat)
            raw_video.append(vid_feat)
            audio_lengths.append(max(1, min(aud_feat.shape[0], self.args.seq_lens[1])))
            video_lengths.append(max(1, min(vid_feat.shape[0], self.args.seq_lens[2])))

        logger.info(
            "IEMOCAP %s split loaded %d/%d samples (drop: missing_meta=%d, unknown_emo=%d, mapping_missing=%d, missing_modal=%d)",
            self.mode,
            len(raw_text),
            len(keys),
            drop_stats['missing_meta'],
            drop_stats['unknown_emo'],
            drop_stats['mapping_missing'],
            drop_stats['missing_modal'],
        )
            
        if len(raw_text) == 0:
            raise ValueError(
                f"CRITICAL ERROR: No samples were loaded from {data_path}. "
                f"drop_stats={drop_stats}"
            )
            
        self.rawText = np.array(raw_text)
        self.labels = {'M': labels_m}
        self.vision = pad_sequence_numpy(raw_video, self.args.seq_lens[2], self.args.feature_dims[2])
        self.audio = pad_sequence_numpy(raw_audio, self.args.seq_lens[1], self.args.feature_dims[1])
        
        if self.args.use_PLM:
            self.text = self.PLM_tokenizer(self.rawText)
            
        if not self.args.need_data_aligned:
            self.audio_lengths = np.array(audio_lengths)
            self.vision_lengths = np.array(video_lengths)

    def __init_cherma(self):
        return self.__init_meld()

    def __init_mosi(self):
        with open(self.args.dataPath, 'rb') as f:
            data = pickle.load(f)
            if self.args.use_PLM:
                self.text = data[self.mode]['raw_text']
                self.text = self.PLM_tokenizer(self.text)

        self.vision = data[self.mode]['vision'].astype(np.float32)
        self.audio = data[self.mode]['audio'].astype(np.float32)
        self.rawText = data[self.mode]['raw_text']
        self.ids = data[self.mode]['id']

        self.labels = {
            'M': data[self.mode][self.args.train_mode+'_labels'].astype(np.float32)
        }

        if self.args.need_label_prefix:
            labels = self.labels['M']
            label_prefix = []
            for i in range(len(labels)):
                if labels[i] < 0:
                    label_prefix.append(f'negative,{labels[i].item():.{1}f}')
                elif labels[i] > 0:
                    label_prefix.append(f'positive,{labels[i].item():.{1}f}')
                else:
                    label_prefix.append(f'neutral,{labels[i].item():.{1}f}')
            self.labels_prefix = label_prefix

        if self.args.datasetName == 'sims':
            for m in "TAV":
                self.labels[m] = data[self.mode][self.args.train_mode+'_labels_'+m]

        logger.info(f"{self.mode} samples: {self.labels['M'].shape}")

        if not self.args.need_data_aligned:
            self.audio_lengths = data[self.mode]['audio_lengths']
            self.vision_lengths = data[self.mode]['vision_lengths']
            self.text_lengths = self.args.seq_lens[0]
        self.audio[self.audio == -np.inf] = 0
        self.vision[self.vision != self.vision] = 0

        if  self.args.need_normalized:
            self.__normalize()
    
    def __init_mosei(self):
        return self.__init_mosi()

    def __init_sims(self):
        return self.__init_mosi()

    def __init_simsv2(self):
        return self.__init_mosi()

    def __truncated(self):
        # NOTE: Here for dataset we manually cut the input into specific length.
        def Truncated(modal_features, length):
            if length == modal_features.shape[1]:
                return modal_features
            truncated_feature = []
            padding = np.array([0 for i in range(modal_features.shape[2])])
            for instance in modal_features:
                for index in range(modal_features.shape[1]):
                    if((instance[index] == padding).all()):
                        if(index + length >= modal_features.shape[1]):
                            truncated_feature.append(instance[index:index+20])
                            break
                    else:                        
                        truncated_feature.append(instance[index:index+20])
                        break
            truncated_feature = np.array(truncated_feature)
            return truncated_feature
                       
        text_length, audio_length, video_length = self.args.seq_lens
        self.vision = Truncated(self.vision, video_length)
        self.text = Truncated(self.text, text_length)
        self.audio = Truncated(self.audio, audio_length)

    def __normalize(self):
        # (num_examples,max_len,feature_dim) -> (max_len, num_examples, feature_dim)
        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))
        # for visual and audio modality, we average across time
        # here the original data has shape (max_len, num_examples, feature_dim)
        # after averaging they become (1, num_examples, feature_dim)
        self.vision = np.mean(self.vision, axis=0, keepdims=True)
        self.audio = np.mean(self.audio, axis=0, keepdims=True)

        # remove possible NaN values
        self.vision[self.vision != self.vision] = 0
        self.audio[self.audio != self.audio] = 0

        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))

    def __len__(self):
        return len(self.labels['M'])

        # 这里text.shape是三维矩阵[sample_num,tokenizer_output,length]
        # tokenizer_output的3个维度分别是token_ids,mask(识别句子中padding的位置),segment_ids
    def get_seq_len(self):
        return (self.text.shape[2], self.audio.shape[1], self.vision.shape[1])

    def get_feature_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]

    def PLM_tokenizer (self, rawtexts):
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.pretrain_LM, trust_remote_code=True)

        # Decoder-only models (e.g., LLaMA) may not define pad_token by default.
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        if self.tokenizer.pad_token_id is None and self.tokenizer.pad_token is not None:
            self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)

        token_list = []
        for text in rawtexts:
            text_tokenizer = self.tokenizer(text,
                                 padding='max_length',  # 如果样本长度不满足最大长度则填充
                                 truncation=True,  # 截断至最大长度
                                 max_length=self.args.seq_lens[0],
                                 return_tensors = 'pt',
                                 add_special_tokens=False
                                )

            token_ids = text_tokenizer['input_ids'].squeeze(0)  # tensor of token ids  torch.Size([max_len])
            attn_masks = text_tokenizer['attention_mask'].squeeze(0)  # binary tensor with "0" for padded values and "1" for the other values  torch.Size([max_len])
            token_type_ids = [0] * len(token_ids)               #不区分上下句

            #调整维度
            input_ids = np.expand_dims(token_ids, 1)
            input_mask = np.expand_dims(attn_masks, 1)
            segment_ids = np.expand_dims(token_type_ids, 1)

            text_pretrain = np.concatenate([input_ids, input_mask, segment_ids], axis=1).T
            token_list.append(text_pretrain)

        # x_dimensions = [array.shape[1] for array in token_list]
        # # 计算 x 维度的平均值
        # average_x = np.mean(x_dimensions)
        # median_x = np.median(x_dimensions)
        token_list = np.array(token_list)
        return token_list


    def __getitem__(self, index):
        if self.args.train_mode == 'regression':
            sample = {
                'raw_text': self.rawText[index],
                'text': torch.Tensor(self.text[index]),
                'audio': torch.Tensor(self.audio[index]),
                'vision': torch.Tensor(self.vision[index]),
                'index': index,
                'id': self.ids[index],
                'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()},
                'labels_prefix': self.labels_prefix[index]
            }
        else:
            sample = {
                'raw_text': self.rawText[index],
                'text': torch.Tensor(self.text[index]),
                'audio': torch.Tensor(self.audio[index]),
                'vision': torch.Tensor(self.vision[index]),
                'index': index,
                'labels': {k: v[index] for k, v in self.labels.items()}
                # 'labels': {torch.Tensor(self.labels)},
            }

        if not self.args.need_data_aligned:
            sample['audio_lengths'] = self.audio_lengths[index]
            sample['vision_lengths'] = self.vision_lengths[index]
            sample['text_lengths'] = self.args.seq_lens[0]

        # Prompt-level context (string, not tensor)
        if getattr(self.args, 'prompt_context', False) and hasattr(self, 'rawContext'):
            sample['context_text'] = self.rawContext[index]

        return sample



def MMDataLoader(args):

    datasets = {
        'train': MMDataset(args, mode='train'),
        'valid': MMDataset(args, mode='valid'),
        'test': MMDataset(args, mode='test')
    }

    if 'seq_lens' in args:
        args.seq_lens = datasets['train'].get_seq_len() 

    _nw = args.num_workers
    dataLoader = {
        ds: DataLoader(datasets[ds],
                       batch_size=args.batch_size,
                       num_workers=_nw,
                       shuffle=(ds == 'train'),
                       pin_memory=True,
                       persistent_workers=(_nw > 0),
                       prefetch_factor=2 if _nw > 0 else None,
                       drop_last=(ds == 'train'))
        for ds in datasets.keys()
    }
    
    return dataLoader