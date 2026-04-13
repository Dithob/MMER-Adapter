import os
import sys
import logging
import collections
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger('MSA')

__all__ = ['Language_model']

class Language_model(nn.Module):
    def __init__(self, args, use_PLM=True):
        """
        language: en / cn
        """
        super(Language_model, self).__init__()
        
        self.model_type = args.model_type
        self.device = args.device
        self.language = args.language
        self.max_new_tokens = args.max_new_tokens
        self.datasetName = args.datasetName
        self.train_mode = args.train_mode
        self.task_specific_prompt = args.task_specific_prompt
        self.prompt_style = getattr(args, 'prompt_style', 'default')
        self._args = args  # keep reference for LoRA config
        self._lora_enabled = False
        
        if use_PLM:
            pretrained_model = args.pretrain_LM
            self._load_model(pretrained_model)
            # ── LoRA or Freeze ──
            if getattr(args, 'use_lora', False):
                self._apply_lora(args)
            else:
                # freeze all LLM parameters
                for param in self.model.parameters():
                    param.requires_grad = False
        else:
            print('please use PLM')
    
    def _load_model(self, pretrained_model):
        """根据模型类型加载不同的语言模型"""
        if self.model_type == 'chatglm3':
            self._load_chatglm3(pretrained_model)
        elif self.model_type == 'gemma':
            self._load_gemma(pretrained_model)
        elif self.model_type in ['qwen', 'qwen3.5', 'llama2', 'deepseek']:
            self._load_modelscope_model(pretrained_model)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _load_chatglm3(self, pretrained_model):
        """加载ChatGLM3模型"""
        from models.ChatGLM3.modeling_chatglm import ChatGLMForConditionalGeneration
        from models.ChatGLM3.tokenization_chatglm import ChatGLMTokenizer
        
        self.model = ChatGLMForConditionalGeneration.from_pretrained(
            pretrained_model, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16
        ).half()
        self.tokenizer = ChatGLMTokenizer.from_pretrained(
            pretrained_model, 
            trust_remote_code=True
        )
    
    def _load_gemma(self, pretrained_model):
        """加载Gemma模型（使用HuggingFace transformers）"""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model,
            padding_side='left',
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model,
            trust_remote_code=True,
            dtype=torch.float16
        )
        
        # 缓存 PLE 维度信息，用于构造 dummy per_layer_inputs
        base_cfg = self.model.config
        text_cfg = getattr(base_cfg, 'text_config', base_cfg)
        self._gemma_ple_dim = getattr(text_cfg, 'hidden_size_per_layer_input', 0)
        self._gemma_num_layers = getattr(text_cfg, 'num_hidden_layers', 0)
        
        # Gemma 使用 eos_token 作为 pad_token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
    
    def _load_modelscope_model(self, pretrained_model):
        """加载ModelScope模型（Qwen, Llama2等）"""
        from modelscope import AutoTokenizer, AutoModelForCausalLM
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model,
            padding_side='left',
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model,
            trust_remote_code=True,
            dtype=torch.bfloat16
        )
        # NOTE: 不再调用 .half()，保持原生 bf16 精度
        # RTX 4090D 原生支持 bf16，比 fp16 更稳定且无需 GradScaler
        
        # 启用 gradient checkpointing 节省显存（允许更大 batch）
        # use_reentrant=False 兼容冻结权重场景（非重入模式正确处理无梯度的中间层）
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        if self.model_type in ['qwen', 'qwen3.5']:
            # Qwen/Qwen3.5 使用相同的特殊 token 设置
            self.eos_token_id = self.tokenizer.convert_tokens_to_ids('<|endoftext|>')
            self.tokenizer.eos_token_id = self.eos_token_id
            self.tokenizer.pad_token_id = self.eos_token_id
            self.bos_token_id = self.tokenizer.convert_tokens_to_ids('<|im_start|>')
            self.tokenizer.bos_token_id = self.bos_token_id
        elif self.model_type == 'llama2':
            self.tokenizer.pad_token_id = 0
            self.eos_token_id = self.tokenizer.convert_tokens_to_ids('</s>')
        elif self.model_type == 'deepseek':
            # DeepSeek模型使用默认的token设置
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
    
    def text_embedding(self, text_ids):
        # When LoRA is enabled, self.model is PeftModel; unwrap to reach original model
        if self._lora_enabled:
            # PeftModel.base_model (LoraModel) .model → original CausalLM
            base_model = self.model.base_model.model
        else:
            base_model = self.model
        
        if self.model_type == 'gemma':
            embeddings = base_model.get_input_embeddings()
        else:
            embeddings = base_model.base_model.get_input_embeddings()
        return embeddings(text_ids)
    
    def _apply_lora(self, args):
        """Apply LoRA adapters to the LLM for fine-tuning.
        
        References Emotion-LLaMA-v2's base_model.py init_llm() implementation:
        first freeze all params, then wrap with peft.get_peft_model() which
        automatically unfreezes the injected LoRA parameters.
        """
        from peft import LoraConfig, get_peft_model, TaskType
        
        # Step 1: freeze all base model params
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Step 2: configure LoRA
        target_modules = [m.strip() for m in args.lora_target_modules.split(',')]
        
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Step 3: wrap model with LoRA
        self.model = get_peft_model(self.model, lora_config)
        self._lora_enabled = True
        
        # Log trainable parameters
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(f"LoRA enabled: r={args.lora_r}, alpha={args.lora_alpha}, "
                    f"targets={target_modules}")
        logger.info(f"LoRA trainable params: {trainable:,} / {total:,} "
                    f"({100 * trainable / total:.2f}%)")
    
    def forward(self, fusion_embedding, labels, input_attn_mask=None):
        """
        Args:
            fusion_embedding: the "concatenate" result of multimodal low rank fusion and text embedding
            label: ground_truth
            input_attn_mask: optional [B, seq_len] mask for dynamic modal attention (0 = masked)
        """
        fusion_embedding = self.multimodal_prompt_wrap(fusion_embedding)
        
        if self.model_type == 'chatglm3':
            return self._forward_chatglm3(fusion_embedding, labels)
        elif self.model_type in ['qwen', 'qwen3.5', 'llama2', 'deepseek', 'gemma']:
            return self._forward_modelscope(fusion_embedding, labels, input_attn_mask=input_attn_mask)
        else:
            raise ValueError(f"Unsupported model type in forward: {self.model_type}")
    
    def _forward_chatglm3(self, fusion_embedding, labels):
        """ChatGLM3的前向传播"""
        opt_tokens, labels = self.input_processing(fusion_embedding, labels, mode='train')
        
        with torch.amp.autocast(device_type='cuda'):
            output = self.model(
                input_ids=opt_tokens, 
                input_fusion=fusion_embedding, 
                labels=labels
            )
        
        return output
    
    def _forward_modelscope(self, fusion_embedding, labels, input_attn_mask=None):
        """ModelScope/Gemma模型的前向传播"""
        opt_tokens, atts_bos, atts_fusion, labels, labels_atts, opt_input_ids = self.input_processing(
            fusion_embedding, labels, mode='train', input_attn_mask=input_attn_mask
        )
        
        # 构建 attention_mask：有 bos 的拼接 bos，否则只用 fusion + labels
        if atts_bos is not None:
            attention_mask = torch.cat([atts_bos, atts_fusion, labels_atts], dim=1)
        else:
            attention_mask = torch.cat([atts_fusion, labels_atts], dim=1)
        
        # 确保 pad_token_id 有值，避免后续生成警告
        if getattr(self.model.config, 'pad_token_id', None) is None and hasattr(self.tokenizer, 'pad_token_id'):
            try:
                self.model.config.pad_token_id = self.tokenizer.pad_token_id
            except AttributeError:
                pass  # Gemma4Config 等嵌套 config 不支持直接设置顶层 pad_token_id

        if self.model_type == 'gemma' and self._gemma_ple_dim > 0:
            # Gemma 4 PLE 机制：当仅传 inputs_embeds 时，多模态模型的 wrap(Gemma4Model)
            # 在自己内部前向计算时，忽略传入的 per_layer_inputs 参数，并强行反向查找 input_ids
            # （创建 [B, seq, vocab, hidden] 的巨大张量导致 937GiB OOM）。
            # 解决方案：完全绕过多模态外壳，直接调用底层的 language_model (Gemma4TextModel)，
            # 这样才能接收我们传入的零值 dummy_pli 并跳过 OOM 逻辑。
            inner_text_model = getattr(self.model.model, 'language_model', self.model.model)
            
            if opt_input_ids is not None and hasattr(inner_text_model, 'get_per_layer_inputs'):
                per_layer_inputs = inner_text_model.get_per_layer_inputs(
                    input_ids=opt_input_ids.to(self.device),
                    inputs_embeds=None
                )
            else:
                per_layer_inputs = torch.zeros(
                    opt_tokens.shape[0], opt_tokens.shape[1], self._gemma_num_layers, self._gemma_ple_dim,
                    dtype=opt_tokens.dtype, device=opt_tokens.device
                )
                
            with torch.amp.autocast(device_type='cuda'):
                base_outputs = inner_text_model(
                    inputs_embeds=opt_tokens,
                    attention_mask=attention_mask,
                    per_layer_inputs=per_layer_inputs,
                    return_dict=True
                )
                hidden_states = base_outputs.last_hidden_state
                logits = self.model.lm_head(hidden_states)
                
                if labels is not None:
                    if hasattr(self.model, 'loss_function'):
                        base_cfg = self.model.config
                        text_cfg = getattr(base_cfg, 'text_config', base_cfg)
                        vocab_size = getattr(self.model, 'vocab_size', getattr(text_cfg, 'vocab_size', 262144))
                        loss = self.model.loss_function(logits, labels, vocab_size)
                    else:
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = labels[..., 1:].contiguous()
                        loss = torch.nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                else:
                    loss = None
            
            from transformers.modeling_outputs import CausalLMOutputWithPast
            output = CausalLMOutputWithPast(loss=loss, logits=logits)
        else:
            model_kwargs = dict(
                inputs_embeds=opt_tokens,
                attention_mask=attention_mask,
                return_dict=True,
                labels=labels
            )
            with torch.amp.autocast(device_type='cuda'):
                output = self.model(**model_kwargs)

        return output
    
    def generate(self, fusion_embedding, input_attn_mask=None):
        """生成预测结果"""
        fusion_embedding = self.multimodal_prompt_wrap(fusion_embedding)
        
        if self.model_type == 'chatglm3':
            return self._generate_chatglm3(fusion_embedding)
        elif self.model_type in ['qwen', 'qwen3.5', 'llama2', 'deepseek', 'gemma']:
            return self._generate_modelscope(fusion_embedding, input_attn_mask=input_attn_mask)
        else:
            raise ValueError(f"Unsupported model type in generate: {self.model_type}")
    
    def _generate_chatglm3(self, fusion_embedding):
        """ChatGLM3的生成"""
        # ChatGLM3's SentencePiece tokenizer encodes standalone digits as 2 tokens:
        # ▁ (word boundary, token 30910) + digit. Training labels also tokenize this way,
        # so max_new_tokens must be +1 to account for the ▁ prefix.
        effective_max_tokens = self.max_new_tokens + 1
        if self.train_mode == 'regression':
            gen_kwargs = {"max_new_tokens": effective_max_tokens, "num_beams": 1, "do_sample": False, "top_k": 10}
        else:
            gen_kwargs = {"max_new_tokens": effective_max_tokens, "num_beams": 1, "do_sample": False, "top_k": 10}
        
        opt_tokens, _ = self.input_processing(fusion_embedding, mode='generate')

        context_length = opt_tokens.size(1)
        all_responses = []

        # NOTE: Do NOT pass attention_mask or pad_token_id here.
        # ChatGLM3's stream_generate handles them internally.
        for outputs in self.model.stream_generate(
            opt_tokens,
            **gen_kwargs,
            input_fusion=fusion_embedding
        ):
            outputs = outputs[:, context_length:].tolist()
            response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # # Debug: print first batch's raw decode results (only once)
        # if not hasattr(self, '_debug_printed'):
        #     self._debug_printed = True
        #     print(f"[DEBUG generate] raw responses (first 5): {response[:5]}")
        #     print(f"[DEBUG generate] raw output token ids (first 5): {outputs[:5]}")

        for x in response:
            x = x.strip()
            if self.train_mode == 'regression':
                try:
                    value = float(
                        x.replace('–', '-').replace('一', '-').replace('：', '').replace('/', '').replace('(', '').replace(':', '')
                    )
                except ValueError:
                    value = 0.0
            else:
                try:
                    value = float(x)
                except ValueError:
                    value = 0.0
            all_responses.append(value)
        
        return all_responses
    
    def _generate_modelscope(self, fusion_embedding, input_attn_mask=None):
        """ModelScope模型的生成"""
        opt_tokens, atts_bos, atts_fusion, _, _, opt_input_ids = self.input_processing(
            fusion_embedding, mode='generate', input_attn_mask=input_attn_mask
        )
        
        if self.model_type in ['qwen', 'qwen3.5']:
            attention_mask = torch.cat([atts_bos, atts_fusion], dim=1)
            gen_kwargs = {
                "num_beams": 1,
                "do_sample": False,
                "bos_token_id": self.tokenizer.bos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "max_new_tokens": self.max_new_tokens
            }
        elif self.model_type == 'llama2':
            attention_mask = torch.cat([atts_bos, atts_fusion], dim=1)
            gen_kwargs = {
                "num_beams": 1,
                "do_sample": False,
                "top_p": None,
                "max_new_tokens": self.max_new_tokens
            }
        else:  # deepseek, gemma
            attention_mask = atts_fusion  # 无 bos 的模型直接使用 fusion 的 attention_mask
            gen_kwargs = {
                "num_beams": 1,
                "do_sample": False,
                "max_new_tokens": self.max_new_tokens
            }

        # 获取 pad_token_id：优先从 config，回退到 tokenizer
        pad_id = getattr(self.model.config, 'pad_token_id', None) or getattr(self.tokenizer, 'pad_token_id', None)

        if self.model_type == 'gemma' and self._gemma_ple_dim > 0:
            # Gemma 4 PLE: model.generate() 同样会触发反向 embedding 查找导致 OOM
            # 手动贪心解码：通过内部 TextModel + lm_head 逐步生成
            with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
                current_embeds = opt_tokens
                current_input_ids = opt_input_ids
                generated_ids = []
                # 关键：调用 self.model.model.language_model 而不是 self.model.model
                inner_text_model = getattr(self.model.model, 'language_model', self.model.model)
                for _ in range(self.max_new_tokens):
                    batch_size = current_embeds.shape[0]
                    if current_input_ids is not None and hasattr(inner_text_model, 'get_per_layer_inputs'):
                        per_layer_inputs = inner_text_model.get_per_layer_inputs(
                            input_ids=current_input_ids.to(self.device),
                            inputs_embeds=None
                        )
                    else:
                        per_layer_inputs = torch.zeros(
                            current_embeds.shape[0], current_embeds.shape[1], self._gemma_num_layers, self._gemma_ple_dim,
                            dtype=current_embeds.dtype, device=current_embeds.device
                        )
                    base_out = inner_text_model(
                        inputs_embeds=current_embeds,
                        attention_mask=attention_mask,
                        per_layer_inputs=per_layer_inputs,
                        return_dict=True
                    )
                    last_hidden = base_out.last_hidden_state[:, -1:, :]
                    next_logits = self.model.lm_head(last_hidden)  # [B, 1, vocab]
                    next_token_id = next_logits.argmax(dim=-1)  # [B, 1]
                    generated_ids.append(next_token_id)
                    # 准备下一步输入
                    next_embed = self.text_embedding(next_token_id)
                    current_embeds = torch.cat([current_embeds, next_embed], dim=1)
                    if current_input_ids is not None:
                        current_input_ids = torch.cat([current_input_ids, next_token_id], dim=1)
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.ones(batch_size, 1, dtype=attention_mask.dtype, device=attention_mask.device)
                    ], dim=1)

                outputs = torch.cat(generated_ids, dim=1)  # [B, max_new_tokens]
        else:
            outputs = self.model.generate(
                inputs_embeds=opt_tokens,
                attention_mask=attention_mask,
                pad_token_id=pad_id,
                **gen_kwargs
            )

        if self.model_type in ['qwen', 'qwen3.5']:
            # When using inputs_embeds, model.generate() returns only new token IDs.
            # Extract only the last max_new_tokens tokens (the actual generated output).
            new_tokens = outputs[:, -self.max_new_tokens:]
            responses = self.tokenizer.batch_decode(
                new_tokens, 
                add_special_tokens=False, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
        else:  # llama2, deepseek, gemma
            new_tokens = outputs[:, -self.max_new_tokens:]
            responses = self.tokenizer.batch_decode(
                new_tokens, 
                add_special_tokens=False, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
        
        all_responses = []
        for response in responses:
            if self.train_mode == 'regression':
                try:
                    value = float(
                        response.replace('–', '-').replace('一', '-').replace('：', '').replace('/', '').replace('(', '').replace(':', '')
                    )
                except ValueError:
                    value = 0.0
            else:
                try:
                    value = float(response)
                except ValueError:
                    value = 0.0
            all_responses.append(value)
        
        return all_responses
    
    def input_processing(self, fusion_embedding, labels=None, mode=None, input_attn_mask=None):
        """
        Args:
            fusion_embedding: the "concatenate" result of multimodal low rank fusion and text embedding
            labels: ground_truth
            mode: 'train' or 'generate'
            input_attn_mask: optional [B, seq_len] mask for dynamic modal attention (0 = masked)
        """
        if self.model_type == 'chatglm3':
            return self._input_processing_chatglm3(fusion_embedding, labels, mode)
        elif self.model_type in ['qwen', 'qwen3.5', 'llama2', 'deepseek', 'gemma']:
            return self._input_processing_modelscope(fusion_embedding, labels, mode, input_attn_mask=input_attn_mask)
        else:
            raise ValueError(f"Unsupported model type in input_processing: {self.model_type}")
    
    def _input_processing_chatglm3(self, fusion_embedding, labels=None, mode=None):
        """ChatGLM3的输入处理"""
        input_lengths = fusion_embedding[:, :, 0]
        fusion_empty = (torch.ones(input_lengths.size(), dtype=torch.long).to(self.device).fill_(0))
        
        task_prompt = self.get_task_prompt()
        prompt_broadcasted = task_prompt.expand(fusion_empty.size(0), -1)
        
        opt_tokens = torch.cat([fusion_empty, prompt_broadcasted], dim=1)
        opt_tokens, labels = self.input_labels_construct(opt_tokens, labels, mode)
        
        return opt_tokens, labels
    
    def _input_processing_modelscope(self, fusion_embedding, labels=None, mode=None, input_attn_mask=None):
        """ModelScope模型的输入处理"""
        batch_size = fusion_embedding.shape[0]
        
        task_prompt = self.get_task_prompt()
        prompt_ids = task_prompt.expand(batch_size, -1)
        task_prompt_embedding = self.text_embedding(prompt_ids)
        
        opt_tokens = torch.cat([fusion_embedding, task_prompt_embedding], dim=1)
        atts_fusion = torch.ones(opt_tokens.size()[:-1], dtype=torch.long).to(self.device)
        
        # ── Dynamic modal attention mask (for raw AV token bypass) ──
        # Zeros out attention for AV tokens of samples with missing (all-zero) modality
        if input_attn_mask is not None:
            prefix_len = getattr(self, '_wrap_prefix_len', 0)
            mask_len = input_attn_mask.shape[1]
            atts_fusion[:, prefix_len:prefix_len + mask_len] *= input_attn_mask.to(self.device)
        # 针对 Gemma 4 等模型构建真实的 input_ids，fusion位由 pad_token_id 填充
        pad_id = getattr(self.tokenizer, 'pad_token_id', 0)
        fusion_ids = torch.full(fusion_embedding.shape[:-1], pad_id, dtype=torch.long, device=self.device)
        opt_input_ids = torch.cat([fusion_ids, prompt_ids], dim=1)
        
        if self.model_type in ['qwen', 'qwen3.5']:
            bos_ids = torch.ones([batch_size, 1], dtype=atts_fusion.dtype, device=self.device) * self.tokenizer.bos_token_id
            bos_embeds = self.text_embedding(bos_ids)
            atts_bos = atts_fusion[:, :1]
            opt_tokens = torch.cat([bos_embeds, opt_tokens], dim=1)
            opt_input_ids = torch.cat([bos_ids, opt_input_ids], dim=1)
        else:  # llama2, deepseek, gemma
            bos_embeds = None
            atts_bos = None
        
        opt_tokens, labels, labels_atts, opt_input_ids = self.input_labels_construct(opt_tokens, labels, mode, opt_input_ids)
        
        if self.model_type in ['qwen', 'qwen3.5']:
            return opt_tokens, atts_bos, atts_fusion, labels, labels_atts, opt_input_ids
        else:  # llama2, deepseek, gemma
            return opt_tokens, None, atts_fusion, labels, labels_atts, opt_input_ids
    
    def input_labels_construct(self, opt_tokens, labels=None, mode=None, opt_input_ids=None):
        """
        Args:
            opt_tokens: the "concatenate" size of multimodal fusion, text embedding and prompt
            labels: ground_truth
        """
        batch_size = opt_tokens.shape[0]
        
        if mode == "train":
            if self.train_mode == "regression":
                if self.model_type in ['qwen', 'qwen3.5']:
                    label_template = [f"+{label.item():.{1}f}" if label >= 0 else f"{label.item():.{1}f}" for label in labels]
                else:
                    label_template = [f"{label.item():.{1}f}" for label in labels]
            else:
                # Append EOS token to classification labels to help model learn generation stopping
                # This aligns with MSE-Qwen3.5-2B's append_eos_to_label behavior
                eos_suffix = ''
                if self.model_type in ['qwen', 'qwen3.5'] and hasattr(self.tokenizer, 'eos_token') and self.tokenizer.eos_token:
                    eos_suffix = self.tokenizer.eos_token
                label_template = [f"{label.item()}{eos_suffix}" for label in labels]
            
            if self.model_type == 'chatglm3':
                labels_id = self.tokenizer(label_template, padding=True, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.device)
                labels_matrix = torch.empty_like(opt_tokens).fill_(-100).long().to(self.device)
                opt_tokens = torch.cat([opt_tokens, labels_id], dim=1)
                labels = torch.cat([labels_matrix, labels_id], dim=1)
                if opt_input_ids is not None:
                    opt_input_ids = torch.cat([opt_input_ids, labels_id], dim=1)
                    return opt_tokens, labels, opt_input_ids
                return opt_tokens, labels
            else:
                labels_dict = self.tokenizer(label_template, padding=True, return_tensors="pt", add_special_tokens=False).to(self.device)
                labels_id = labels_dict["input_ids"]
                labels_atts = labels_dict["attention_mask"]
                
                labels_embedding = self.text_embedding(labels_id)
                labels_matrix = torch.empty(opt_tokens.size(0), opt_tokens.size(1)).fill_(-100).long().to(self.device)
                opt_tokens = torch.cat([opt_tokens, labels_embedding], dim=1)
                labels = torch.cat([labels_matrix, labels_id], dim=1)
                
                if opt_input_ids is not None:
                    opt_input_ids = torch.cat([opt_input_ids, labels_id], dim=1)
                    return opt_tokens, labels, labels_atts, opt_input_ids
                return opt_tokens, labels, labels_atts
        else:
            if self.model_type == 'chatglm3':
                if opt_input_ids is not None:
                    return opt_tokens, None, opt_input_ids
                return opt_tokens, None
            else:
                if opt_input_ids is not None:
                    return opt_tokens, None, None, opt_input_ids
                return opt_tokens, None, None
    
    def get_task_prompt(self):
        """获取任务特定的prompt"""
        prompt_text = self.task_specific_prompt
        prompt_ids = self.tokenizer(prompt_text, padding=True, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.device)
        return prompt_ids
    
    def multimodal_prompt_wrap(self, fusion_embeddings):
        """
        Args:
            Wrap the input with a special token
        """
        if self.language == "en":
            prompt = '{question}\n\n <Multimodal><MultimodalHere></Multimodal>'
            special_token = '<MultimodalHere>'
        else:
            prompt = '{问题}\n\n <多模态><MultimodalHere></多模态>'
            special_token = '<MultimodalHere>'
        
        batch_size = fusion_embeddings.shape[0]
        
        if self.model_type == 'chatglm3':
            p_before, p_after = prompt.split(special_token)
            p_before_tokens = self.tokenizer(p_before, return_tensors="pt", add_special_tokens=True).to(self.device)
            p_after_tokens = self.tokenizer(p_after, return_tensors="pt", add_special_tokens=False).to(self.device)
            p_before_embeds = self.text_embedding(p_before_tokens.input_ids).expand(batch_size, -1, -1)
            p_after_embeds = self.text_embedding(p_after_tokens.input_ids).expand(batch_size, -1, -1)
        else:
            if self.prompt_style == 'enhanced':
                if self.language == "en":
                    prompt = 'Based on the following multimodal signals: <Multimodal><MultimodalHere></Multimodal>'
                else:
                    prompt = '基于以下多模态信号：<多模态><MultimodalHere></多模态>'
            else:
                if self.language == "en":
                    prompt = '<Multimodal><MultimodalHere></Multimodal>'
                else:
                    prompt = '<多模态><MultimodalHere></多模态>'
            
            p_before, p_after = prompt.split(special_token)
            p_before_tokens = self.tokenizer(p_before, return_tensors="pt", add_special_tokens=True).to(self.device)
            p_after_tokens = self.tokenizer(p_after, return_tensors="pt", add_special_tokens=False).to(self.device)
            p_before_embeds = self.text_embedding(p_before_tokens.input_ids.expand(batch_size, -1))
            p_after_embeds = self.text_embedding(p_after_tokens.input_ids.expand(batch_size, -1))
        
        # Track prefix/suffix lengths for dynamic attention mask alignment
        self._wrap_prefix_len = p_before_embeds.shape[1]
        self._wrap_suffix_len = p_after_embeds.shape[1]
        
        wrapped_fusion_embeddings = torch.cat([p_before_embeds, fusion_embeddings, p_after_embeds], dim=1)
        return wrapped_fusion_embeddings
