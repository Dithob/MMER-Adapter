import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.llm_backends.factory import build_llm_backend

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
        self._args = args
        self._lora_enabled = False
        self._wrap_prefix_len = 0
        self._wrap_suffix_len = 0

        # ── Label format: index (default) or text ──
        self.label_format = getattr(args, 'label_format', 'index')
        self.label_index_mapping = getattr(args, 'label_index_mapping', {})
        # Build reverse mapping: idx → label name
        self._idx_to_name = {v: k for k, v in self.label_index_mapping.items()}
        # Auto-adjust max_new_tokens for text label format
        # Emotion words like 'surprise', 'frustrated' may need 2-3 tokens
        if self.label_format == 'text' and self.train_mode == 'classification':
            self.max_new_tokens = max(self.max_new_tokens, 3)

        self.backend = build_llm_backend(args)
        self._gemma_ple_dim = getattr(self.backend, '_gemma_ple_dim', 0)
        self._gemma_num_layers = getattr(self.backend, '_gemma_num_layers', 0)

        if use_PLM:
            pretrained_model = args.pretrain_LM
            self.model, self.tokenizer = self.backend.load(pretrained_model)

            if getattr(args, 'use_lora', False):
                self._apply_lora(args)
            else:
                for param in self.model.parameters():
                    param.requires_grad = False
        else:
            print('please use PLM')

        # Track warmup state for two-stage training
        self._lora_warmup_active = False

    def text_embedding(self, text_ids):
        if self._lora_enabled:
            base_model = self.model.base_model.model
        else:
            base_model = self.model

        return self.backend.text_embedding(base_model, self.tokenizer, text_ids)

    def _apply_lora(self, args):
        """Apply LoRA adapters to the LLM for fine-tuning."""
        from peft import LoraConfig, get_peft_model, TaskType

        for param in self.model.parameters():
            param.requires_grad = False

        target_modules = [m.strip() for m in args.lora_target_modules.split(',')]

        # ── ChatGLM3 module name remapping ──
        # ChatGLM3 uses different naming conventions:
        #   q_proj/k_proj/v_proj → query_key_value  (merged QKV)
        #   o_proj              → dense             (attention output)
        #   gate_proj/up_proj   → dense_h_to_4h     (FFN up, merged)
        #   down_proj           → dense_4h_to_h     (FFN down)
        if self.model_type == 'chatglm3':
            _CHATGLM3_MAP = {
                'q_proj': 'query_key_value',
                'k_proj': 'query_key_value',
                'v_proj': 'query_key_value',
                'o_proj': 'dense',
                'gate_proj': 'dense_h_to_4h',
                'up_proj': 'dense_h_to_4h',
                'down_proj': 'dense_4h_to_h',
            }
            remapped = []
            for m in target_modules:
                mapped = _CHATGLM3_MAP.get(m, m)
                if mapped not in remapped:
                    remapped.append(mapped)
            logger.info(f"ChatGLM3 LoRA target remapping: {target_modules} → {remapped}")
            target_modules = remapped

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        self.model = get_peft_model(self.model, lora_config)
        self._lora_enabled = True

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(f"LoRA enabled: r={args.lora_r}, alpha={args.lora_alpha}, targets={target_modules}")
        logger.info(f"LoRA trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    def enable_lora(self):
        """Enable LoRA adapters (Stage 2 of two-stage training).
        Re-enables gradient computation on all LoRA parameters.
        """
        if not self._lora_enabled:
            return
        count = 0
        for name, param in self.model.named_parameters():
            if 'lora_' in name:
                param.requires_grad = True
                count += 1
        self._lora_warmup_active = False
        logger.info(f"LoRA enabled: {count} parameter tensors unfrozen")

    def disable_lora(self):
        """Disable LoRA adapters (Stage 1 warmup: train only external modules).
        Freezes all LoRA parameters while keeping adapter/mixer/MoE trainable.
        """
        if not self._lora_enabled:
            return
        count = 0
        for name, param in self.model.named_parameters():
            if 'lora_' in name:
                param.requires_grad = False
                count += 1
        self._lora_warmup_active = True
        logger.info(f"LoRA disabled (warmup): {count} parameter tensors frozen")

    def forward_encode(self, fusion_embedding, input_attn_mask=None, context_text=None):
        """Get LLM hidden states without generative loss (for cls_head mode).
        Returns the hidden state at the last position: [batch, hidden_dim].
        """
        fusion_embedding = self.multimodal_prompt_wrap(fusion_embedding, context_text=context_text)

        if self.model_type == 'chatglm3':
            opt_tokens, _ = self.input_processing(fusion_embedding, mode='generate')
            with torch.amp.autocast(device_type='cuda'):
                outputs = self.model(input_ids=opt_tokens, input_fusion=fusion_embedding,
                                     output_hidden_states=True, return_dict=True)
            # ChatGLM3 hidden_states shape: [seq_len, batch, hidden] (not [batch, seq_len, hidden])
            last_layer_hs = outputs.hidden_states[-1]  # [seq_len, batch, hidden]
            if last_layer_hs.shape[0] != fusion_embedding.shape[0]:
                # Transpose to [batch, seq_len, hidden] then take last position
                last_hidden = last_layer_hs.permute(1, 0, 2)[:, -1, :]  # [batch, hidden]
            else:
                last_hidden = last_layer_hs[:, -1, :]  # already [batch, seq_len, hidden]
            return last_hidden

        # qwen / qwen3.5 / llama2 / deepseek / gemma
        opt_tokens, atts_bos, atts_fusion, _, _, opt_input_ids = self.input_processing(
            fusion_embedding, mode='generate', input_attn_mask=input_attn_mask)
        if atts_bos is not None:
            attention_mask = torch.cat([atts_bos, atts_fusion], dim=1)
        else:
            attention_mask = atts_fusion

        if self.model_type == 'gemma' and self._gemma_ple_dim > 0:
            inner_text_model = getattr(self.model.model, 'language_model', self.model.model)
            if opt_input_ids is not None and hasattr(inner_text_model, 'get_per_layer_inputs'):
                per_layer_inputs = inner_text_model.get_per_layer_inputs(
                    input_ids=opt_input_ids.to(self.device), inputs_embeds=None)
            else:
                per_layer_inputs = torch.zeros(
                    opt_tokens.shape[0], opt_tokens.shape[1],
                    self._gemma_num_layers, self._gemma_ple_dim,
                    dtype=opt_tokens.dtype, device=opt_tokens.device)
            with torch.amp.autocast(device_type='cuda'):
                base_outputs = inner_text_model(
                    inputs_embeds=opt_tokens, attention_mask=attention_mask,
                    per_layer_inputs=per_layer_inputs,
                    output_hidden_states=True, return_dict=True)
            last_hidden = base_outputs.hidden_states[-1][:, -1, :]
            return last_hidden

        model_to_call = self.model
        if self._lora_enabled:
            # LoRA-wrapped model still supports output_hidden_states
            pass

        with torch.amp.autocast(device_type='cuda'):
            outputs = model_to_call(
                inputs_embeds=opt_tokens, attention_mask=attention_mask,
                output_hidden_states=True, return_dict=True)
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        return last_hidden

    def forward(self, fusion_embedding, labels, input_attn_mask=None, context_text=None):
        fusion_embedding = self.multimodal_prompt_wrap(fusion_embedding, context_text=context_text)
        if self.model_type == 'chatglm3':
            return self._forward_chatglm3(fusion_embedding, labels)
        elif self.model_type in ['qwen', 'qwen3.5', 'llama2', 'deepseek', 'gemma']:
            return self._forward_modelscope(fusion_embedding, labels, input_attn_mask=input_attn_mask)
        raise ValueError(f"Unsupported model type in forward: {self.model_type}")

    def _forward_chatglm3(self, fusion_embedding, labels):
        opt_tokens, labels = self.input_processing(fusion_embedding, labels, mode='train')
        with torch.amp.autocast(device_type='cuda'):
            return self.model(input_ids=opt_tokens, input_fusion=fusion_embedding, labels=labels)

    def _forward_modelscope(self, fusion_embedding, labels, input_attn_mask=None):
        opt_tokens, atts_bos, atts_fusion, labels, labels_atts, opt_input_ids = self.input_processing(
            fusion_embedding, labels, mode='train', input_attn_mask=input_attn_mask
        )

        if atts_bos is not None:
            attention_mask = torch.cat([atts_bos, atts_fusion, labels_atts], dim=1)
        else:
            attention_mask = torch.cat([atts_fusion, labels_atts], dim=1)

        if getattr(self.model.config, 'pad_token_id', None) is None and hasattr(self.tokenizer, 'pad_token_id'):
            try:
                self.model.config.pad_token_id = self.tokenizer.pad_token_id
            except AttributeError:
                pass

        if self.model_type == 'gemma' and self._gemma_ple_dim > 0:
            inner_text_model = getattr(self.model.model, 'language_model', self.model.model)
            if opt_input_ids is not None and hasattr(inner_text_model, 'get_per_layer_inputs'):
                per_layer_inputs = inner_text_model.get_per_layer_inputs(input_ids=opt_input_ids.to(self.device), inputs_embeds=None)
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
            return CausalLMOutputWithPast(loss=loss, logits=logits)

        model_kwargs = dict(inputs_embeds=opt_tokens, attention_mask=attention_mask, return_dict=True, labels=labels)
        with torch.amp.autocast(device_type='cuda'):
            return self.model(**model_kwargs)

    def generate(self, fusion_embedding, input_attn_mask=None, context_text=None):
        fusion_embedding = self.multimodal_prompt_wrap(fusion_embedding, context_text=context_text)
        if self.model_type == 'chatglm3':
            return self._generate_chatglm3(fusion_embedding)
        elif self.model_type in ['qwen', 'qwen3.5', 'llama2', 'deepseek', 'gemma']:
            return self._generate_modelscope(fusion_embedding, input_attn_mask=input_attn_mask)
        raise ValueError(f"Unsupported model type in generate: {self.model_type}")

    def _generate_chatglm3(self, fusion_embedding):
        effective_max_tokens = self.max_new_tokens + 1
        gen_kwargs = {"max_new_tokens": effective_max_tokens, "num_beams": 1, "do_sample": False, "top_k": 10}
        opt_tokens, _ = self.input_processing(fusion_embedding, mode='generate')
        context_length = opt_tokens.size(1)
        all_responses = []
        for outputs in self.model.stream_generate(opt_tokens, **gen_kwargs, input_fusion=fusion_embedding):
            outputs = outputs[:, context_length:].tolist()
            response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for x in response:
            x = x.strip()
            if self.train_mode == 'regression':
                try:
                    value = float(x.replace('–', '-').replace('一', '-').replace('：', '').replace('/', '').replace('(', '').replace(':', ''))
                except ValueError:
                    value = 0.0
            else:
                value = self._parse_classification_output(x)
            all_responses.append(value)
        return all_responses

    def _generate_modelscope(self, fusion_embedding, input_attn_mask=None):
        opt_tokens, atts_bos, atts_fusion, _, _, opt_input_ids = self.input_processing(
            fusion_embedding, mode='generate', input_attn_mask=input_attn_mask
        )
        if self.model_type in ['qwen', 'qwen3.5']:
            attention_mask = torch.cat([atts_bos, atts_fusion], dim=1)
            gen_kwargs = {"num_beams": 1, "do_sample": False, "bos_token_id": self.tokenizer.bos_token_id, "eos_token_id": self.tokenizer.eos_token_id, "max_new_tokens": self.max_new_tokens,
                          "min_new_tokens": self.max_new_tokens}  # force full generation, prevent early EOS (critical for text label mode)
        elif self.model_type == 'llama2':
            attention_mask = atts_fusion if atts_bos is None else torch.cat([atts_bos, atts_fusion], dim=1)
            # Llama2 SentencePiece tokenizer prepends "▁" (space, ID=29871) before content tokens,
            # so we need at least 2 tokens to capture: [▁] + [digit/word]
            effective_max = max(self.max_new_tokens, 2)
            gen_kwargs = {"num_beams": 1, "do_sample": False, "max_new_tokens": effective_max,
                          "min_new_tokens": effective_max}  # force full generation, prevent early EOS
        else:
            attention_mask = atts_fusion
            gen_kwargs = {"num_beams": 1, "do_sample": False, "max_new_tokens": self.max_new_tokens}

        pad_id = getattr(self.model.config, 'pad_token_id', None) or getattr(self.tokenizer, 'pad_token_id', None)

        if self.model_type == 'gemma' and self._gemma_ple_dim > 0:
            with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
                current_embeds = opt_tokens
                current_input_ids = opt_input_ids
                generated_ids = []
                inner_text_model = getattr(self.model.model, 'language_model', self.model.model)
                for _ in range(self.max_new_tokens):
                    batch_size = current_embeds.shape[0]
                    if current_input_ids is not None and hasattr(inner_text_model, 'get_per_layer_inputs'):
                        per_layer_inputs = inner_text_model.get_per_layer_inputs(input_ids=current_input_ids.to(self.device), inputs_embeds=None)
                    else:
                        per_layer_inputs = torch.zeros(
                            current_embeds.shape[0], current_embeds.shape[1], self._gemma_num_layers, self._gemma_ple_dim,
                            dtype=current_embeds.dtype, device=current_embeds.device
                        )
                    base_out = inner_text_model(inputs_embeds=current_embeds, attention_mask=attention_mask, per_layer_inputs=per_layer_inputs, return_dict=True)
                    last_hidden = base_out.last_hidden_state[:, -1:, :]
                    next_logits = self.model.lm_head(last_hidden)
                    next_token_id = next_logits.argmax(dim=-1)
                    generated_ids.append(next_token_id)
                    next_embed = self.text_embedding(next_token_id)
                    current_embeds = torch.cat([current_embeds, next_embed], dim=1)
                    if current_input_ids is not None:
                        current_input_ids = torch.cat([current_input_ids, next_token_id], dim=1)
                    attention_mask = torch.cat([attention_mask, torch.ones(batch_size, 1, dtype=attention_mask.dtype, device=attention_mask.device)], dim=1)
                outputs = torch.cat(generated_ids, dim=1)
        else:
            # Temporarily restore use_cache and disable gradient checkpointing for generation
            # (MSE-Adapter doesn't use gradient_checkpointing at all; it can interfere with generate())
            _was_gc = getattr(self.model, 'is_gradient_checkpointing', False)
            if _was_gc and hasattr(self.model, 'gradient_checkpointing_disable'):
                self.model.gradient_checkpointing_disable()
            _old_use_cache = getattr(self.model.config, 'use_cache', True)
            self.model.config.use_cache = True

            if self.model_type == 'llama2':
                # Match MSE-Adapter: no attention_mask, no pad_token_id
                outputs = self.model.generate(inputs_embeds=opt_tokens,
                                              num_beams=1, do_sample=False,
                                              max_new_tokens=effective_max,
                                              min_new_tokens=effective_max)
            else:
                outputs = self.model.generate(inputs_embeds=opt_tokens, attention_mask=attention_mask, pad_token_id=pad_id, **gen_kwargs)

            # Restore original settings
            self.model.config.use_cache = _old_use_cache
            if _was_gc and hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        # ── Diagnostic logging for generate output (helps debug text label failures) ──
        if not hasattr(self, '_gen_debug_logged'):
            self._gen_debug_logged = True
            logger.info(f"[GenDebug] model_type={self.model_type}, max_new_tokens={self.max_new_tokens}")
            logger.info(f"[GenDebug] inputs_embeds.shape={opt_tokens.shape}, outputs.shape={outputs.shape}")
            logger.info(f"[GenDebug] outputs[0] token IDs: {outputs[0].tolist()}")
            logger.info(f"[GenDebug] outputs[0] full decode: '{self.tokenizer.decode(outputs[0], skip_special_tokens=True)}'")

        # Decode all generated tokens (not sliced by max_new_tokens, since Llama2
        # SentencePiece may prepend extra ▁ space tokens beyond the configured count)
        responses = self.tokenizer.batch_decode(outputs, add_special_tokens=False, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        # Log first batch parsed responses (once)
        if not hasattr(self, '_resp_debug_logged'):
            self._resp_debug_logged = True
            logger.info(f"[GenDebug] decoded responses[:8]: {responses[:8]}")

        all_responses = []
        for response in responses:
            if self.train_mode == 'regression':
                try:
                    value = float(response.replace('–', '-').replace('一', '-').replace('：', '').replace('/', '').replace('(', '').replace(':', ''))
                except ValueError:
                    value = 0.0
            else:
                value = self._parse_classification_output(response)
            all_responses.append(value)
        return all_responses

    def input_processing(self, fusion_embedding, labels=None, mode=None, input_attn_mask=None):
        if self.model_type == 'chatglm3':
            return self._input_processing_chatglm3(fusion_embedding, labels, mode)
        elif self.model_type in ['qwen', 'qwen3.5', 'llama2', 'deepseek', 'gemma']:
            return self._input_processing_modelscope(fusion_embedding, labels, mode, input_attn_mask=input_attn_mask)
        raise ValueError(f"Unsupported model type in input_processing: {self.model_type}")

    def _input_processing_chatglm3(self, fusion_embedding, labels=None, mode=None):
        input_lengths = fusion_embedding[:, :, 0]
        fusion_empty = torch.ones(input_lengths.size(), dtype=torch.long).to(self.device).fill_(0)
        task_prompt = self.get_task_prompt()
        prompt_broadcasted = task_prompt.expand(fusion_empty.size(0), -1)
        opt_tokens = torch.cat([fusion_empty, prompt_broadcasted], dim=1)
        opt_tokens, labels = self.input_labels_construct(opt_tokens, labels, mode)
        return opt_tokens, labels

    def _input_processing_modelscope(self, fusion_embedding, labels=None, mode=None, input_attn_mask=None):
        batch_size = fusion_embedding.shape[0]
        task_prompt = self.get_task_prompt()
        prompt_ids = task_prompt.expand(batch_size, -1)
        task_prompt_embedding = self.text_embedding(prompt_ids)
        opt_tokens = torch.cat([fusion_embedding, task_prompt_embedding], dim=1)
        atts_fusion = torch.ones(opt_tokens.size()[:-1], dtype=torch.long).to(self.device)

        if input_attn_mask is not None:
            prefix_len = getattr(self, '_wrap_prefix_len', 0)
            mask_len = input_attn_mask.shape[1]
            atts_fusion[:, prefix_len:prefix_len + mask_len] *= input_attn_mask.to(self.device)

        pad_id = getattr(self.tokenizer, 'pad_token_id', 0)
        fusion_ids = torch.full(fusion_embedding.shape[:-1], pad_id, dtype=torch.long, device=self.device)
        opt_input_ids = torch.cat([fusion_ids, prompt_ids], dim=1)

        if self.model_type in ['qwen', 'qwen3.5', 'llama2']:
            bos_ids = torch.ones([batch_size, 1], dtype=atts_fusion.dtype, device=self.device) * self.tokenizer.bos_token_id
            bos_embeds = self.text_embedding(bos_ids)
            atts_bos = atts_fusion[:, :1]
            opt_tokens = torch.cat([bos_embeds, opt_tokens], dim=1)
            opt_input_ids = torch.cat([bos_ids, opt_input_ids], dim=1)
        else:
            atts_bos = None

        opt_tokens, labels, labels_atts, opt_input_ids = self.input_labels_construct(opt_tokens, labels, mode, opt_input_ids)

        if self.model_type in ['qwen', 'qwen3.5', 'llama2']:
            return opt_tokens, atts_bos, atts_fusion, labels, labels_atts, opt_input_ids
        return opt_tokens, None, atts_fusion, labels, labels_atts, opt_input_ids

    def input_labels_construct(self, opt_tokens, labels=None, mode=None, opt_input_ids=None):
        batch_size = opt_tokens.shape[0]
        if mode == "train":
            if self.train_mode == "regression":
                if self.model_type in ['qwen', 'qwen3.5']:
                    label_template = [f"+{label.item():.{1}f}" if label >= 0 else f"{label.item():.{1}f}" for label in labels]
                else:
                    label_template = [f"{label.item():.{1}f}" for label in labels]
            else:
                eos_suffix = ''
                if self.model_type in ['qwen', 'qwen3.5'] and hasattr(self.tokenizer, 'eos_token') and self.tokenizer.eos_token:
                    eos_suffix = self.tokenizer.eos_token
                if self.label_format == 'text' and self._idx_to_name:
                    label_template = [f"{self._idx_to_name.get(int(label.item()), str(int(label.item())))}{eos_suffix}" for label in labels]
                else:
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

    def _parse_classification_output(self, response):
        """Parse classification output for both index and text label formats."""
        response = response.strip()
        # First try direct numeric parse (works for index mode and numeric outputs)
        try:
            return float(response)
        except ValueError:
            pass
        # Text label mode: fuzzy match against known label names
        response_lower = response.lower()
        for name, idx in self.label_index_mapping.items():
            if response_lower.startswith(name.lower()):
                return float(idx)
        # Fallback: return 0 (first class)
        return 0.0

    def get_task_prompt(self):
        import re
        prompt_text = self.task_specific_prompt
        # For text label format, strip ":N" from prompt labels
        # e.g. "<neutral:0, surprise:1, ...>" → "<neutral, surprise, ...>"
        if self.label_format == 'text' and self.train_mode == 'classification':
            prompt_text = re.sub(r':(\d+)', '', prompt_text)
        return self.tokenizer(prompt_text, padding=True, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.device)

    def multimodal_prompt_wrap(self, fusion_embeddings, context_text=None):
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
            if self.prompt_style == 'instructerc':
                # InstructERC-style prompt: role preamble + multimodal wrap
                # Matches the "Now you are expert..." pattern from SpeechCueLLM
                if self.language == "en":
                    prompt = ('Now you are an expert of sentiment and emotional analysis. '
                              'Analyze the following multimodal content: '
                              '<Multimodal><MultimodalHere></Multimodal>')
                else:
                    prompt = ('你现在是情感分析专家。'
                              '请分析以下多模态内容：'
                              '<多模态><MultimodalHere></多模态>')
            elif self.prompt_style == 'enhanced':
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

        self._wrap_prefix_len = p_before_embeds.shape[1]
        self._wrap_suffix_len = p_after_embeds.shape[1]

        wrapped = torch.cat([p_before_embeds, fusion_embeddings, p_after_embeds], dim=1)

        # ── Prompt-level context injection (UniSA-inspired) ──
        # Inject dialogue context BEFORE the multimodal wrap so that
        # the LLM sees: [context] → [<Multimodal>...tokens...</Multimodal>] → [task prompt]
        if context_text is not None and getattr(self._args, 'prompt_context', False):
            # Filter: only inject if at least one sample has non-empty context
            has_context = any(c.strip() for c in context_text if isinstance(c, str))
            if has_context:
                ctx_max = getattr(self._args, 'context_max_tokens', 64)
                # Build context prompt strings per sample
                ctx_strings = []
                for c in context_text:
                    c = str(c).strip() if c else ''
                    if c:
                        ctx_strings.append(f" Dialogue context: {c}")
                    else:
                        ctx_strings.append("")  # empty placeholder

                # Tokenize all context strings with padding
                ctx_tokenized = self.tokenizer(
                    ctx_strings,
                    padding='max_length',
                    truncation=True,
                    max_length=ctx_max,
                    return_tensors='pt',
                    add_special_tokens=False
                ).to(self.device)

                ctx_embeds = self.text_embedding(ctx_tokenized['input_ids'])  # [B, ctx_max, hidden]
                # Zero out padding positions so they don't contribute
                ctx_mask = ctx_tokenized['attention_mask'].unsqueeze(-1).to(ctx_embeds.dtype)  # [B, ctx_max, 1]
                ctx_embeds = ctx_embeds * ctx_mask

                # Prepend context BEFORE wrapped multimodal tokens
                wrapped = torch.cat([ctx_embeds, wrapped], dim=1)
                self._wrap_prefix_len += ctx_max  # update for attention mask accounting

        return wrapped
