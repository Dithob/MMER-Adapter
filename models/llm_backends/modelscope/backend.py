import torch

from ..base import BaseLLMBackend


class ModelScopeBackend(BaseLLMBackend):
    def load(self, pretrained_model):
        from modelscope import AutoTokenizer, AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model,
            padding_side='left',
            trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        if hasattr(model, 'generation_config') and model.generation_config is not None:
            if getattr(model.generation_config, 'max_length', None) is not None:
                model.generation_config.max_length = None
            if getattr(model.generation_config, 'do_sample', None) is not None:
                model.generation_config.do_sample = False
            if getattr(model.generation_config, 'temperature', None) is not None:
                model.generation_config.temperature = 1.0
            if getattr(model.generation_config, 'top_p', None) is not None:
                model.generation_config.top_p = 1.0

        if self.model_type in ['qwen', 'qwen3.5']:
            eos = tokenizer.convert_tokens_to_ids('<|endoftext|>')
            tokenizer.eos_token_id = eos
            tokenizer.pad_token_id = eos
            tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids('<|im_start|>')
        elif self.model_type == 'llama2':
            eos = tokenizer.convert_tokens_to_ids('</s>')
            if eos is None:
                eos = getattr(tokenizer, 'eos_token_id', None)
            if getattr(tokenizer, 'eos_token_id', None) is None and eos is not None:
                tokenizer.eos_token_id = eos
            if tokenizer.pad_token is None:
                if getattr(tokenizer, 'eos_token', None) is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                elif eos is not None:
                    tokenizer.pad_token_id = eos
                else:
                    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = getattr(tokenizer, 'eos_token_id', 0) or 0
            if getattr(model.config, 'pad_token_id', None) is None:
                try:
                    model.config.pad_token_id = tokenizer.pad_token_id
                except AttributeError:
                    pass
        elif self.model_type == 'deepseek':
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
        elif self.model_type == 'chatglm3':
            # chatglm3 通过 ModelScope 加载时的特殊处理
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id

        return model, tokenizer
