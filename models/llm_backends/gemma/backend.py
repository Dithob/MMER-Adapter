import torch

from ..base import BaseLLMBackend


class GemmaBackend(BaseLLMBackend):
    def load(self, pretrained_model):
        from transformers import AutoTokenizer, AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model,
            padding_side='left',
            trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model,
            trust_remote_code=True,
            dtype=torch.float16,
        )

        base_cfg = model.config
        text_cfg = getattr(base_cfg, 'text_config', base_cfg)
        self._gemma_ple_dim = getattr(text_cfg, 'hidden_size_per_layer_input', 0)
        self._gemma_num_layers = getattr(text_cfg, 'num_hidden_layers', 0)

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        return model, tokenizer
