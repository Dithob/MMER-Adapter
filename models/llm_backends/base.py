import torch


class BaseLLMBackend:
    def __init__(self, args):
        self.args = args
        self.model_type = args.model_type
        self.device = args.device
        self._gemma_ple_dim = 0
        self._gemma_num_layers = 0

    def load(self, pretrained_model):
        raise NotImplementedError

    def text_embedding(self, model, tokenizer, text_ids):
        if self.model_type == 'gemma':
            embeddings = model.get_input_embeddings()
        else:
            embeddings = model.base_model.get_input_embeddings()
        return embeddings(text_ids)
