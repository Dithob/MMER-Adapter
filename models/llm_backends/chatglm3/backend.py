import torch

from ..base import BaseLLMBackend


class ChatGLM3Backend(BaseLLMBackend):
    def load(self, pretrained_model):
        from models.llm_backends.chatglm3.modeling_chatglm import ChatGLMForConditionalGeneration
        from models.llm_backends.chatglm3.tokenization_chatglm import ChatGLMTokenizer

        model = ChatGLMForConditionalGeneration.from_pretrained(
            pretrained_model,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).half()
        tokenizer = ChatGLMTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)

        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        return model, tokenizer
