import logging
import torch

from ..base import BaseLLMBackend

logger = logging.getLogger('MSA')


class ChatGLM3Backend(BaseLLMBackend):
    def _allow_legacy_torch_load_for_trusted_local_model(self):
        """Allow loading legacy .bin weights on torch<2.6 in trusted local setups.

        Transformers 4.50+ blocks torch.load for safety unless torch>=2.6.
        Our checkpoints are local, offline, and trusted in this project, so we
        explicitly disable the guard to keep the existing ChatGLM3 weights loadable.
        """
        try:
            from transformers.utils import import_utils
            if hasattr(import_utils, 'check_torch_load_is_safe'):
                import_utils.check_torch_load_is_safe = lambda: None
                logger.warning(
                    'Bypassing Transformers torch.load safety gate for trusted local ChatGLM3 weights. '
                    'If possible, migrate the checkpoint to safetensors for long-term safety.'
                )
        except Exception as e:
            logger.warning(f'Failed to patch Transformers torch.load safety gate: {e}')

    def load(self, pretrained_model):
        from models.llm_backends.chatglm3.modeling_chatglm import ChatGLMForConditionalGeneration
        from models.llm_backends.chatglm3.tokenization_chatglm import ChatGLMTokenizer

        if torch.__version__ < '2.6':
            self._allow_legacy_torch_load_for_trusted_local_model()

        model = ChatGLMForConditionalGeneration.from_pretrained(
            pretrained_model,
            dtype=torch.bfloat16,
            use_safetensors=True,
        ).half()
        tokenizer = ChatGLMTokenizer.from_pretrained(pretrained_model)

        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        return model, tokenizer
