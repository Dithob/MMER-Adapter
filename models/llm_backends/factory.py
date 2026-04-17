from models.llm_backends.chatglm3.backend import ChatGLM3Backend
from models.llm_backends.gemma.backend import GemmaBackend
from models.llm_backends.modelscope.backend import ModelScopeBackend


def build_llm_backend(args):
    if args.model_type == 'chatglm3':
        return ChatGLM3Backend(args)
    if args.model_type == 'gemma':
        return GemmaBackend(args)
    if args.model_type in ['qwen', 'qwen3.5', 'llama2', 'deepseek']:
        return ModelScopeBackend(args)
    raise ValueError(f'Unsupported model type: {args.model_type}')
