import os
import sys
import collections
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        
        if use_PLM:
            pretrained_model = args.pretrain_LM
            self._load_model(pretrained_model)
            # freeze parameter
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            print('please use PLM')
    
    def _load_model(self, pretrained_model):
        """根据模型类型加载不同的语言模型"""
        if self.model_type == 'chatglm3':
            self._load_chatglm3(pretrained_model)
        elif self.model_type in ['qwen', 'llama2']:
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
            torch_dtype=torch.bfloat16
        ).half()
        
        # 设置token id
        if self.model_type == 'qwen':
            self.eos_token_id = self.tokenizer.convert_tokens_to_ids('')
            self.tokenizer.pad_token_id = self.eos_token_id
            self.bos_token_id = self.tokenizer.convert_tokens_to_ids('<|im_start|>')
            self.tokenizer.bos_token_id = self.bos_token_id
        elif self.model_type == 'llama2':
            self.tokenizer.pad_token_id = 0
            self.eos_token_id = self.tokenizer.convert_tokens_to_ids('')
    
    def text_embedding(self, text_ids):
        embeddings = self.model.base_model.get_input_embeddings()
        return embeddings(text_ids)
    
    def forward(self, fusion_embedding, labels):
        """
        Args:
            fusion_embedding: the "concatenate" result of multimodal low rank fusion and text embedding
            label: ground_truth
        """
        fusion_embedding = self.multimodal_prompt_wrap(fusion_embedding)
        
        if self.model_type == 'chatglm3':
            return self._forward_chatglm3(fusion_embedding, labels)
        else:
            return self._forward_modelscope(fusion_embedding, labels)
    
    def _forward_chatglm3(self, fusion_embedding, labels):
        """ChatGLM3的前向传播"""
        opt_tokens, labels = self.input_processing(fusion_embedding, labels, mode='train')
        
        with torch.cuda.amp.autocast():
            output = self.model(
                input_ids=opt_tokens, 
                input_fusion=fusion_embedding, 
                labels=labels
            )
        
        return output
    
    def _forward_modelscope(self, fusion_embedding, labels):
        """ModelScope模型的前向传播"""
        opt_tokens, atts_bos, atts_fusion, labels, labels_atts = self.input_processing(
            fusion_embedding, labels, mode='train'
        )
        
        attention_mask = torch.cat([atts_bos, atts_fusion, labels_atts], dim=1)
        
        with torch.cuda.amp.autocast():
            output = self.model(
                inputs_embeds=opt_tokens, 
                return_dict=True, 
                labels=labels
            )
        
        return output
    
    def generate(self, fusion_embedding):
        """生成预测结果"""
        fusion_embedding = self.multimodal_prompt_wrap(fusion_embedding)
        
        if self.model_type == 'chatglm3':
            return self._generate_chatglm3(fusion_embedding)
        else:
            return self._generate_modelscope(fusion_embedding)
    
    def _generate_chatglm3(self, fusion_embedding):
        """ChatGLM3的生成"""
        if self.train_mode == 'regression':
            gen_kwargs = {"max_new_tokens": self.max_new_tokens, "num_beams": 1, "do_sample": False, "top_k": 10}
        else:
            gen_kwargs = {"max_new_tokens": self.max_new_tokens, "num_beams": 1, "do_sample": False, "top_k": 10}
        
        opt_tokens, _ = self.input_processing(fusion_embedding, mode='generate')
        
        context_length = opt_tokens.size(1)
        all_responses = []
        
        for outputs in self.model.stream_generate(opt_tokens, **gen_kwargs, input_fusion=fusion_embedding):
            outputs = outputs[:, context_length:].tolist()
            response = self.tokenizer.batch_decode(outputs)
        
        for x in response:
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
    
    def _generate_modelscope(self, fusion_embedding):
        """ModelScope模型的生成"""
        opt_tokens, atts_bos, atts_fusion, _, _ = self.input_processing(fusion_embedding, mode='generate')
        
        if self.model_type == 'qwen':
            attention_mask = torch.cat([atts_bos, atts_fusion], dim=1)
            gen_kwargs = {
                "num_beams": 1,
                "do_sample": False,
                "bos_token_id": self.tokenizer.bos_token_id,
                "max_new_tokens": self.max_new_tokens
            }
        else:  # llama2
            attention_mask = None
            gen_kwargs = {
                "num_beams": 1,
                "do_sample": False,
                "top_p": None,
                "max_new_tokens": self.max_new_tokens
            }
        
        outputs = self.model.generate(inputs_embeds=opt_tokens, **gen_kwargs)
        
        if self.model_type == 'qwen':
            responses = self.tokenizer.batch_decode(
                outputs[:, 1:], 
                add_special_tokens=False, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
        else:  # llama2
            responses = self.tokenizer.batch_decode(
                outputs[:, 1:], 
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
    
    def input_processing(self, fusion_embedding, labels=None, mode=None):
        """
        Args:
            fusion_embedding: the "concatenate" result of multimodal low rank fusion and text embedding
            labels: ground_truth
            mode: 'train' or 'generate'
        """
        if self.model_type == 'chatglm3':
            return self._input_processing_chatglm3(fusion_embedding, labels, mode)
        else:
            return self._input_processing_modelscope(fusion_embedding, labels, mode)
    
    def _input_processing_chatglm3(self, fusion_embedding, labels=None, mode=None):
        """ChatGLM3的输入处理"""
        input_lengths = fusion_embedding[:, :, 0]
        fusion_empty = (torch.ones(input_lengths.size(), dtype=torch.long).to(self.device).fill_(0))
        
        task_prompt = self.get_task_prompt()
        prompt_broadcasted = task_prompt.expand(fusion_empty.size(0), -1)
        
        opt_tokens = torch.cat([fusion_empty, prompt_broadcasted], dim=1)
        opt_tokens, labels = self.input_labels_construct(opt_tokens, labels, mode)
        
        return opt_tokens, labels
    
    def _input_processing_modelscope(self, fusion_embedding, labels=None, mode=None):
        """ModelScope模型的输入处理"""
        batch_size = fusion_embedding.shape[0]
        
        task_prompt = self.get_task_prompt()
        task_prompt_embedding = self.text_embedding(task_prompt.expand(batch_size, -1))
        
        opt_tokens = torch.cat([fusion_embedding, task_prompt_embedding], dim=1)
        atts_fusion = torch.ones(opt_tokens.size()[:-1], dtype=torch.long).to(self.device)
        
        if self.model_type == 'qwen':
            bos = torch.ones([batch_size, 1], dtype=atts_fusion.dtype, device=self.device) * self.tokenizer.bos_token_id
            bos_embeds = self.text_embedding(bos)
            atts_bos = atts_fusion[:, :1]
            opt_tokens = torch.cat([bos_embeds, opt_tokens], dim=1)
        else:  # llama2
            bos_embeds = None
            atts_bos = None
        
        opt_tokens, labels, labels_atts = self.input_labels_construct(opt_tokens, labels, mode)
        
        if self.model_type == 'qwen':
            return opt_tokens, atts_bos, atts_fusion, labels, labels_atts
        else:  # llama2
            return opt_tokens, None, atts_fusion, labels, labels_atts
    
    def input_labels_construct(self, opt_tokens, labels=None, mode=None):
        """
        Args:
            opt_tokens: the "concatenate" size of multimodal fusion, text embedding and prompt
            labels: ground_truth
        """
        batch_size = opt_tokens.shape[0]
        
        if mode == "train":
            if self.train_mode == "regression":
                if self.model_type == 'qwen':
                    label_template = [f"+{label.item():.{1}f}" if label >= 0 else f"{label.item():.{1}f}" for label in labels]
                else:
                    label_template = [f"{label.item():.{1}f}" for label in labels]
            else:
                label_template = [f"{label.item()}" for label in labels]
            
            if self.model_type == 'chatglm3':
                labels_id = self.tokenizer(label_template, padding=True, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.device)
                labels_matrix = torch.empty_like(opt_tokens).fill_(-100).long().to(self.device)
                opt_tokens = torch.cat([opt_tokens, labels_id], dim=1)
                labels = torch.cat([labels_matrix, labels_id], dim=1)
                return opt_tokens, labels
            else:
                labels = self.tokenizer(label_template, padding=True, return_tensors="pt", add_special_tokens=False).to(self.device)
                labels_id = labels["input_ids"]
                labels_atts = labels["attention_mask"]
                
                labels_embedding = self.text_embedding(labels_id)
                labels_matrix = torch.empty(opt_tokens.size(0), opt_tokens.size(1)).fill_(-100).long().to(self.device)
                opt_tokens = torch.cat([opt_tokens, labels_embedding], dim=1)
                labels = torch.cat([labels_matrix, labels_id], dim=1)
                
                return opt_tokens, labels, labels_atts
        else:
            if self.model_type == 'chatglm3':
                return opt_tokens, None
            else:
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
            if self.language == "en":
                prompt = '<Multimodal><MultimodalHere></Multimodal>'
            else:
                prompt = '<多模态><MultimodalHere></多模态>'
            
            p_before, p_after = prompt.split(special_token)
            p_before_tokens = self.tokenizer(p_before, return_tensors="pt", add_special_tokens=True).to(self.device)
            p_after_tokens = self.tokenizer(p_after, return_tensors="pt", add_special_tokens=False).to(self.device)
            p_before_embeds = self.text_embedding(p_before_tokens.input_ids.expand(batch_size, -1))
            p_after_embeds = self.text_embedding(p_after_tokens.input_ids.expand(batch_size, -1))
        
        wrapped_fusion_embeddings = torch.cat([p_before_embeds, fusion_embeddings, p_after_embeds], dim=1)
        return wrapped_fusion_embeddings
