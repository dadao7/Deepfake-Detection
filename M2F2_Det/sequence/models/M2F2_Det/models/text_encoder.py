import torch
from transformers import AutoTokenizer, CLIPTextModel
from torch import nn
from typing import Optional


class CLIPTextEncoder(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path: str = "openai/clip-vit-large-patch14-336",
        prompt_token_num: int = 7,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.model = CLIPTextModel.from_pretrained(pretrained_model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.hidden_size = self.model.config.hidden_size
        self.prompt_tokens = nn.Parameter(
            torch.randn(1, prompt_token_num, self.model.config.hidden_size)
        )
        self.dtype = dtype
        self.model.to(dtype)
        self.prompt_tokens.to(dtype)

        # 구버전/신버전 transformers API 모두 지원
        # 구버전: CLIPTextModel.text_model → CLIPTextTransformer
        # 신버전: CLIPTextModel 자체가 transformer 역할
        self._tm = getattr(self.model, 'text_model', self.model)

    def forward(
        self,
        input_embeds: Optional[torch.Tensor] = None
    ):
        if input_embeds is None:
            input_embeds = self.prompt_tokens
        batch_size, seq_length = input_embeds.shape[:2]
        input_embeds = input_embeds.to(self.dtype)

        position_ids = torch.arange(
            seq_length, device=input_embeds.device
        ).expand(input_embeds.shape[0], -1)

        position_embeddings = self._tm.embeddings.position_embedding(
            position_ids
        ).to(input_embeds.dtype)

        input_embeds = input_embeds + position_embeddings

        attention_mask = torch.zeros(
            batch_size, 1, seq_length, seq_length,
            dtype=input_embeds.dtype, device=input_embeds.device
        )
        casual_attention_mask = torch.triu(
            torch.full(
                [batch_size, 1, seq_length, seq_length],
                torch.finfo(input_embeds.dtype).min
            ), diagonal=1
        ).to(input_embeds.dtype).to(input_embeds.device)

        outputs = self._tm.encoder(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            causal_attention_mask=casual_attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
        )
        last_hidden_state = outputs[0]
        last_hidden_state = self._tm.final_layer_norm(last_hidden_state)
        pooled_output = last_hidden_state[:, -1, :]
        return pooled_output
