import torch
from transformers import CLIPImageProcessor, CLIPVisionModel
from torch import nn


class CLIPVisionEncoder(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path: str = "openai/clip-vit-large-patch14-336",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.model = CLIPVisionModel.from_pretrained(pretrained_model_name_or_path)
        self.processor = CLIPImageProcessor.from_pretrained(pretrained_model_name_or_path)
        self.select_layer = -2
        self.hidden_size = self.model.config.hidden_size
        for p in self.model.parameters():
            p.requires_grad = False
        print('Freezing CLIP vision encoder.')
        self.dtype = dtype
        self.model.to(dtype)

        # 구버전(vision_model 속성) / 신버전(flat 구조) 모두 지원
        self._vm  = getattr(self.model, 'vision_model', self.model)
        self._emb = self._vm.embeddings

    def _manual_patch_embed(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        patch_embedding Conv2d(sm_120 미지원) → unfold + matmul 대체.
        pixel_values: (B, 3, H, W)  →  returns: (B, N_patches+1, hidden_size)
        """
        B = pixel_values.shape[0]
        patch_size = self.model.config.patch_size   # 14
        emb = self._emb

        # ── 1) unfold → 패치 추출 ──────────────────────────────────────────
        p = pixel_values.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        # p: (B, C, h_p, w_p, ph, pw)
        n_h, n_w = p.shape[2], p.shape[3]
        p = p.permute(0, 2, 3, 1, 4, 5).contiguous().view(B, n_h * n_w, -1)
        # p: (B, n_patches, C*ph*pw) = (B, 576, 588)

        # ── 2) 선형 투영 (Conv2d 대체) ────────────────────────────────────
        w = emb.patch_embedding.weight.view(emb.patch_embedding.weight.shape[0], -1)
        patch_embeds = p @ w.T   # (B, 576, 1024)
        if emb.patch_embedding.bias is not None:
            patch_embeds = patch_embeds + emb.patch_embedding.bias

        # ── 3) CLS 토큰 prepend ───────────────────────────────────────────
        cls = emb.class_embedding.to(pixel_values.dtype)
        cls = cls.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)   # (B, 1, 1024)
        embeddings = torch.cat([cls, patch_embeds], dim=1)      # (B, 577, 1024)

        # ── 4) 위치 임베딩 ────────────────────────────────────────────────
        position_ids = torch.arange(
            embeddings.shape[1], device=pixel_values.device
        ).unsqueeze(0)
        pos_emb = emb.position_embedding(position_ids).to(pixel_values.dtype)
        embeddings = embeddings + pos_emb    # (B, 577, 1024)

        return embeddings

    def forward(self, inputs_embeds: torch.Tensor):
        pixel_values = inputs_embeds.to(self.dtype)
        outputs = self.model(pixel_values, output_hidden_states=True)
        h = outputs.hidden_states
        return h[6][:, :-1, :], h[10][:, :-1, :], h[14][:, :-1, :], h[self.select_layer]
