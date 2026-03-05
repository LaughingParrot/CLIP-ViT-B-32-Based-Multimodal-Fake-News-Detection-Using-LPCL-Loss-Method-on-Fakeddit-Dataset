import torch
import torch.nn as nn
import open_clip


class FakeNewsTextEncoder(nn.Module):

    def __init__(
        self,
        model_name="ViT-B-32",
        pretrained="openai",
        freeze_text_encoder=False,
    ):
        super().__init__()

        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained
        )

        self.freeze_text_encoder = freeze_text_encoder

        if self.freeze_text_encoder:
            for param in self.clip_model.parameters():
                param.requires_grad = False

    def forward(self, text_tokens):

        if text_tokens.ndim == 1:
            text_tokens = text_tokens.unsqueeze(0)

        clip_device = next(self.clip_model.parameters()).device

        text_tokens = text_tokens.to(device=clip_device, dtype=torch.long)

        encode_text = getattr(self.clip_model, "encode_text")

        if self.freeze_text_encoder:
            with torch.no_grad():
                text_features = encode_text(text_tokens)
        else:
            text_features = encode_text(text_tokens)

        return text_features