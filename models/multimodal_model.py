import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
import math

from models.text_encoder import FakeNewsTextEncoder
from models.image_encoder import FakeNewsImageEncoder
from models.classifier import FakeNewsClassifier


class FakeNewsMultimodalModel(nn.Module):

    def __init__(
        self,
        model_name="ViT-B-32",
        pretrained="openai",
        freeze_text_encoder=True,
        freeze_image_encoder=True
    ):
        super().__init__()

        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained
        )
        
        # Enable Gradient Checkpointing: Trades compute for massive VRAM savings
        set_gc = getattr(self.clip_model, "set_grad_checkpointing", None)
        if callable(set_gc):
            set_gc(True)


        self.text_encoder = FakeNewsTextEncoder(
            model_name=model_name,
            pretrained=pretrained,
            freeze_text_encoder=freeze_text_encoder,
            clip_model=self.clip_model
        )

        self.image_encoder = FakeNewsImageEncoder(
            model_name=model_name,
            pretrained=pretrained,
            freeze_image_encoder=freeze_image_encoder,
            clip_model=self.clip_model
        )

        # NLI Fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(512 * 4, 512),
            nn.GELU(),   
            nn.Dropout(0.4) # Increased regularization
        )

        self.classifier = FakeNewsClassifier(
            input_dim=512 * 3 + 1,
            num_classes=2,
            dropout=0.5 # Increased regularization
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def forward(self, text_tokens, images):

        text_features = self.text_encoder(text_tokens)
        image_features = self.image_encoder(images)

        text_features = F.normalize(text_features, dim=1)
        image_features = F.normalize(image_features, dim=1)

        interaction = text_features * image_features
        difference = torch.abs(text_features - image_features)  

        fusion = self.fusion_layer(
            torch.cat([text_features, image_features, interaction, difference], dim=1)
        )

        similarity = F.cosine_similarity(text_features, image_features).unsqueeze(1)

        combined = torch.cat(
            [text_features, image_features, fusion, similarity],
            dim=1
        )

        logits = self.classifier(combined)
        temperature = torch.clamp(self.logit_scale.exp(), min=1e-3, max=100)

        return logits, text_features, image_features, temperature