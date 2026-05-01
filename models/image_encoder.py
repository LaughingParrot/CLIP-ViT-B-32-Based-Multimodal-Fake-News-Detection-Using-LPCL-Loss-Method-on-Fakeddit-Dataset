import torch
import torch.nn as nn
import open_clip


class FakeNewsImageEncoder(nn.Module):

    def __init__(
        self,
        model_name="ViT-B-32",
        pretrained="openai",
        freeze_image_encoder=False,
        clip_model = None
    ):
        super().__init__()

        if clip_model is None:
            self.clip_model, _, _ = open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained
            )
        else:
            self.clip_model = clip_model

        self.freeze_image_encoder = freeze_image_encoder

        if self.freeze_image_encoder:
            for param in self.clip_model.visual.parameters():  # type: ignore
                param.requires_grad = False

    def forward(self, images):

        if images.ndim == 3:
            images = images.unsqueeze(0)

        clip_device = next(self.clip_model.parameters()).device
        images = images.to(device=clip_device)

        if self.freeze_image_encoder:
            with torch.no_grad():
                image_features = self.clip_model.encode_image(images)  # type: ignore
        else:
            image_features = self.clip_model.encode_image(images)  # type: ignore

        return image_features