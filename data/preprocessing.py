import torch
from torch.utils.data import Dataset
from PIL import Image
import os


class FakedditMultimodalDataset(Dataset):

    def __init__(self, dataframe, tokenizer, image_preprocess, image_dir):

        self.tokenizer        = tokenizer
        self.image_preprocess = image_preprocess
        self.image_dir        = image_dir

        original_count = len(dataframe)

        # Filter to only rows where the image file actually exists on disk
        valid_rows = []
        for _, row in dataframe.iterrows():
            img_path = os.path.join(image_dir, f"{row['id']}.jpg")
            if os.path.exists(img_path):
                valid_rows.append(row)

        # FIX: The old code said "Original samples" but was printing the
        # FILTERED count (rows with an image on disk). Now both are shown.
        filtered_count = len(valid_rows)
        print(
            f"Dataset init: {original_count} rows requested → "
            f"{filtered_count} have images on disk "
            f"({original_count - filtered_count} skipped — image not downloaded yet)"
        )

        if filtered_count == 0:
            # Don't crash — let the caller decide what to do.
            # train_model.py will detect this and use a fallback.
            self.texts  = []
            self.labels = []
            self.ids    = []
            return

        dataframe = dataframe.loc[[row.name for row in valid_rows]].reset_index(drop=True)

        self.texts  = dataframe["text"].tolist()
        self.labels = dataframe["label"].tolist()
        self.ids    = dataframe["id"].tolist()

    def __len__(self):
        return len(self.texts)

    def load_image(self, img_id):
        path  = os.path.join(self.image_dir, f"{img_id}.jpg")
        image = Image.open(path).convert("RGB")
        return image

    def __getitem__(self, idx):
        text   = str(self.texts[idx])
        label  = int(self.labels[idx])
        img_id = self.ids[idx]

        text_tokens  = self.tokenizer([text])[0]
        image        = self.load_image(img_id)
        image_tensor = self.image_preprocess(image)

        return {
            "text_tokens": text_tokens,
            "image":       image_tensor,
            "label":       torch.tensor(label, dtype=torch.long)
        }
