import open_clip
import torch

from data.dataset_loader import FakedditDatasetLoader
from data.preprocessing import FakedditMultimodalDataset
from models.multimodal_model import FakeNewsMultimodalModel
from training.trainer import Trainer

if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dataset_path = "Fakeddit"
    image_dir = "Fakeddit/images"
    model_name = "ViT-B-32"

    loader = FakedditDatasetLoader(dataset_path)
    train_df, val_df, test_df = loader.load_datasets()

    print(f"\nValid rows available — Train: {len(train_df)}  Val: {len(val_df)}\n")

    n_train = min(100000, len(train_df))
    n_val   = min(30000,  len(val_df))

    train_df = train_df.sample(n_train, random_state=42)
    val_df = val_df.sample(n_val, random_state=42)

    tokenizer = open_clip.get_tokenizer(model_name)
    _, _, image_preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained="openai"
    )

    train_dataset = FakedditMultimodalDataset(train_df, tokenizer, image_preprocess, image_dir)
    val_dataset = FakedditMultimodalDataset(val_df, tokenizer, image_preprocess, image_dir)
    
    if len(val_dataset) == 0:
        print("[WARNING] Val dataset empty. Falling back to a slice of train data.")
        fallback_val_df = train_df.sample(min(2000, len(train_df)), random_state=99)
        val_dataset = FakedditMultimodalDataset(fallback_val_df, tokenizer, image_preprocess, image_dir)
        
    print(f"\nFinal — Train: {len(train_dataset)} samples  |  Val: {len(val_dataset)} samples\n")
    
    model = FakeNewsMultimodalModel(
        freeze_text_encoder=False,
        freeze_image_encoder=False
    )

    trainer = Trainer(
        model,
        train_dataset,
        val_dataset,
        batch_size=6, # Reduced to lower VRAM (Trainer accumulates gradients)
        lr=1e-5       
    )

    trainer.train(epochs=3)