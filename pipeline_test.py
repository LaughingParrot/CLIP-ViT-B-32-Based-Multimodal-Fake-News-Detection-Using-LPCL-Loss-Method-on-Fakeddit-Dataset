# from data.dataset_loader import FakedditDatasetLoader
# from data.preprocessing import FakedditTextDataset
# import open_clip
# from models.text_encoder import FakeNewsTextEncoder
# dataset_path = "Fakeddit"

# loader = FakedditDatasetLoader(dataset_path)

# train_df, val_df, test_df = loader.load_datasets()

# tokenizer = open_clip.get_tokenizer("ViT-B-32")

# train_dataset = FakedditTextDataset(train_df, tokenizer)

# sample = train_dataset[0]

# model = FakeNewsTextEncoder()

# text_tokens = sample["text_tokens"].unsqueeze(0)

# output = model(text_tokens)

# print("Model output:", output)
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))