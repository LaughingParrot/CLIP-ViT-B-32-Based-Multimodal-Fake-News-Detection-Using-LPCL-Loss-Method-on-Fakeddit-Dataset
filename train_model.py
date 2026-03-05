from data.dataset_loader import FakedditDatasetLoader
from data.preprocessing import FakedditTextDataset
from models.text_encoder import FakeNewsTextEncoder
from models.classifier import FakeNewsClassifier
from training.trainer import Trainer

import open_clip


dataset_path = "Fakeddit"

loader = FakedditDatasetLoader(dataset_path)

train_df, val_df, test_df = loader.load_datasets()

# The below 2 lines are for sample testing as training model may take hours
train_df = train_df.sample(20000)
val_df = val_df.sample(5000)

tokenizer = open_clip.get_tokenizer("ViT-B-32")

train_dataset = FakedditTextDataset(train_df, tokenizer)

val_dataset = FakedditTextDataset(val_df, tokenizer)

encoder = FakeNewsTextEncoder()

classifier = FakeNewsClassifier()

trainer = Trainer(
    encoder,
    classifier,
    train_dataset,
    val_dataset
)

trainer.train(epochs=3)