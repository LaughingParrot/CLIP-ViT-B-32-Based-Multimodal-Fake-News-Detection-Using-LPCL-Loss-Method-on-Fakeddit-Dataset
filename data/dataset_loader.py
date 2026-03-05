import pandas as pd
import os


class FakedditDatasetLoader:
    def __init__(self, dataset_dir):

        self.dataset_dir = dataset_dir      #dataset_dir: path to Fakeddit folder

        self.train_file = os.path.join(dataset_dir, "multimodal_train.tsv")
        self.val_file = os.path.join(dataset_dir, "multimodal_validate.tsv")
        self.test_file = os.path.join(dataset_dir, "multimodal_test_public.tsv")

    def load_split(self, filepath):

        df = pd.read_csv(
            filepath,
            sep="\t",
            usecols=[
                "id",
                "clean_title",
                "2_way_label",
                "domain",
                "hasImage",
                "image_url"
            ]
        )

        df = df.rename(columns={
            "clean_title": "text",
            "2_way_label": "label"
        })

        df = df.dropna(subset=["text"])

        return df

    def load_datasets(self):

        """Load train, validation and test datasets"""

        train_df = self.load_split(self.train_file)
        val_df = self.load_split(self.val_file)
        test_df = self.load_split(self.test_file)

        return train_df, val_df, test_df