import os
import pandas as pd


class FakedditDatasetLoader:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.train_file  = os.path.join(dataset_dir, "multimodal_train.tsv")
        self.val_file    = os.path.join(dataset_dir, "multimodal_validate.tsv")
        self.test_file   = os.path.join(dataset_dir, "multimodal_test_public.tsv")

    def load_split(self, filepath):
        df = pd.read_csv(
            filepath,
            sep="\t",
            usecols=["id", "clean_title", "2_way_label", "domain", "hasImage", "image_url"]
        )
        df = df.rename(columns={"clean_title": "text", "2_way_label": "label"})
        df = df.dropna(subset=["text"])
        return df

    def filter_to_available_images(self, df, image_dir):
        """
        Filter dataframe to only rows whose image file exists on disk.
        Call this BEFORE .sample() so your sample budget is not wasted
        on rows that have no image downloaded yet.
        """
        mask = df["id"].apply(
            lambda img_id: os.path.exists(os.path.join(image_dir, f"{img_id}.jpg"))
        )
        filtered = df[mask].reset_index(drop=True)
        print(
            f"  {len(df)} total rows → {len(filtered)} have images on disk "
            f"({len(df) - len(filtered)} skipped)"
        )
        return filtered

    def load_datasets(self, image_dir=None):
        """
        Load train, validation and test splits.

        Args:
            image_dir: If provided, each split is pre-filtered to only rows
                       whose image exists in this directory. Do this BEFORE
                       calling .sample() so you sample from valid rows only.
        """
        train_df = self.load_split(self.train_file)
        val_df   = self.load_split(self.val_file)
        test_df  = self.load_split(self.test_file)

        if image_dir is not None:
            print("Pre-filtering splits to available images...")
            print("  Train:", end=" ")
            train_df = self.filter_to_available_images(train_df, image_dir)
            print("  Val:  ", end=" ")
            val_df   = self.filter_to_available_images(val_df,   image_dir)
            print("  Test: ", end=" ")
            test_df  = self.filter_to_available_images(test_df,  image_dir)

        return train_df, val_df, test_df
