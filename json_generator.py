import pandas as pd
import json
import os

def generate_sample_jsons():
    tsv_path = "Fakeddit/multimodal_test_public.tsv"
    out_dir = "interface/sample_jsons"

    os.makedirs(out_dir, exist_ok=True)

    # Load and clean TSV
    df = pd.read_csv(tsv_path, sep="\t")
    df = df.dropna(subset=["id", "clean_title", "image_url", "2_way_label"])
    df["2_way_label"] = df["2_way_label"].astype(int)
    df = df.reset_index(drop=True)   # ensure clean 0-based integer index

    total_rows = len(df)
    print(f"Total usable rows in TSV: {total_rows}")

    # File sizes: 100, 150, 200, 250, 300, 350, 400, 450, 500, 550
    file_sizes = list(range(100, 600, 50))

    offset = 0  # running cursor — advances after each file is written

    for count in file_sizes:
        if offset >= total_rows:
            print(f"Dataset exhausted at offset {offset}. Stopping early.")
            break

        # Clamp so we never read past the end of the dataset
        actual_count = min(count, total_rows - offset)
        sliced_df = df.iloc[offset : offset + actual_count]

        json_data = [
            {
                "id":        str(row["id"]),
                "text":      str(row["clean_title"]),
                "image_url": str(row["image_url"]),
                "label":     int(row["2_way_label"]),
            }
            for _, row in sliced_df.iterrows()
        ]

        out_file = os.path.join(out_dir, f"sample_news_{count}.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=4)

        print(
            f"Generated: {out_file}  "
            f"(rows {offset}–{offset + actual_count - 1}, "
            f"{actual_count} items)"
        )

        offset += actual_count   # advance cursor to next unused row

if __name__ == "__main__":
    generate_sample_jsons()