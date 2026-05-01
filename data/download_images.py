import os
import requests
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import threading

from dataset_loader import FakedditDatasetLoader

SAVE_DIR = "Fakeddit/images"
os.makedirs(SAVE_DIR, exist_ok=True)

# Thread-local storage to maintain persistent connection pools per worker
thread_local = threading.local()

def get_session():
    if not hasattr(thread_local, "session"):
        thread_local.session = requests.Session()
        # Optimize TCP connection reuse and enable auto-retries for dropped packets
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=20, 
            pool_maxsize=20,
            max_retries=3
        )
        thread_local.session.mount('http://', adapter)
        thread_local.session.mount('https://', adapter)
    return thread_local.session

def download_image(url, path):
    session = get_session()
    try:
        # Tuple timeout: (connect_timeout, read_timeout)
        # Fails fast on dead servers (3s) to free up the thread immediately
        response = session.get(url, timeout=(3.0, 10.0))

        if response.status_code != 200:
            return False

        image = Image.open(BytesIO(response.content)).convert("RGB")
        image.save(path, format="JPEG", quality=85)

        return True

    except Exception:
        return False

def process_row(row):
    img_id = row["id"]
    url = row["image_url"]
    save_path = os.path.join(SAVE_DIR, f"{img_id}.jpg")

    # skip already downloaded
    if os.path.exists(save_path):
        return "skip"

    ok = download_image(url, save_path)
    return "ok" if ok else "fail"

def main():
    loader = FakedditDatasetLoader("Fakeddit")
    train_df, val_df, test_df = loader.load_datasets()

    # combine all splits
    df = pd.concat([train_df,val_df,test_df], ignore_index=True)

    # OPTIONAL: limit for testing (comment if full download)
    # df = df.sample(20000, random_state=42)

    print(f"Total images to process: {len(df)}")

    success = 0
    fail = 0
    skip = 0

    # Increased workers to 64 for maximum throughput
    max_threads = 64

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        # Submit futures to allow real-time accurate processing
        futures = [executor.submit(process_row, row) for row in df.to_dict("records")]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            res = future.result()
            if res == "ok":
                success += 1
            elif res == "fail":
                fail += 1
            else:
                skip += 1

    print("\nDownload Summary")
    print("----------------------")
    print("Downloaded :", success)
    print("Skipped    :", skip)
    print("Failed     :", fail)

if __name__ == "__main__":
    main()