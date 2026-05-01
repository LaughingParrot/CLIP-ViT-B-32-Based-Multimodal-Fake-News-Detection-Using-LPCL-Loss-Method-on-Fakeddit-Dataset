import torch
import open_clip
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from data.dataset_loader import FakedditDatasetLoader
from data.preprocessing import FakedditMultimodalDataset

from models.multimodal_model import FakeNewsMultimodalModel


if not torch.cuda.is_available():
    raise RuntimeError("No GPU detected. This script requires a CUDA-capable GPU.")

device = "cuda"


# -------------------------------
# SAFETY CHECK
# -------------------------------
if not os.path.exists("multimodal_model.pt"):
    print("\nERROR: Trained multimodal model not found.")
    print("Please run training first:")
    print("   python train_model.py\n")
    exit()


# -------------------------------
# LOAD DATASET
# -------------------------------
dataset_path = "Fakeddit"
image_dir = "Fakeddit/images"

loader = FakedditDatasetLoader(dataset_path)

train_df, val_df, test_df = loader.load_datasets(image_dir=image_dir)

# Fall back to val, then train, if test images are missing on disk
if len(test_df) == 0:
    print("\nWARNING: No images found in test split. Trying val split...")
    if len(val_df) > 0:
        print("INFO: Using val split for evaluation.")
        test_df = val_df
    elif len(train_df) > 0:
        print("INFO: Val split also empty. Falling back to train split.")
        test_df = train_df
    else:
        print("\nERROR: No valid image-text rows found in any split.")
        exit()

# test_df is already pre-filtered to rows with images on disk — sample directly from those
n_test = min(500, len(test_df))
test_df = test_df.sample(n=n_test, random_state=42).reset_index(drop=True)
print(f"INFO: Sampled {n_test} rows from {len(test_df)} valid image rows.")


# -------------------------------
# TOKENIZER + IMAGE PREPROCESS
# -------------------------------
model_name = "ViT-B-32"

tokenizer = open_clip.get_tokenizer(model_name)

_, _, image_preprocess = open_clip.create_model_and_transforms(
    model_name,
    pretrained="openai"
)


# -------------------------------
# DATASET (FILTERED BY IMAGES)
# -------------------------------
test_dataset = FakedditMultimodalDataset(
    test_df,
    tokenizer,
    image_preprocess,
    image_dir
)

total_rows = len(test_dataset)

if total_rows == 0:
    print("\nERROR: No valid image-text samples found in test set.")
    exit()


# -------------------------------
# LOAD MODEL
# -------------------------------
model = FakeNewsMultimodalModel().to(device)

model.load_state_dict(torch.load("multimodal_model.pt"))

model.eval()


# -------------------------------
# PRINT HEADER
# -------------------------------
print(
    f"{'COUNT':8} {'ID':10} {'DOMAIN':18} {'TITLE':55} {'PREDICTED':10} {'CONF':6} {'TRUE':6} {'NOTE':18}"
)

print("-" * 150)


# -------------------------------
# EVALUATION
# -------------------------------
correct = 0
incorrect = 0
expert_cases = 0

y_true = []
y_pred = []

results = []

confidence_threshold = 0.65


for i in range(total_rows):

    sample = test_dataset[i]

    tokens = sample["text_tokens"].unsqueeze(0).to(device)
    image = sample["image"].unsqueeze(0).to(device)

    label = sample["label"].item()

    with torch.no_grad():

        logits, _, _, _ = model(tokens, image)

        probs = torch.softmax(logits, dim=1)

        confidence, pred = torch.max(probs, dim=1)

    pred = pred.item()
    confidence = float(confidence.item())

    pred_label = "Fake" if pred == 1 else "Real"
    true_label = "Fake" if label == 1 else "Real"

    # dataframe index aligned after filtering
    df_row = test_dataset.ids[i]

    title = str(test_dataset.texts[i])[:90]

    # fallback if domain missing
    domain = str(test_df.iloc[i]["domain"])[:18] if "domain" in test_df.columns else "NA"
    post_id = str(df_row)

    note = ""

    if pred == 1 and label == 1:
        note = "(O) True_positive"
    elif pred == 0 and label == 0:
        note = "(O) True_negative"
    elif pred == 1 and label == 0:
        note = "(X) False_positive"
    elif pred == 0 and label == 1:
        note = "(X) False_negative"

    if confidence < confidence_threshold:
        expert_cases += 1
        note = "(E) Directed_to_expert"

    print(
        f"{str(i+1)+'/'+str(total_rows):8} "
        f"{post_id:10} "
        f"{domain:18} "
        f"{title:90} "
        f"{pred_label:10} "
        f"{confidence:.2f} "
        f"{true_label:6} "
        f"{note:18}"
    )

    y_true.append(label)
    y_pred.append(pred)

    results.append({
        "count": f"{i+1}/{total_rows}",
        "id": post_id,
        "domain": domain,
        "title": title,
        "predicted": pred_label,
        "confidence": round(confidence, 3),
        "true_label": true_label,
        "note": note
    })


# -------------------------------
# METRICS
# -------------------------------
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

print("\nConfusion Matrix Components")
print("----------------------------------")
print("True Positive  (A.Fake→P.Fake) :", tp)
print("True Negative  (A.Real→P.Real) :", tn)
print("False Positive (A.Real→P.Fake) :", fp)
print("False Negative (A.Fake→P.Real) :", fn)

print("\nEvaluation Summary")
print("------------------------------")
correct = tn + tp
incorrect = total_rows - correct

print("Correct Predictions   :", correct)
print("Incorrect Predictions :", incorrect)
print("Sent To Expert        :", expert_cases)

print("\nModel Metrics")
print("----------------------")
print("Accuracy :", f"{accuracy:.4f}")
print("Precision:", f"{precision:.4f}")
print("Recall   :", f"{recall:.4f}")
print("F1 Score :", f"{f1:.4f}")

expert_ratio = expert_cases / total_rows
print("Expert Review Ratio   :", f"{expert_ratio:.4f}")


# -------------------------------
# SAVE RESULTS
# -------------------------------
results_df = pd.DataFrame(results)

results_df.to_csv(
    "evaluation_results.tsv",
    sep="\t",
    index=False
)

print("\nResults saved to evaluation_results.tsv")