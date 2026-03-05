import torch
import open_clip
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from data.dataset_loader import FakedditDatasetLoader
from data.preprocessing import FakedditTextDataset

from models.text_encoder import FakeNewsTextEncoder
from models.classifier import FakeNewsClassifier
from sklearn.metrics import confusion_matrix

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# SAFETY CHECK
# -------------------------------
if not os.path.exists("encoder.pt") or not os.path.exists("classifier.pt"):
    print("\nERROR: Trained model not found.")
    print("Please run training first:")
    print("   python train_model.py\n")
    exit()


# -------------------------------
# LOAD DATASET
# -------------------------------
dataset_path = "Fakeddit"

loader = FakedditDatasetLoader(dataset_path)

_, _, test_df = loader.load_datasets()

# restrict to 500 rows
test_df = test_df.head(500)

tokenizer = open_clip.get_tokenizer("ViT-B-32")

test_dataset = FakedditTextDataset(test_df, tokenizer)


# -------------------------------
# LOAD MODELS
# -------------------------------
encoder = FakeNewsTextEncoder().to(device)
classifier = FakeNewsClassifier().to(device)

encoder.load_state_dict(torch.load("encoder.pt", map_location=device))
classifier.load_state_dict(torch.load("classifier.pt", map_location=device))

encoder.eval()
classifier.eval()


# -------------------------------
# TABLE HEADER
# -------------------------------
total_rows = len(test_dataset)

print(
    f"{'COUNT':8} {'ID':10} {'DOMAIN':18} {'TITLE':55} {'PREDICTED':10} {'CONF':6} {'TRUE':6} {'NOTE':18}"
)

print("-"*150)


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

    label = sample["label"].item()

    with torch.no_grad():

        features = encoder(tokens)

        logits = classifier(features)

        probs = torch.softmax(logits, dim=1)

        confidence, pred = torch.max(probs, dim=1)

    pred = pred.item()
    confidence = confidence.item()

    pred_label = "Fake" if pred == 1 else "Real"
    true_label = "Fake" if label == 1 else "Real"

    title = str(test_df.iloc[i]["text"])[:90]
    domain = str(test_df.iloc[i]["domain"])[:18]
    post_id = str(test_df.iloc[i]["id"])

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
        "confidence": round(confidence,3),
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

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel() # compute confusion matrix

print("\nConfusion Matrix Components")
print("----------------------------------")
print("True Positive  (A.Fake→P.Fake) :", tp)
print("True Negative  (A.Real→P.Real) :", tn)
print("False Positive (A.Real→P.Fake) :", fp)
print("False Negative (A.Fake→P.Real) :", fn)

print("\nEvaluation Summary")
print("------------------------------")
correct = tn+tp
incorrect = total_rows - correct
print("Correct Predictions   :", correct)
print("Incorrect Predictions :", incorrect)
print("Sent To Expert        :", expert_cases)



print("\nModel Metrics")
print("----------------------")
print("Accuracy :", f"{float(accuracy):.4f}")
print("Precision:", f"{float(precision):.4f}")
print("Recall   :", f"{float(recall):.4f}")
print("F1 Score :", f"{float(f1):.4f}")
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