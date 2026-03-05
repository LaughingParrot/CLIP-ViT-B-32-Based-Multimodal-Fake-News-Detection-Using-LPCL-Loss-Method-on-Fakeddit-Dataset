import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:

    def __init__(
        self,
        encoder,
        classifier,
        train_dataset,
        val_dataset,
        batch_size=8,
        lr=1e-5,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):

        self.device = device

        # move models to GPU/CPU
        self.encoder = encoder.to(device)
        self.classifier = classifier.to(device)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4, # number of CPU processes loading data
            pin_memory=True # faster CPU → GPU memory transfer
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4, # number of CPU processes loading data
            pin_memory=True # faster CPU → GPU memory transfer
        )

        self.criterion = torch.nn.CrossEntropyLoss()

        # optimizer must train BOTH models
        self.optimizer = torch.optim.AdamW(
            list(self.encoder.parameters()) +
            list(self.classifier.parameters()),
            lr=lr
        )
        self.scaler = torch.GradScaler("cuda", enabled=(self.device == "cuda"))

    def train_epoch(self):

        self.encoder.train()
        self.classifier.train()

        total_loss = 0

        for batch in tqdm(self.train_loader):

            tokens = batch["text_tokens"].to(self.device)
            labels = batch["label"].to(self.device)

            # forward pass
            # features = self.encoder(tokens)

            # outputs = self.classifier(features)

            # loss = self.criterion(outputs, labels)

            # self.optimizer.zero_grad()

            # loss.backward()

            # self.optimizer.step()
            self.optimizer.zero_grad()

            with torch.autocast(device_type="cuda", enabled=(self.device == "cuda")):

                features = self.encoder(tokens)

                outputs = self.classifier(features)

                loss = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()

            self.scaler.step(self.optimizer)

            self.scaler.update()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate(self):

        self.encoder.eval()
        self.classifier.eval()

        correct = 0
        total = 0

        with torch.no_grad():

            for batch in self.val_loader:

                tokens = batch["text_tokens"].to(self.device)
                labels = batch["label"].to(self.device)

                features = self.encoder(tokens)

                outputs = self.classifier(features)

                predictions = torch.argmax(outputs, dim=1)

                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        return correct / total

    def train(self, epochs=3):

        if not torch.cuda.is_available():
            print("Cuda is not available, hence training cancelled.")
            return

        for epoch in range(epochs):

            train_loss = self.train_epoch()

            val_acc = self.validate()

            print(
                f"Epoch {epoch+1} | Loss: {train_loss:.4f} | Validation Accuracy: {val_acc:.4f}"
            )

        # save BOTH models
        torch.save(self.encoder.state_dict(), "encoder.pt")
        torch.save(self.classifier.state_dict(), "classifier.pt")

        print("Models saved successfully.")