from __future__ import annotations

from contextlib import nullcontext
from math import ceil
import os
import time
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.amp.grad_scaler import GradScaler
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def in_batch_lpcl_loss(text_feat: Tensor, image_feat: Tensor, temperature: Tensor) -> Tensor:
    logits = (text_feat @ image_feat.T) * temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


class Trainer:
    def __init__(
        self,
        model: Module,
        train_dataset: Dataset[Any],
        val_dataset: Dataset[Any],
        batch_size: int = 6,
        accumulation_steps: int = 4,
        lr: float = 1e-5,
        warmup_pct: float = 0.10,
        early_stopping_patience: int = 3,
        grad_clip: float | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        num_workers: int | None = None,
        log_every: int = 20,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)

        self.accumulation_steps = max(1, accumulation_steps)
        self.warmup_pct = min(max(warmup_pct, 0.0), 0.99)
        self.patience = max(1, early_stopping_patience)
        self.grad_clip = grad_clip
        self.log_every = max(1, log_every)

        self._head_lr = lr
        self._encoder_lr = lr / 10.0

        if num_workers is None:
            num_workers = 0 if os.name == "nt" else 2

        loader_kwargs: dict[str, Any] = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": self.device.type == "cuda",
        }
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 2

        self.train_loader: DataLoader[Any] = DataLoader(
            train_dataset,
            shuffle=True,
            **loader_kwargs,
        )
        self.val_loader: DataLoader[Any] = DataLoader(
            val_dataset,
            shuffle=False,
            **loader_kwargs,
        )

        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

        encoder_params: list[Tensor] = []
        head_params: list[Tensor] = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(key in name for key in ("clip_model", "text_encoder", "image_encoder")):
                encoder_params.append(param)
            else:
                head_params.append(param)

        self.optimizer: Optimizer = torch.optim.AdamW(
            [
                {"params": encoder_params, "lr": self._encoder_lr},
                {"params": head_params, "lr": self._head_lr},
            ],
            weight_decay=0.05,
            eps=1e-6,
        )

        self.use_amp = self.device.type == "cuda"
        self.amp_dtype = (
            torch.bfloat16 if (self.use_amp and torch.cuda.is_bf16_supported()) else torch.float16
        )
        self.use_grad_scaler = self.use_amp and self.amp_dtype == torch.float16
        self.scaler: GradScaler | None = GradScaler("cuda") if self.use_grad_scaler else None

        self.non_blocking = self.device.type == "cuda"
        self._skipped_steps = 0
        self._vram_total_mb = (
            torch.cuda.get_device_properties(self.device).total_memory / 1024**2
            if self.device.type == "cuda"
            else 0.0
        )

    def _vram_str(self) -> str:
        if self.device.type != "cuda":
            return "N/A"
        used_mb = torch.cuda.memory_allocated(self.device) / 1024**2
        return f"{used_mb:,.0f}/{self._vram_total_mb:,.0f}MB"

    def _has_finite_grads(self) -> bool:
        for param in self.model.parameters():
            grad = param.grad
            if grad is not None and not torch.isfinite(grad).all():
                return False
        return True

    def _train_epoch(self, scheduler: OneCycleLR, epoch_idx: int, total_epochs: int) -> float:
        self.model.train()
        n_batches = len(self.train_loader)
        if n_batches == 0:
            return 0.0

        total_loss = 0.0
        running_loss = 0.0
        self._skipped_steps = 0
        self.optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch_idx + 1}/{total_epochs} [train]",
            leave=False,
            unit="batch",
            dynamic_ncols=True,
        )

        for i, batch in enumerate(pbar):
            tokens = batch["text_tokens"].to(self.device, non_blocking=self.non_blocking)
            images = batch["image"].to(self.device, non_blocking=self.non_blocking)
            labels = batch["label"].to(self.device, non_blocking=self.non_blocking)

            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=self.amp_dtype)
                if self.use_amp
                else nullcontext()
            )

            with autocast_ctx:
                logits, text_feat, image_feat, temp = self.model(tokens, images)
                cls_loss = self.criterion(logits, labels)
                align_loss = in_batch_lpcl_loss(text_feat, image_feat, temp)
                loss = (cls_loss + 0.1 * align_loss) / self.accumulation_steps

            if self.use_grad_scaler and self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            should_step = ((i + 1) % self.accumulation_steps == 0) or ((i + 1) == n_batches)
            if should_step:
                if self.use_grad_scaler and self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)

                if self.grad_clip is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                    can_step = bool(torch.isfinite(grad_norm))
                else:
                    can_step = self._has_finite_grads()

                if can_step:
                    if self.use_grad_scaler and self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    scheduler.step()
                else:
                    self._skipped_steps += 1
                    if self.use_grad_scaler and self.scaler is not None:
                        self.scaler.update()

                self.optimizer.zero_grad(set_to_none=True)

            raw_loss = float(loss.detach().item() * self.accumulation_steps)
            total_loss += raw_loss
            running_loss = raw_loss if i == 0 else (0.9 * running_loss + 0.1 * raw_loss)

            if ((i + 1) % self.log_every == 0) or ((i + 1) == n_batches):
                pbar.set_postfix(
                    {
                        "loss": f"{running_loss:.4f}",
                        "lr": f"{self.optimizer.param_groups[1]['lr']:.2e}",
                        "vram": self._vram_str(),
                    }
                )

        return total_loss / n_batches

    def _validate(self, epoch_idx: int, total_epochs: int) -> tuple[float, float]:
        self.model.eval()
        total_samples = 0
        correct = 0
        total_loss = 0.0

        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch_idx + 1}/{total_epochs} [val]",
            leave=False,
            unit="batch",
            dynamic_ncols=True,
        )

        with torch.inference_mode():
            for batch in pbar:
                tokens = batch["text_tokens"].to(self.device, non_blocking=self.non_blocking)
                images = batch["image"].to(self.device, non_blocking=self.non_blocking)
                labels = batch["label"].to(self.device, non_blocking=self.non_blocking)

                logits, _, _, _ = self.model(tokens, images)
                preds = torch.argmax(logits, dim=1)

                batch_size = labels.size(0)
                batch_loss = float(self.criterion(logits, labels).item())

                total_loss += batch_loss * batch_size
                correct += int((preds == labels).sum().item())
                total_samples += int(batch_size)

                if (total_samples % max(1, self.log_every * self.val_loader.batch_size)) == 0:
                    pbar.set_postfix({"val_loss": f"{batch_loss:.4f}"})

        if total_samples == 0:
            return 0.0, 0.0

        return correct / total_samples, total_loss / total_samples

    def train(self, epochs: int = 3, output_file: str = "training_log.txt") -> None:
        if len(self.train_loader) == 0:
            raise ValueError("Training dataset is empty after preprocessing.")

        effective_epochs = max(1, epochs)
        steps_per_epoch = ceil(len(self.train_loader) / self.accumulation_steps)
        total_steps = max(1, steps_per_epoch * effective_epochs)

        scheduler = OneCycleLR(
            self.optimizer,
            max_lr=[self._encoder_lr, self._head_lr],
            total_steps=total_steps,
            pct_start=self.warmup_pct,
            anneal_strategy="cos",
            div_factor=10.0,
            final_div_factor=1e3,
            three_phase=False,
        )

        header = (
            f"{'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>8}  "
            f"{'Val Acc':>7}  {'LR (head)':>10}  {'VRAM':>13}  {'Skip':>4}  {'Time':>6}"
        )
        sep = "-" * len(header)

        print(f"\n{sep}\n{header}\n{sep}")

        best_val_acc = 0.0
        best_epoch = 0
        patience_counter = 0

        with open(output_file, "w", encoding="utf-8", newline="\n") as log_file:
            log_file.write(header + "\n" + sep + "\n")

            epoch_bar = tqdm(
                range(effective_epochs),
                desc="Overall",
                unit="epoch",
                leave=True,
                dynamic_ncols=True,
            )

            for epoch in epoch_bar:
                start = time.time()

                train_loss = self._train_epoch(scheduler, epoch, effective_epochs)
                val_acc, val_loss = self._validate(epoch, effective_epochs)

                elapsed = time.time() - start
                head_lr = self.optimizer.param_groups[1]["lr"]

                row = (
                    f"{epoch + 1:>5}  {train_loss:>10.4f}  {val_loss:>8.4f}  "
                    f"{val_acc:>7.4f}  {head_lr:>10.2e}  {self._vram_str():>13}  "
                    f"{self._skipped_steps:>4}  {elapsed:>5.0f}s"
                )

                print("\n" + row)
                log_file.write(row + "\n")

                if self._skipped_steps:
                    note = f"         [!] Skipped non-finite optimizer steps: {self._skipped_steps}"
                    print(note)
                    log_file.write(note + "\n")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch + 1
                    patience_counter = 0
                    torch.save(self.model.state_dict(), "multimodal_model.pt")
                    msg = f"         [+] Checkpoint updated at val acc {best_val_acc:.4f}"
                else:
                    patience_counter += 1
                    msg = (
                        f"         [-] No improvement ({patience_counter}/{self.patience}). "
                        f"Best {best_val_acc:.4f} at epoch {best_epoch}."
                    )

                print(msg)
                log_file.write(msg + "\n")
                log_file.flush()

                epoch_bar.set_postfix(
                    {
                        "val_acc": f"{val_acc:.4f}",
                        "val_loss": f"{val_loss:.4f}",
                        "best": f"{best_val_acc:.4f}",
                    }
                )

                if patience_counter >= self.patience:
                    stop_msg = (
                        f"\nEarly stopping at epoch {epoch + 1}. "
                        f"Best val acc {best_val_acc:.4f} at epoch {best_epoch}."
                    )
                    print(stop_msg)
                    log_file.write(stop_msg + "\n")
                    break

        print(
            f"\n{sep}\n"
            f"Done. Best checkpoint: multimodal_model.pt (epoch {best_epoch}, val acc {best_val_acc:.4f})\n"
        )
