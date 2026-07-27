from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from models.cnn_mamba_paper import PaperCNNMambaModel
from scripts.ascad_dataset import ASCADDataset


# ============================================================================
# Configuration
# ============================================================================

ASCAD_PATH = Path("data/raw/ascad/ASCAD_data/ASCAD_databases/ASCAD.h5")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RANDOM_SEED = 42

NUM_SAMPLES = 512
BATCH_SIZE = 64
NUM_EPOCHS = 200
LEARNING_RATE = 1e-4

TARGET_ACCURACY = 0.99


def set_seed(seed: int) -> None:
    """Make the experiment as reproducible as possible."""

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[float, float]:
    """Train the model for one epoch on the small diagnostic subset."""

    model.train()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for traces, labels in dataloader:
        traces = traces.to(
            DEVICE,
            dtype=torch.float32,
            non_blocking=True,
        )

        labels = labels.to(
            DEVICE,
            dtype=torch.long,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        logits = model(traces)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)

        running_loss += loss.item() * batch_size

        predictions = logits.argmax(dim=1)
        correct_predictions += (predictions == labels).sum().item()
        total_samples += batch_size

    mean_loss = running_loss / total_samples
    accuracy = correct_predictions / total_samples

    return mean_loss, accuracy


@torch.no_grad()
def evaluate_training_subset(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
) -> tuple[float, float]:
    """
    Evaluate the same subset in evaluation mode.

    This removes stochastic effects such as dropout and uses the inference
    behavior of normalization layers.
    """

    model.eval()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for traces, labels in dataloader:
        traces = traces.to(
            DEVICE,
            dtype=torch.float32,
            non_blocking=True,
        )

        labels = labels.to(
            DEVICE,
            dtype=torch.long,
            non_blocking=True,
        )

        logits = model(traces)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)

        running_loss += loss.item() * batch_size

        predictions = logits.argmax(dim=1)
        correct_predictions += (predictions == labels).sum().item()
        total_samples += batch_size

    mean_loss = running_loss / total_samples
    accuracy = correct_predictions / total_samples

    return mean_loss, accuracy


def main() -> None:
    set_seed(RANDOM_SEED)

    print(f"Using device: {DEVICE}")
    print()

    full_dataset = ASCADDataset(
        ascad_path=ASCAD_PATH,
        variant="700",
        split="profiling",
    )

    if NUM_SAMPLES > len(full_dataset):
        raise ValueError(
            f"NUM_SAMPLES={NUM_SAMPLES} exceeds dataset size "
            f"{len(full_dataset)}."
        )

    # Select a deterministic random subset instead of simply taking the first
    # traces.
    generator = torch.Generator().manual_seed(RANDOM_SEED)

    subset_indices = torch.randperm(
        len(full_dataset),
        generator=generator,
    )[:NUM_SAMPLES].tolist()

    training_subset = Subset(
        full_dataset,
        subset_indices,
    )

    train_loader = DataLoader(
        training_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # A separate loader without shuffling is used to measure accuracy on the
    # exact same samples in evaluation mode.
    evaluation_loader = DataLoader(
        training_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = PaperCNNMambaModel(
        num_classes=256,
    ).to(DEVICE)

    num_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"Subset size:          {len(training_subset)}")
    print(f"Batch size:           {BATCH_SIZE}")
    print(f"Learning rate:        {LEARNING_RATE}")
    print(f"Maximum epochs:       {NUM_EPOCHS}")
    print(f"Total parameters:     {num_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")
    print()

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0,
    )

    start_time = time.perf_counter()

    reached_target = False

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.perf_counter()

        train_loss, train_accuracy = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
        )

        evaluation_loss, evaluation_accuracy = evaluate_training_subset(
            model=model,
            dataloader=evaluation_loader,
            criterion=criterion,
        )

        epoch_time = time.perf_counter() - epoch_start

        print(
            f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Eval Loss: {evaluation_loss:.4f} | "
            f"Eval Acc: {evaluation_accuracy:.4f} | "
            f"Time: {epoch_time:.2f} s"
        )

        if evaluation_accuracy >= TARGET_ACCURACY:
            reached_target = True

            print()
            print(
                f"Target accuracy reached: "
                f"{evaluation_accuracy:.2%} at epoch {epoch}."
            )

            break

    total_time = time.perf_counter() - start_time

    print()
    print(f"Test completed in {total_time:.2f} seconds.")

    if reached_target:
        print(
            "Result: the model successfully memorized the small subset."
        )
    else:
        print(
            "Result: the model did not reach the target accuracy."
        )

    full_dataset.close()


if __name__ == "__main__":
    main()