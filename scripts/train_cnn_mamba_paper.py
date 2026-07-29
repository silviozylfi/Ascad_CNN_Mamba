from pathlib import Path
import time
from xml.parsers.expat import model

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset, random_split

from models.cnn_mamba_paper import PaperCNNMambaModel
from scripts.ascad_dataset import ASCADDataset
from scripts.transforms import NormalizeTrace


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ASCAD_700_PATH = (
    PROJECT_ROOT
    / "data/raw/ascad/ASCAD_data/ASCAD_databases/ASCAD.h5"
)

NORMALIZATION_PATH = Path(
    "data/processed/ascad_700_normalization.npz"
)

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

# =============================================================================
# Hyperparameters
# =============================================================================

BATCH_SIZE = 32
LEARNING_RATE = 1e-5
EPOCHS = 150

TRAIN_RATIO = 0.9
VALIDATION_RATIO = 0.1

NUM_WORKERS = 0

RANDOM_SEED = 42

# =============================================================================
# Device
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# Data loaders
# =============================================================================

def create_dataloaders() -> tuple[ASCADDataset, DataLoader, DataLoader]:
    """
    Create the training and validation DataLoaders.

    The profiling traces are randomly split into:
    - 90% training
    - 10% validation
    """
    trace_transform = NormalizeTrace(NORMALIZATION_PATH)


    dataset = ASCADDataset(
        ascad_path=ASCAD_700_PATH,
        variant="700",
        split="profiling",
        transform=trace_transform
    )


    train_size = int(TRAIN_RATIO * len(dataset))
    validation_size = int(VALIDATION_RATIO * len(dataset))

    train_dataset, validation_dataset = random_split(
        dataset,
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    return dataset, train_loader, validation_loader

# =============================================================================
# Training
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[float, float]:
    """
    Train the model for one epoch.

    Returns:
        (average_loss, accuracy)
    """

    model.train()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for traces, labels in dataloader:

        traces = traces.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        logits = model(traces)

        loss = criterion(logits, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * traces.size(0)

        predictions = logits.argmax(dim=1)

        correct_predictions += (predictions == labels).sum().item()

        total_samples += labels.size(0)

    average_loss = running_loss / total_samples
    accuracy = correct_predictions / total_samples

    return average_loss, accuracy

# =============================================================================
# Validation
# =============================================================================

@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
) -> tuple[float, float]:
    """
    Evaluate the model on the validation set.

    Returns:
        (average_loss, accuracy)
    """

    model.eval()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for traces, labels in dataloader:

        traces = traces.to(DEVICE)
        labels = labels.to(DEVICE)

        logits = model(traces)

        loss = criterion(logits, labels)

        running_loss += loss.item() * traces.size(0)

        predictions = logits.argmax(dim=1)

        correct_predictions += (predictions == labels).sum().item()

        total_samples += labels.size(0)

    average_loss = running_loss / total_samples
    accuracy = correct_predictions / total_samples

    return average_loss, accuracy

# =============================================================================
# Checkpoint
# =============================================================================

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_loss: float,
    path: Path,
) -> None:
    """
    Save a training checkpoint.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "validation_loss": validation_loss,
    }

    torch.save(checkpoint, path)

# =============================================================================
# Main
# =============================================================================

def main() -> None:

    print(f"Using device: {DEVICE}")
    print()

    training_start_time = time.perf_counter()

    dataset, train_loader, validation_loader = create_dataloaders()

    try:

        model = PaperCNNMambaModel(num_classes=256).to(DEVICE)

        num_parameters = sum(p.numel() for p in model.parameters())
        trainable_parameters = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )       

        print(f"Total parameters:     {num_parameters:,}")
        print(f"Trainable parameters: {trainable_parameters:,}")
        print()

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        optimizer = Adam(
            model.parameters(),
            lr=LEARNING_RATE,
        )

        scheduler = None

        best_validation_loss = float("inf")

        for epoch in range(EPOCHS):

            epoch_start_time = time.perf_counter()

            train_loss, train_accuracy = train_one_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
            )

            validation_loss, validation_accuracy = evaluate(
                model=model,
                dataloader=validation_loader,
                criterion=criterion,
            )

            if scheduler is not None:
                scheduler.step()

            epoch_time = time.perf_counter() - epoch_start_time
            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch + 1:3d}/{EPOCHS} | "
                f"LR: {current_lr:.2e} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_accuracy:.4f} | "
                f"Val Loss: {validation_loss:.4f} | "
                f"Val Acc: {validation_accuracy:.4f} | "
                f"Time: {epoch_time:.2f} s"
            )

            if validation_loss < best_validation_loss:

                best_validation_loss = validation_loss

                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch + 1,
                    validation_loss=validation_loss,
                    path=CHECKPOINTS_DIR / "cnn_mamba_paper_best.pt",
                )

                print("Best model updated.")

        training_time = time.perf_counter() - training_start_time

        hours = int(training_time // 3600)
        minutes = int((training_time % 3600) // 60)
        seconds = training_time % 60

        print()
        print(
            f"Training completed in "
            f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"
        )

    finally:

        dataset.close()


if __name__ == "__main__":
    main()