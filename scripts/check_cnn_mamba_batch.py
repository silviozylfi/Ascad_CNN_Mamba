from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models.cnn_mamba_paper import PaperCNNMambaModel
from scripts.ascad_dataset import ASCADDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ASCAD_700_PATH = (
    PROJECT_ROOT
    / "data/raw/ascad/ASCAD_data/ASCAD_databases/ASCAD.h5"
)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.device("cuda")

    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    dataset = ASCADDataset(
        ascad_path=ASCAD_700_PATH,
        variant="700",
        split="profiling",
    )

    try:
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0,
        )

        traces, labels = next(iter(dataloader))

        print(f"Input traces shape: {traces.shape}")
        print(f"Input labels shape: {labels.shape}")
        print(f"Labels range: [{labels.min().item()}, {labels.max().item()}]")
        print()

        traces = traces.to(device)
        labels = labels.to(device)

        model = PaperCNNMambaModel(num_classes=256).to(device)
        model.eval()

        with torch.no_grad():
            logits = model(traces)

        print(f"Output logits shape: {logits.shape}")

        expected_shape = (traces.shape[0], 256)

        if logits.shape != expected_shape:
            raise ValueError(
                f"Unexpected output shape: "
                f"expected {expected_shape}, "
                f"found {tuple(logits.shape)}."
            )

        print()
        print("Forward pass completed successfully.")

    finally:
        dataset.close()


if __name__ == "__main__":
    main()