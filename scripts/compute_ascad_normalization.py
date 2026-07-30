from pathlib import Path

import h5py
import numpy as np


ASCAD_PATH = Path("data/raw/ascad/ASCAD_data/ASCAD_databases/ASCAD.h5")
OUTPUT_PATH = Path("data/processed/ascad_700_normalization.npz")

CHUNK_SIZE = 5_000
EPSILON = 1e-8


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(ASCAD_PATH, "r") as ascad_file:
        profiling_group = ascad_file["Profiling_traces"]
        assert isinstance(profiling_group, h5py.Group)

        traces = profiling_group["traces"]
        assert isinstance(traces, h5py.Dataset)

        num_traces, trace_length = traces.shape

        print(f"Profiling traces: {num_traces}")
        print(f"Trace length:     {trace_length}")
        print(f"Original dtype:   {traces.dtype}")
        print()

        # Float64 is used here to reduce numerical error while accumulating
        # statistics across all profiling traces.
        running_sum = np.zeros(trace_length, dtype=np.float64)
        running_squared_sum = np.zeros(trace_length, dtype=np.float64)

        for start in range(0, num_traces, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, num_traces)

            chunk = np.asarray(
                traces[start:end],
                dtype=np.float64
            )

            running_sum += chunk.sum(axis=0)
            running_squared_sum += np.square(chunk).sum(axis=0)

            print(f"Processed traces {start:5d}–{end - 1:5d}")

        mean = running_sum / num_traces

        variance = (
            running_squared_sum / num_traces
            - np.square(mean)
        )

        # Numerical rounding could theoretically produce tiny negative values.
        variance = np.maximum(variance, 0.0)

        std = np.sqrt(variance)

        near_zero_std = std < EPSILON
        num_near_zero = int(near_zero_std.sum())

        # Avoid division by zero for constant sample positions.
        std[near_zero_std] = 1.0

        mean = mean.astype(np.float32)
        std = std.astype(np.float32)

        np.savez(
            OUTPUT_PATH,
            mean=mean,
            std=std
        )

    print()
    print(f"Mean shape:                  {mean.shape}")
    print(f"Standard deviation shape:    {std.shape}")
    print(f"Constant/near-constant cols: {num_near_zero}")
    print(f"Mean range:                  [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"Std range:                   [{std.min():.4f}, {std.max():.4f}]")
    print()
    print(f"Statistics saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()