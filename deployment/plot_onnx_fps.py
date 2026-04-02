import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

CSV_PATH = r"outputs\deployment\onnx_resolution_benchmark.csv"
OUT_PNG = r"outputs\deployment\fps_vs_resolution.png"

def main():
    df = pd.read_csv(CSV_PATH)
    df = df.sort_values("img_size")

    Path("outputs/deployment").mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(df["img_size"], df["fps"], marker="o")
    plt.xlabel("Input Resolution")
    plt.ylabel("FPS")
    plt.title("ONNX FP32 FPS vs Input Resolution")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=200)

    print(f"Saved: {OUT_PNG}")

if __name__ == "__main__":
    main()