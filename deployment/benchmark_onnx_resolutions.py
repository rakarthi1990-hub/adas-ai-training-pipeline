import time
import statistics
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import onnxruntime as ort

MODEL_PATHS = {
    320: r"src/runs/runs/nuscenes_yolov8n/weights/best_320.onnx",
    480: r"src/runs/runs/nuscenes_yolov8n/weights/best_480.onnx",
    640: r"src/runs/runs/nuscenes_yolov8n/weights/best_640.onnx",
}

IMG_DIR = r"C:\Users\Karthikeyan\adas-ai-training-pipeline\datasets\nuscenes_yolo\images\val"
RUNS = 50
WARMUP = 10
OUT_CSV = r"outputs\deployment\onnx_resolution_benchmark.csv"


def load_images(img_dir, imgsz):
    exts = {".jpg", ".jpeg", ".png"}
    paths = [p for p in Path(img_dir).iterdir() if p.suffix.lower() in exts]

    imgs = []
    for p in paths[:100]:
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = cv2.resize(img, (imgsz, imgsz))
        img = img[:, :, ::-1] / 255.0
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        imgs.append(img)
    return imgs


def benchmark_model(model_path, images):
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    latencies = []

    for i in range(RUNS + WARMUP):
        img = images[i % len(images)]
        img = np.expand_dims(img, axis=0)

        start = time.perf_counter()
        _ = session.run(None, {input_name: img})
        end = time.perf_counter()

        if i >= WARMUP:
            latencies.append((end - start) * 1000)

    avg_latency = statistics.mean(latencies)
    fps = 1000 / avg_latency
    size_mb = Path(model_path).stat().st_size / (1024 * 1024)
    return avg_latency, fps, size_mb


def main():
    Path("outputs/deployment").mkdir(parents=True, exist_ok=True)
    rows = []

    for imgsz, model_path in MODEL_PATHS.items():
        images = load_images(IMG_DIR, imgsz)
        print(f"Loaded {len(images)} images for {imgsz}")

        if not images:
            raise ValueError(f"No images loaded for size {imgsz}")

        avg_latency, fps, size_mb = benchmark_model(model_path, images)

        row = {
            "variant": f"onnx_fp32_{imgsz}",
            "img_size": imgsz,
            "num_images": len(images),
            "avg_latency_ms": round(avg_latency, 2),
            "fps": round(fps, 2),
            "model_size_mb": round(size_mb, 2),
        }
        rows.append(row)
        print(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    print(f"\nSaved: {OUT_CSV}")
    print(df)


if __name__ == "__main__":
    main()