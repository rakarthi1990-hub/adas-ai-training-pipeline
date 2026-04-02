import time
import statistics
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort

MODEL_PATH = r"src/runs/runs/nuscenes_yolov8n/weights/best_int8.onnx"
IMG_DIR = r"C:\Users\Karthikeyan\adas-ai-training-pipeline\datasets\nuscenes_yolo\images\val"  # update if needed

IMG_SIZE = 640
RUNS = 50
WARMUP = 10

def load_images(img_dir, imgsz):
    paths = list(Path(img_dir).glob("*.jpg"))
    imgs = []
    for p in paths[:100]:  # limit
        img = cv2.imread(str(p))
        img = cv2.resize(img, (imgsz, imgsz))
        img = img[:, :, ::-1] / 255.0
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        imgs.append(img)
    return imgs

def main():
    session = ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name
    images = load_images(IMG_DIR, IMG_SIZE)
    print(f"Loaded {len(images)} images")

    latencies = []

    for i in range(RUNS + WARMUP):
        img = images[i % len(images)]
        img = np.expand_dims(img, axis=0)

        start = time.perf_counter()
        _ = session.run(None, {input_name: img})
        end = time.perf_counter()

        if i >= WARMUP:
            latencies.append((end - start) * 1000)

    avg = statistics.mean(latencies)
    fps = 1000 / avg

    print(f"\nAvg latency: {avg:.2f} ms")
    print(f"FPS: {fps:.2f}")

if __name__ == "__main__":
    main()