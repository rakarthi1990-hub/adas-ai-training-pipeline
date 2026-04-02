from ultralytics import YOLO
from pathlib import Path
import shutil

MODEL_PATH = r"src/runs/runs/nuscenes_yolov8n/weights/best.pt"
WEIGHTS_DIR = Path(r"src/runs/runs/nuscenes_yolov8n/weights")
IMG_SIZES = [320, 480, 640]

def main():
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = YOLO(MODEL_PATH)

    for imgsz in IMG_SIZES:
        print(f"\nExporting ONNX for imgsz={imgsz} ...")
        exported_path = model.export(
            format="onnx",
            imgsz=imgsz,
            dynamic=False,
            simplify=False
        )

        exported_path = Path(exported_path)
        target_path = WEIGHTS_DIR / f"best_{imgsz}.onnx"

        if exported_path.resolve() != target_path.resolve():
            shutil.copy2(exported_path, target_path)

        print(f"Saved: {target_path}")
        print(f"Size: {target_path.stat().st_size / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    main()