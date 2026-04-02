from onnxruntime.quantization import quantize_dynamic, QuantType
from pathlib import Path

FP32_MODEL = r"src/runs/runs/nuscenes_yolov8n/weights/best.onnx"
INT8_MODEL = r"src/runs/runs/nuscenes_yolov8n/weights/best_int8.onnx"

def main():
    if not Path(FP32_MODEL).exists():
        raise FileNotFoundError("ONNX model not found")

    quantize_dynamic(
        model_input=FP32_MODEL,
        model_output=INT8_MODEL,
        weight_type=QuantType.QInt8
    )

    size_fp32 = Path(FP32_MODEL).stat().st_size / (1024 * 1024)
    size_int8 = Path(INT8_MODEL).stat().st_size / (1024 * 1024)

    print(f"\nFP32 size: {size_fp32:.2f} MB")
    print(f"INT8 size: {size_int8:.2f} MB")

if __name__ == "__main__":
    main()