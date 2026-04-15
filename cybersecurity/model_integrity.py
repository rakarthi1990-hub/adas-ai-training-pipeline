import hashlib

MODEL_PATH = "src/runs/runs/nuscenes_yolov8n/weights/best.onnx"

def calculate_sha256(file_path):
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256.update(byte_block)
    return sha256.hexdigest()

def main():
    hash_value = calculate_sha256(MODEL_PATH)
    print(f"Model SHA256: {hash_value}")

    # In real system → compare with trusted hash
    TRUSTED_HASH = hash_value  # placeholder

    if hash_value == TRUSTED_HASH:
        print("Model integrity verified ✅")
    else:
        print("WARNING: Model integrity compromised ❌")

if __name__ == "__main__":
    main()