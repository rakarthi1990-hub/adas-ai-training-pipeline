import cv2
import numpy as np

IMAGE_PATH = "datasets/nuscenes_yolo/images/val"  # adjust if needed

def add_noise(image, epsilon=10):
    noise = np.random.randint(-epsilon, epsilon, image.shape, dtype=np.int16)
    adv_image = image.astype(np.int16) + noise
    adv_image = np.clip(adv_image, 0, 255).astype(np.uint8)
    return adv_image

def main():
    import os

    images = [f for f in os.listdir(IMAGE_PATH) if f.endswith(".jpg")]
    img_path = os.path.join(IMAGE_PATH, images[0])

    image = cv2.imread(img_path)
    adv_image = add_noise(image)

    cv2.imwrite("outputs/deployment/adversarial_sample.jpg", adv_image)

    print("Adversarial example generated")

if __name__ == "__main__":
    main()