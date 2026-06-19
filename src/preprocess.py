import cv2
import numpy as np
from PIL import Image
from pathlib import Path


class ImagePreprocessor:
    def load_image(self, path: str) -> np.ndarray:
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Could not load image: {path}")
        return img

    def enhance_brightness(self, img: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def denoise(self, img: np.ndarray) -> np.ndarray:
        return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    def sharpen(self, img: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(img, (0, 0), 3.0)
        return cv2.addWeighted(img, 1.5, blurred, -0.5, 0)

    def normalize(self, img: np.ndarray) -> Image.Image:
        resized = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def preprocess(self, path: str):
        original = self.load_image(path)
        processed = self.enhance_brightness(original)
        processed = self.denoise(processed)
        processed = self.sharpen(processed)
        pil_image = self.normalize(processed)
        return original, pil_image


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os

    test_dir = Path(__file__).resolve().parent.parent / "data" / "test_images"
    if not test_dir.exists():
        test_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created {test_dir}. Place test images there and re-run.")
        exit(0)

    images = list(test_dir.glob("*.*"))
    if not images:
        print(f"No images found in {test_dir}. Add some test images first.")
        exit(0)

    preprocessor = ImagePreprocessor()
    n = min(len(images), 3)
    fig, axes = plt.subplots(n, 2, figsize=(10, 5 * n))
    if n == 1:
        axes = [axes]

    for i, img_path in enumerate(images[:n]):
        original, pil_processed = preprocessor.preprocess(str(img_path))
        axes[i][0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[i][0].set_title(f"Before — {img_path.name}")
        axes[i][0].axis("off")
        axes[i][1].imshow(np.array(pil_processed))
        axes[i][1].set_title(f"After — {img_path.name}")
        axes[i][1].axis("off")

    plt.tight_layout()
    plt.show()
