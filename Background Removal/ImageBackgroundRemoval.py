import cv2
import numpy as np
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==============================
# Configuration
# ==============================

MODEL_PATH = "./selfie_segmentation.tflite"
INPUT_IMAGE_PATH = "image.jpg"
OUTPUT_IMAGE_PATH = "selfie_segmentation_output.png"

BG_COLOR = (192, 192, 192)  # Gray background
MASK_COLOR = (255, 255, 255)  # White foreground
THRESHOLD = 0.2  # Segmentation confidence threshold
BLUR_BACKGROUND = True  # Set to False for solid color background

# ==============================
# Utility Functions
# ==============================

def load_segmenter(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.ImageSegmenterOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        output_category_mask=True
    )

    return vision.ImageSegmenter.create_from_options(options)


def segment_image(segmenter, image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = vision.Image(
        image_format=vision.ImageFormat.SRGB,
        data=rgb_image
    )

    result = segmenter.segment(mp_image)
    mask = result.category_mask.numpy_view().astype(np.float32)

    # Normalize mask
    mask = mask / 255.0

    # Smooth mask to improve edges
    mask = cv2.GaussianBlur(mask, (7, 7), 0)

    return mask


def apply_effect(image, mask):
    condition = np.stack((mask,) * 3, axis=-1) > THRESHOLD

    if BLUR_BACKGROUND:
        blurred_bg = cv2.GaussianBlur(image, (55, 55), 0)
        output = np.where(condition, image, blurred_bg)
    else:
        fg = np.zeros_like(image)
        fg[:] = MASK_COLOR

        bg = np.zeros_like(image)
        bg[:] = BG_COLOR

        output = np.where(condition, fg, bg)

    return output


# ==============================
# Main Execution
# ==============================

def main():
    if not os.path.exists(INPUT_IMAGE_PATH):
        print(f"❌ Input image not found: {INPUT_IMAGE_PATH}")
        return

    print("🔄 Loading model...")
    segmenter = load_segmenter(MODEL_PATH)

    print("📷 Reading image...")
    image = cv2.imread(INPUT_IMAGE_PATH)

    print("🧠 Running segmentation...")
    mask = segment_image(segmenter, image)

    print("🎨 Applying background effect...")
    output = apply_effect(image, mask)

    cv2.imwrite(OUTPUT_IMAGE_PATH, output)
    print(f"✅ Output saved to {OUTPUT_IMAGE_PATH}")

    segmenter.close()


if __name__ == "__main__":
    main()