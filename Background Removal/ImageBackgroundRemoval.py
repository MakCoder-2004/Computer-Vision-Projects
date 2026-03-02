import cv2
import numpy as np
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==============================
# Configuration
# ==============================
MODEL_PATH = "model/selfie_segmenter.tflite"
INPUT_IMAGE_PATH = "images/image.jpg"
OUTPUT_IMAGE_PATH = "selfie_segmentation_output.png"

BLUR_BACKGROUND = True  # Set to False for solid color background
BG_COLOR = (192, 192, 192)  # Gray background

# ==============================
# Load MediaPipe Segmenter
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

# ==============================
# Run Segmentation
# ==============================
def segment_image(segmenter, image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    result = segmenter.segment(mp_image)
    mask = result.category_mask.numpy_view().astype(np.float32)

    # Normalize mask
    mask = mask / 255.0

    # Smooth mask edges
    mask = cv2.GaussianBlur(mask, (7, 7), 0)

    return mask

# ==============================
# Apply Segmentation Effect
# ==============================
def apply_effect(image, mask):

    # Invert mask so foreground = person
    mask = 1.0 - mask

    # Expand mask to 3 channels
    mask_3ch = np.stack((mask,) * 3, axis=-1)

    if BLUR_BACKGROUND:
        # Blur entire image
        blurred = cv2.GaussianBlur(image, (55, 55), 0)
        # Keep person sharp, blur background
        output = image * mask_3ch + blurred * (1 - mask_3ch)
    else:
        # Solid background
        bg = np.full(image.shape, BG_COLOR, dtype=np.uint8)
        output = image * mask_3ch + bg * (1 - mask_3ch)

    return output.astype(np.uint8)

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