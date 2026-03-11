# Autoplay Dino Game

Python automation bot that captures the Chrome Dino game, detects obstacles via simple contour checks on screen crops, and presses space to jump.

## Requirements
- Python 3.8+
- Packages: `opencv-python`, `numpy`, `pyautogui`, `pynput`

Install deps:
```bash
python -m pip install opencv-python numpy pyautogui pynput
```

## How it works
- Takes full-screen screenshots with `pyautogui`.
- Crops to the game area (`STATUS_IMAGE_X/Y`).
- Extracts obstacle and high-jump regions via fixed bounding boxes.
- Uses Canny edges + contour area to decide if something is present; presses space when detected.
- Shows a preview window with bounding boxes and status values.

## Usage
1) Open the Chrome Dino game and position it so the game fits within the configured crop.
2) Update `STATUS_IMAGE_X/Y` and bbox constants in `AutoplayDinoGame.py` if your screen layout differs.
3) Run:
```bash
python AutoplayDinoGame.py
```
4) The script waits ~5s, presses space to start the game, and then autoplays. Press `Esc` in the preview window to quit.

## Key tuning points
- `STATUS_IMAGE_X/Y`: the crop of the game area. Align this to your browser location.
- `OBSTACLE_BOUNDING_*`, `HIGHEST_JUMP_BOUNDING_*`: boxes where ground/bird obstacles are detected. Shift/resize to fit your view.
- Detection sensitivity: `check_contours_of_expected_area` uses contour area (`area_thresh`) if you want to ignore small noise.

## Notes
- Keep Chrome zoom at 100% for consistent bounding boxes.
- If detection lags, reduce crop size or lower preview resolution.
- Run in a well-lit, uncluttered screen for best results.

