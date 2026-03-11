# Dino Game Controlled by Hand Gestures

A desktop clone of the Chrome Dino runner game, controlled using **hand gestures** via a webcam.

- **Left fist** → Jump  
- **Right fist** → Duck  
- Keyboard controls still work (`Space`/`↑` to jump, `↓` to duck).

This project uses **Python**, **Pygame**, **OpenCV**, and **MediaPipe** hand tracking.

---

## Project Structure

```text
Play Dino Game Using Hand/
├── model/
│   └── hand_landmarker.task         # MediaPipe hand-landmarker model
├── Python Version/
│   ├── game.py                      # Main game entry (hand-controlled dino)
│   ├── HandLandmark_mediapipe_utils.py  # MediaPipe hand-tracking helpers
│   ├── Detect Hand Geasture.py      # Standalone gesture-visualisation script
│   ├── requirements.txt             # Python dependencies
│   ├── game.spec                    # PyInstaller build spec for EXE
│   ├── build.bat                    # Windows build script for EXE
│   ├── Assets/
│   │   ├── Dino/                    # Dino sprites + EXE icon
│   │   ├── Cactus/                  # Obstacle sprites
│   │   ├── Bird/                    # Bird sprites
│   │   ├── Other/                   # Track, clouds, Game Over, Reset
│   │   ├── Font/                    # Pixel font (Press Start 2P)
│   │   └── Sounds/                  # jump / die / milestone sounds
│   └── dist/                        # (ignored) PyInstaller build output
├── .gitignore
└── README.md
```

---

## Getting Started (Development)

### 1. Prerequisites

- **Windows 10 or 11** (64‑bit)  
- Python **3.11** (the project was built and tested with this version)  
- A webcam (for hand control; keyboard works without it)

### 2. Create and activate a virtual environment (recommended)

```powershell
cd "Play Dino Game Using Hand\Python Version"
python -m venv .venv
.venv\Scripts\activate
```

If `python` is not on PATH, you can also use the Windows launcher:

```powershell
cd "Play Dino Game Using Hand\Python Version"
py -3.11 -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```powershell
cd "Play Dino Game Using Hand\Python Version"
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the game

```powershell
cd "Play Dino Game Using Hand\Python Version"
python game.py
```

When you start the game you will first see a **loading screen** with the Dino and a  
"Loading hand tracking…" message while MediaPipe and the webcam initialise.  
Then the normal start screen appears.

---

## Controls

| Action | Hand gesture         | Keyboard        |
|--------|----------------------|-----------------| 
| Jump   | Close **left** fist  | `Space` or `↑`  |
| Duck   | Close **right** fist | `↓`             |
| Quit   | —                    | `Esc`           |

Hand controls require a webcam and a reasonably lit environment.

---

## Building a Standalone EXE (Windows)

You can turn the game into a single-folder Windows application using **PyInstaller**.

### One‑click build (recommended)

From PowerShell:

```powershell
cd "Play Dino Game Using Hand\Python Version"
build.bat
```

The script will:

1. Install/upgrade dependencies (`pip`, `pygame`, `mediapipe`, `opencv-python`, `pyinstaller`, etc.).  
2. Clean previous `build/` and `dist/` folders.  
3. Run PyInstaller with `game.spec`.

After it finishes, the EXE will be at:

```text
Play Dino Game Using Hand\Python Version\dist\DinoHandGame\DinoHandGame.exe
```

Zip the **entire** `DinoHandGame` folder to share with others. They can then run  
`DinoHandGame.exe` without installing Python.

### Manual build

```powershell
cd "Play Dino Game Using Hand\Python Version"
pip install -r requirements.txt
py -3.11 -m PyInstaller game.spec --clean --noconfirm
```

---

## How Hand Tracking Works

- `HandLandmark_mediapipe_utils.py` wraps the **MediaPipe Tasks** Hand Landmarker API.
- A background thread in `game.py`:
  - Opens the webcam with OpenCV.  
  - Feeds frames into the MediaPipe hand model.  
  - Interprets hand landmarks into simple gestures:
    - Left closed fist → `jump`  
    - Right closed fist → `crouch`
- These gestures are read by the main game loop and combined with keyboard input.

If there is **no webcam**, the game automatically falls back to **keyboard‑only** mode.

---

## Repository Hygiene

- `dist/`, `build/`, and other generated files are ignored via `.gitignore`.  
- IDE metadata (`.idea/`, `.vscode/`) are not tracked.  
- Compiled Python artefacts (`__pycache__/`, `*.pyc`) are ignored.

When pushing to GitHub, you mainly need:

- `model/hand_landmarker.task`  
- `Python Version/game.py`  
- `Python Version/HandLandmark_mediapipe_utils.py`  
- `Python Version/Detect Hand Geasture.py` (optional demo)  
- `Python Version/Assets/` (images, font, sounds)  
- `Python Version/requirements.txt`  
- `Python Version/game.spec`  
- `Python Version/build.bat`  
- Root `README.md` and `.gitignore`

---

## License

Add your chosen license here (for example, MIT) if you plan to make this public on GitHub.

