import sys
import os


import pygame
import random
import cv2
import threading
import time
import mediapipe as mp
from HandLandmark_mediapipe_utils import load_detection_model, visualize_detection

# ── Resource path helper ───────────────────────────────────────────────────────
# Works both during normal development AND inside a PyInstaller bundle.
# PyInstaller extracts everything to sys._MEIPASS; in dev we use the script dir.
def resource_path(*parts):
    base = getattr(sys, "_MEIPASS",
                   os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, *parts)
# ──────────────────────────────────────────────────────────────────────────────


pygame.mixer.pre_init(44100, -16, 1, 512)
pygame.init()

# ── Window / layout ────────────────────────────────────────────────────────────
SCREEN_HEIGHT = 600
GAME_WIDTH    = 1100
CAM_W, CAM_H  = 320, 240
PANEL_W       = CAM_W + 20           # 340 px right panel
SCREEN_WIDTH  = GAME_WIDTH + PANEL_W # 1440

SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Dino Game – Hand Control")

# ── Assets ─────────────────────────────────────────────────────────────────────
RUNNING      = [pygame.image.load(resource_path("Assets", "Dino", "DinoRun1.png")),
                pygame.image.load(resource_path("Assets", "Dino", "DinoRun2.png"))]
JUMPING      = pygame.image.load(resource_path("Assets", "Dino", "DinoJump.png"))
DUCKING      = [pygame.image.load(resource_path("Assets", "Dino", "DinoDuck1.png")),
                pygame.image.load(resource_path("Assets", "Dino", "DinoDuck2.png"))]
DEAD_IMG     = pygame.image.load(resource_path("Assets", "Dino", "DinoDead.png"))

SMALL_CACTUS = [pygame.image.load(resource_path("Assets", "Cactus", "SmallCactus1.png")),
                pygame.image.load(resource_path("Assets", "Cactus", "SmallCactus2.png")),
                pygame.image.load(resource_path("Assets", "Cactus", "SmallCactus3.png"))]
LARGE_CACTUS = [pygame.image.load(resource_path("Assets", "Cactus", "LargeCactus1.png")),
                pygame.image.load(resource_path("Assets", "Cactus", "LargeCactus2.png")),
                pygame.image.load(resource_path("Assets", "Cactus", "LargeCactus3.png"))]
BIRD         = [pygame.image.load(resource_path("Assets", "Bird", "Bird1.png")),
                pygame.image.load(resource_path("Assets", "Bird", "Bird2.png"))]

CLOUD        = pygame.image.load(resource_path("Assets", "Other", "Cloud.png"))
BG           = pygame.image.load(resource_path("Assets", "Other", "Track.png"))
GAME_OVER    = pygame.image.load(resource_path("Assets", "Other", "GameOver.png"))
RESET_BTN    = pygame.image.load(resource_path("Assets", "Other", "Reset.png"))

# ── Sounds ─────────────────────────────────────────────────────────────────────
class _SilentSound:
    """No-op stub used when the audio device is unavailable."""
    def play(self): pass
    def set_volume(self, v): pass

try:
    if not pygame.mixer.get_init():
        pygame.mixer.init()
    JUMP_SOUND      = pygame.mixer.Sound(resource_path("Assets", "Sounds", "jump.wav"))
    JUMP_SOUND.set_volume(0.4)
    DIE_SOUND       = pygame.mixer.Sound(resource_path("Assets", "Sounds", "die.wav"))
    DIE_SOUND.set_volume(0.5)
    MILESTONE_SOUND = pygame.mixer.Sound(resource_path("Assets", "Sounds", "milestone.wav"))
    MILESTONE_SOUND.set_volume(0.4)
    print("[INFO] Audio initialised.")
except Exception as e:
    print(f"[WARNING] Audio unavailable ({e}) – sounds disabled.")
    JUMP_SOUND      = _SilentSound()
    DIE_SOUND       = _SilentSound()
    MILESTONE_SOUND = _SilentSound()

# ── Shared fonts ───────────────────────────────────────────────────────────────
_PIXEL_FONT    = resource_path("Assets", "Font", "PressStart2P-Regular.ttf")
SCORE_FONT     = pygame.font.Font(_PIXEL_FONT, 14)   # score / HI display
START_FONT     = pygame.font.Font(_PIXEL_FONT, 11)   # start / game-over text
HINT_FONT      = pygame.font.Font(_PIXEL_FONT,  8)   # sub-hints
PANEL_FONT     = pygame.font.SysFont("freesansbold", 14)
BADGE_FONT     = pygame.font.SysFont("freesansbold", 13)

# ── Physics / speed ───────────────────────────────────────────────────────────
FPS           = 60
GROUND_Y      = 310     # dino run Y
GROUND_Y_DUCK = 340     # dino duck Y
JUMP_VEL_INIT = 8.5
JUMP_GRAVITY  = 0.4     # vel decrease per frame  (was 0.8 @ 30fps)
JUMP_MOVE     = 2       # px per vel unit           (was 4   @ 30fps)
INIT_SPEED    = 10      # px/frame                  (was 20  @ 30fps)
SPEED_INC     = 0.5     # speed boost every 100 pts

# ── Chrome palette ─────────────────────────────────────────────────────────────
BG_COLOUR    = (247, 247, 247)
SCORE_COLOUR = ( 83,  83,  83)
HI_COLOUR    = (175, 175, 175)

SPLASH_MIN_TIME = 1.0  # seconds – avoid instant flash on very fast machines


# ==============================================================================
# Camera / gesture  (background daemon thread)
# ==============================================================================
gesture_lock  = threading.Lock()
gesture_state = {"jump": False, "crouch": False}

result_lock    = threading.Lock()
_latest_result = None
_last_ts       = -1

camera_running      = True
hand_control_active = False

cam_frame_lock  = threading.Lock()
cam_frame_bytes = None      # (bytes, w, h)  or  None


def _is_closed_fist(lm):
    return all(lm[tip].y > lm[4].y for tip in (8, 12, 16, 20))


def _detection_callback(result, _img, timestamp_ms):
    global _latest_result, _last_ts
    with result_lock:
        if timestamp_ms > _last_ts:
            _latest_result = result
            _last_ts = timestamp_ms


def _process_gestures(result):
    jmp = crch = False
    if result and result.hand_landmarks and result.handedness:
        for i, lm in enumerate(result.hand_landmarks):
            label = result.handedness[i][0].category_name
            closed = _is_closed_fist(lm)
            if label == "Left":  jmp  = closed
            if label == "Right": crch = closed
    with gesture_lock:
        gesture_state["jump"]   = jmp
        gesture_state["crouch"] = crch


def _try_open_camera():
    backends = ([cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
                if os.name == "nt" else [cv2.CAP_V4L2, cv2.CAP_ANY])
    for b in backends:
        for i in range(0, 4):
            cap = cv2.VideoCapture(i, b)
            if cap.isOpened():
                print(f"[INFO] Webcam opened: index={i}, backend={b}")
                return cap
            cap.release()
    return None


def _camera_thread():
    global camera_running, hand_control_active, cam_frame_bytes
    landmarker = load_detection_model(resource_path("model", "hand_landmarker.task"),
                                      callback=_detection_callback)
    cap = _try_open_camera()
    if cap is None:
        print("[WARNING] No webcam found – keyboard-only mode.")
        camera_running = False
        landmarker.close()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    hand_control_active = True
    prev_t = time.time()

    while camera_running:
        ok, frame = cap.read()
        if not ok:
            break

        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (256, 256))
        ts = int(time.time() * 1000)
        landmarker.detect_async(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=small), ts)

        with result_lock:
            cur = _latest_result
        if cur:
            frame = visualize_detection(frame, cur)
            _process_gestures(cur)

        with gesture_lock:
            jmp  = gesture_state["jump"]
            crch = gesture_state["crouch"]
        labels = (["JUMP (L)"] if jmp else []) + (["DUCK (R)"] if crch else [])
        if labels:
            cv2.putText(frame, " | ".join(labels), (8, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)

        now = time.time()
        fps_cam = int(1 / (now - prev_t)) if (now - prev_t) > 0 else 0
        prev_t = now
        cv2.putText(frame, f"FPS:{fps_cam}", (8, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)

        pip = cv2.resize(frame, (CAM_W, CAM_H))
        with cam_frame_lock:
            cam_frame_bytes = (cv2.cvtColor(pip, cv2.COLOR_BGR2RGB).tobytes(),
                               CAM_W, CAM_H)

    cap.release()
    landmarker.close()


def start_camera_thread():
    """Start the background camera / hand-tracking thread once.

    Called from the splash-screen bootstrap so the window appears
    immediately, then heavy MediaPipe / webcam init happens in
    the background.
    """
    t = threading.Thread(target=_camera_thread, daemon=True)
    t.start()
    return t


# ==============================================================================
# Camera panel  (drawn on every screen)
# ==============================================================================
def draw_camera_panel():
    px = GAME_WIDTH
    pygame.draw.rect(SCREEN, (30, 30, 30), (px, 0, PANEL_W, SCREEN_HEIGHT))
    pygame.draw.line(SCREEN, (80, 80, 80), (px, 0), (px, SCREEN_HEIGHT), 2)

    if hand_control_active:
        t = PANEL_FONT.render("HAND CONTROL", True, (220, 220, 220))
        SCREEN.blit(t, t.get_rect(centerx=px + PANEL_W // 2, centery=22))

        with cam_frame_lock:
            fd = cam_frame_bytes
        if fd:
            surf = pygame.image.frombuffer(fd[0], (fd[1], fd[2]), "RGB")
            cx = px + (PANEL_W - CAM_W) // 2
            SCREEN.blit(surf, (cx, 40))
            pygame.draw.rect(SCREEN, (80, 80, 80),
                             (cx - 2, 38, CAM_W + 4, CAM_H + 4), 2)

        with gesture_lock:
            jmp  = gesture_state["jump"]
            crch = gesture_state["crouch"]
        by = 40 + CAM_H + 14
        for lbl, act in [("LEFT FIST = JUMP", jmp), ("RIGHT FIST = DUCK", crch)]:
            br = pygame.Rect(px + 10, by, PANEL_W - 20, 26)
            pygame.draw.rect(SCREEN, (0, 180, 0) if act else (60, 60, 60),
                             br, border_radius=5)
            ts = BADGE_FONT.render(lbl, True, (255, 255, 255))
            SCREEN.blit(ts, ts.get_rect(center=br.center))
            by += 34
        h = PANEL_FONT.render("ESC = Quit", True, (120, 120, 120))
        SCREEN.blit(h, h.get_rect(centerx=px + PANEL_W // 2,
                                   centery=SCREEN_HEIGHT - 14))
    else:
        y = 40
        for line, col in [
            ("HAND CONTROL",     (180,  60,  60)),
            ("DISABLED",         (180,  60,  60)),
            ("No webcam found.", (180, 180, 180)),
            ("",                  None),
            ("Use keyboard:",    (180, 180, 180)),
            ("Up Arrow / SPACE = Jump", (180, 180, 180)),
            ("Down Arrow =  Duck",       (180, 180, 180)),
            ("",                  None),
            ("ESC = Quit",       (120, 120, 120)),
        ]:
            if col:
                s = PANEL_FONT.render(line, True, col)
                SCREEN.blit(s, s.get_rect(centerx=px + PANEL_W // 2, centery=y))
            y += 22


# ==============================================================================
# HUD helpers
# ==============================================================================
def draw_score(points, high_score, blink=False):
    if blink:
        return
    sc = SCORE_FONT.render(f"{points:05d}",        True, SCORE_COLOUR)
    hi = SCORE_FONT.render(f"HI {high_score:05d}", True, HI_COLOUR)
    SCREEN.blit(hi, (GAME_WIDTH - 250, 20))
    SCREEN.blit(sc, (GAME_WIDTH - 115, 20))


def draw_bg(g, scroll=True):
    iw = BG.get_width()
    x  = g["x_pos_bg"]
    SCREEN.blit(BG, (x,      380))
    SCREEN.blit(BG, (x + iw, 380))
    if scroll:
        x -= g["game_speed"]
        if x <= -iw:
            x = 0
        g["x_pos_bg"] = x


# ==============================================================================
# Game objects
# ==============================================================================
class Dinosaur:
    X_POS = 80

    def __init__(self):
        self.dino_duck = False
        self.dino_run  = True
        self.dino_jump = False
        self.is_dead   = False
        self.step_index = 0
        self.jump_vel   = JUMP_VEL_INIT
        self.image      = RUNNING[0]
        self.dino_rect  = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = GROUND_Y
        self._prev_jump = False

    def update(self, keys):
        if self.is_dead:
            return
        if self.dino_duck:  self._duck()
        if self.dino_run:   self._run()
        if self.dino_jump:  self._do_jump()
        if self.step_index >= 20:
            self.step_index = 0

        with gesture_lock:
            jump_g   = gesture_state["jump"]
            crouch_g = gesture_state["crouch"]

        jump_trigger  = jump_g and not self._prev_jump
        self._prev_jump = jump_g

        up   = keys[pygame.K_UP]   or keys[pygame.K_SPACE]
        down = keys[pygame.K_DOWN]

        if (up or jump_trigger) and not self.dino_jump:
            self.dino_duck = False
            self.dino_run  = False
            self.dino_jump = True
            JUMP_SOUND.play()
        elif (down or crouch_g) and not self.dino_jump:
            self.dino_duck = True
            self.dino_run  = False
            self.dino_jump = False
        elif not (self.dino_jump or down or crouch_g):
            self.dino_duck = False
            self.dino_run  = True
            self.dino_jump = False

    def die(self):
        self.is_dead = True
        self.image   = DEAD_IMG
        self.dino_rect = DEAD_IMG.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = GROUND_Y

    def _run(self):
        self.image = RUNNING[self.step_index // 10]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = GROUND_Y
        self.step_index += 1

    def _duck(self):
        self.image = DUCKING[self.step_index // 10]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = GROUND_Y_DUCK
        self.step_index += 1

    def _do_jump(self):
        self.image = JUMPING
        self.dino_rect.y -= self.jump_vel * JUMP_MOVE
        self.jump_vel    -= JUMP_GRAVITY
        if self.jump_vel < -JUMP_VEL_INIT:
            self.dino_jump   = False
            self.jump_vel    = JUMP_VEL_INIT
            self.dino_rect.y = GROUND_Y   # snap to ground

    def draw(self, surf):
        surf.blit(self.image, (self.dino_rect.x, self.dino_rect.y))


class Cloud:
    def __init__(self, x=None):
        self.x = x if x is not None else GAME_WIDTH + random.randint(200, 700)
        self.y = random.randint(50, 110)

    def update(self, speed):
        self.x -= speed * 0.45   # parallax
        if self.x < -CLOUD.get_width():
            self.x = GAME_WIDTH + random.randint(300, 800)
            self.y = random.randint(50, 110)

    def draw(self, surf):
        surf.blit(CLOUD, (self.x, self.y))


class Obstacle:
    def __init__(self, image, otype):
        self.image = image
        self.otype = otype
        self.rect  = self.image[self.otype].get_rect()
        self.rect.x = GAME_WIDTH

    def update(self, speed):
        self.rect.x -= speed
        return self.rect.x < -self.rect.width   # True → remove

    def draw(self, surf):
        surf.blit(self.image[self.otype], self.rect)


class SmallCactus(Obstacle):
    def __init__(self):
        super().__init__(SMALL_CACTUS, random.randint(0, 2))
        self.rect.y = 325


class LargeCactus(Obstacle):
    def __init__(self):
        super().__init__(LARGE_CACTUS, random.randint(0, 2))
        self.rect.y = 300


class Bird(Obstacle):
    # Two unambiguous heights — no mid-zone that sits inside the jump arc:
    #   y=180 → bird bottom ≈ 260, dino top when standing = 310  → run under
    #   y=325 → dino bottom at jump apex ≈ 213 < 325              → jump over
    HIGH = 180   # avoid by doing NOTHING (don't jump!)
    LOW  = 325   # avoid by JUMPING

    def __init__(self, force_height=None):
        super().__init__(BIRD, 0)
        self.rect.y   = force_height if force_height is not None \
                        else random.choice([self.HIGH, self.LOW])
        self.anim_idx = 0

    def draw(self, surf):
        surf.blit(self.image[self.anim_idx // 5 % 2], self.rect)
        self.anim_idx += 1


# ==============================================================================
# Main loop  —  flat state machine, zero recursion
# ==============================================================================
STATE_START = "start"
STATE_PLAY  = "play"
STATE_DEAD  = "dead"


def new_game_state():
    return {
        "player":           Dinosaur(),
        "clouds":           [
                                Cloud(x=150),
                                Cloud(x=380),
                                Cloud(x=620),
                                Cloud(x=850),
                                Cloud(x=1050),
                            ],
        "obstacles":        [],
        "game_speed":       INIT_SPEED,
        "x_pos_bg":         0,
        "points":           0,
        "score_acc":        0.0,
        "spawn_timer":      80,
        "blink_timer":      0,
        "last_was_cactus":  False,
    }


def run_game():
    global camera_running
    clock      = pygame.time.Clock()
    high_score = 0
    state      = STATE_START
    g          = new_game_state()
    prev_jump  = False          # gesture edge for start / restart


    while True:
        dt_ms = clock.get_time()   # actual ms elapsed since the previous frame

        # ── events (all screens) ──────────────────────────────────────────────
        events = pygame.event.get()
        for ev in events:
            if ev.type == pygame.QUIT:
                camera_running = False
                pygame.quit()
                return
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                camera_running = False
                pygame.quit()
                return

        keys = pygame.key.get_pressed()

        with gesture_lock:
            jump_now = gesture_state["jump"]
        jump_edge = jump_now and not prev_jump
        prev_jump = jump_now

        # ══════════════════════════════════════════════════════════════════════
        # START SCREEN
        # ══════════════════════════════════════════════════════════════════════
        if state == STATE_START:
            SCREEN.fill(BG_COLOUR)
            iw = BG.get_width()
            SCREEN.blit(BG, (0, 380)); SCREEN.blit(BG, (iw, 380))
            for cloud in g["clouds"]:
                cloud.draw(SCREEN)
            SCREEN.blit(RUNNING[0],
                        RUNNING[0].get_rect(x=Dinosaur.X_POS, y=GROUND_Y))

            t = START_FONT.render(
                "PRESS SPACE OR CLOSE LEFT FIST",
                True, SCORE_COLOUR)
            SCREEN.blit(t, t.get_rect(centerx=GAME_WIDTH // 2,
                                       centery=SCREEN_HEIGHT // 2 + 20))
            s = HINT_FONT.render(
                "LEFT FIST=JUMP  RIGHT FIST=DUCK  ESC=QUIT",
                True, HI_COLOUR)
            SCREEN.blit(s, s.get_rect(centerx=GAME_WIDTH // 2,
                                       centery=SCREEN_HEIGHT // 2 + 60))
            draw_camera_panel()
            pygame.display.update()
            clock.tick(FPS)

            if keys[pygame.K_SPACE] or keys[pygame.K_UP] or jump_edge:
                g     = new_game_state()
                state = STATE_PLAY
            continue

        # ══════════════════════════════════════════════════════════════════════
        # GAME OVER SCREEN
        # ══════════════════════════════════════════════════════════════════════
        if state == STATE_DEAD:
            SCREEN.fill(BG_COLOUR)
            draw_bg(g, scroll=False)
            for cloud in g["clouds"]:
                cloud.draw(SCREEN)
            for obs in g["obstacles"]:
                obs.draw(SCREEN)
            g["player"].draw(SCREEN)
            draw_score(g["points"], high_score)

            # ── Chrome-style game-over overlay ────────────────────────────────
            go_rect  = GAME_OVER.get_rect(centerx=GAME_WIDTH // 2,
                                           centery=SCREEN_HEIGHT // 2 - 30)
            rst_rect = RESET_BTN.get_rect(centerx=GAME_WIDTH // 2,
                                           centery=SCREEN_HEIGHT // 2 + 45)
            SCREEN.blit(GAME_OVER, go_rect)
            SCREEN.blit(RESET_BTN, rst_rect)

            draw_camera_panel()
            pygame.display.update()
            clock.tick(FPS)

            restart = jump_edge
            for ev in events:
                if ev.type == pygame.KEYDOWN and ev.key != pygame.K_ESCAPE:
                    restart = True
                if (ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1
                        and rst_rect.collidepoint(ev.pos)):
                    restart = True
            if restart:
                g     = new_game_state()
                state = STATE_PLAY
            continue

        # ══════════════════════════════════════════════════════════════════════
        # PLAYING
        # ══════════════════════════════════════════════════════════════════════
        SCREEN.fill(BG_COLOUR)

        g["player"].update(keys)

        # ── obstacle spawning ─────────────────────────────────────────────────
        if g["spawn_timer"] > 0:
            g["spawn_timer"] -= 1
        else:
            spd      = g["game_speed"]
            # Larger gap → more breathing room between obstacles
            min_gap  = max(550, int(900 - (spd - INIT_SPEED) * 10))
            last_x   = max((obs.rect.right for obs in g["obstacles"]), default=-1)

            if not g["obstacles"] or last_x < GAME_WIDTH - min_gap:
                # Birds only become common at higher speeds
                bird_w = max(1, int((spd - INIT_SPEED) // 2))
                cls = random.choices(
                    [SmallCactus, LargeCactus, Bird],
                    weights=[5, 5, bird_w])[0]

                if cls is Bird:
                    height = Bird.LOW if g["last_was_cactus"] else None
                    g["obstacles"].append(Bird(force_height=height))
                    g["last_was_cactus"] = False
                else:
                    g["obstacles"].append(cls())
                    g["last_was_cactus"] = True

                    # 6 % chance of a second SAME-TYPE cactus right behind
                    if random.random() < 0.06:
                        extra = cls()
                        extra.rect.x = GAME_WIDTH + random.randint(500, 700)
                        g["obstacles"].append(extra)

                # Longer timer → fewer obstacles overall
                base = max(30, int(100 - (spd - INIT_SPEED) * 4))
                g["spawn_timer"] = base + random.randint(0, 50)

        # ── draw + update obstacles + collision ───────────────────────────────
        hit = False
        for obs in g["obstacles"][:]:
            obs.draw(SCREEN)
            if obs.update(g["game_speed"]):
                g["obstacles"].remove(obs)
            elif g["player"].dino_rect.colliderect(obs.rect):
                hit = True

        if hit:
            g["player"].die()
            g["points"] = int(g["score_acc"])          # finalise score
            high_score  = max(high_score, g["points"])
            DIE_SOUND.play()
            # redraw one last frame with dead dino, then brief freeze
            # redraw one last frame with dead dino, then brief freeze
            draw_bg(g, scroll=False)
            for cloud in g["clouds"]:
                cloud.draw(SCREEN)
            for obs in g["obstacles"]:
                obs.draw(SCREEN)
            g["player"].draw(SCREEN)
            draw_score(g["points"], high_score)
            draw_camera_panel()
            pygame.display.update()
            pygame.time.delay(600)
            state = STATE_DEAD
            clock.tick(FPS)
            continue

        # ── background + cloud ────────────────────────────────────────────────
        draw_bg(g, scroll=True)
        for cloud in g["clouds"]:
            cloud.update(g["game_speed"])
            cloud.draw(SCREEN)
        g["player"].draw(SCREEN)

        # ── score ─────────────────────────────────────────────────────────────
        # 0.012 pts/ms ≈ 12 pts/sec at 60 fps  (original Chrome feel)
        # Using actual frame delta so speed stays correct at any frame rate.
        g["score_acc"] += dt_ms * 0.012
        new_pts = int(g["score_acc"])
        if new_pts > g["points"]:
            prev_hundred = g["points"] // 100
            g["points"]  = new_pts
            if g["points"] // 100 > prev_hundred:      # crossed a 100 milestone
                g["game_speed"] += SPEED_INC
                g["blink_timer"] = 24
                MILESTONE_SOUND.play()

        blink_off = False
        if g["blink_timer"] > 0:
            g["blink_timer"] -= 1
            blink_off = (g["blink_timer"] % 8) < 4

        draw_score(g["points"], high_score, blink=blink_off)
        draw_camera_panel()
        clock.tick(FPS)
        pygame.display.update()


# ============================================================================
# Splash / loading screen
# ============================================================================

def show_splash_and_init():
    """Show a simple loading screen while hand-tracking initialises.

    This starts the camera / MediaPipe thread, then keeps
    drawing a 'Loading…' screen until either:
      * hand_control_active becomes True (hand control ready), or
      * camera_running becomes False (no webcam – keyboard only),
    and at least SPLASH_MIN_TIME seconds have passed.
    """
    global camera_running

    # Use the first running sprite as the splash dino
    splash_img = RUNNING[0]
    clock      = pygame.time.Clock()

    # Kick off the heavy work
    start_camera_thread()

    start_t     = time.time()
    dots        = 0

    while True:
        dt = clock.tick(30)  # 30 FPS is enough for a loading screen

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                camera_running = False
                pygame.quit()
                sys.exit(0)
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                camera_running = False
                pygame.quit()
                sys.exit(0)

        SCREEN.fill(BG_COLOUR)

        # Center the dino roughly where it appears in-game
        drect = splash_img.get_rect()
        drect.midbottom = (GAME_WIDTH // 2, SCREEN_HEIGHT // 2 + 60)
        SCREEN.blit(splash_img, drect)

        # Title
        title = START_FONT.render("DINO HAND GAME", True, SCORE_COLOUR)
        SCREEN.blit(title, title.get_rect(centerx=GAME_WIDTH // 2,
                                          centery=SCREEN_HEIGHT // 2 - 40))

        # Loading text with simple animated dots
        dots = (dots + 1) % 60
        n    = dots // 20  # 0,1,2
        txt  = "Loading hand tracking" + "." * n
        t    = HINT_FONT.render(txt, True, HI_COLOUR)
        SCREEN.blit(t, t.get_rect(centerx=GAME_WIDTH // 2,
                                   centery=SCREEN_HEIGHT // 2 + 5))

        # Small hint line
        hint = HINT_FONT.render("This may take a few seconds the first time…",
                                 True, HI_COLOUR)
        SCREEN.blit(hint, hint.get_rect(centerx=GAME_WIDTH // 2,
                                        centery=SCREEN_HEIGHT // 2 + 28))

        pygame.display.update()

        # Exit conditions
        elapsed = time.time() - start_t
        if elapsed < SPLASH_MIN_TIME:
            continue

        if hand_control_active:
            # Camera + model ready – proceed
            break
        if not camera_running:
            # Camera thread gave up (no webcam) – keyboard-only mode
            break


# ============================================================================
# Main entry point
# ============================================================================
if __name__ == "__main__":
    show_splash_and_init()
    run_game()
