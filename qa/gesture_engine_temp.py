# gesture_engine_fixed_v4.py
"""
PalmCtrl — Gesture Engine (Fixed v4)
- Accurate forward-pointing detection (PIP->TIP forward vector + angle checks)
- Correct mirror handling: separate display coords vs screen coords
- Performance tweak: set image.flags.writeable False before mediapipe.process
- Return both cursor_norm_screen and cursor_norm_display so Dev B chooses mapping
- Demo toggles: m = toggle preview mirror, s = toggle map_to_screen (which cursor_norm_screen updates)
"""

from __future__ import annotations
import time, math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp

try:
    import pyautogui
    PYAUTO_OK = True
except Exception:
    PYAUTO_OK = False


@dataclass
class GestureConfig:
    # MediaPipe / detection
    detection_confidence: float = 0.5
    tracking_confidence: float = 0.5
    max_hands: int = 1
    model_complexity: int = 1  # set 1 for better detection accuracy

    # preview mirroring (True if the preview shown to user is mirrored like selfie preview)
    preview_mirror: bool = False

    # whether the engine should produce screen-space coords for mapping to actual screen (non-mirrored)
    map_to_screen: bool = True

    # Finger geometry thresholds
    angle_extended_deg: float = 150.0
    angle_folded_deg: float = 140.0

    # forward pointing thresholds (PIP->TIP)
    forward_dot_thresh: float = 0.50
    lateral_ratio_thresh: float = 0.75

    # pinch threshold for HOLD (normalized)
    hold_pinch_thresh: float = 0.22

    # smoothing
    smooth_window_min: int = 2
    smooth_window_max: int = 5
    motion_threshold: float = 0.04

    # debounce frames
    min_stable_frames: int = 1

    # scrolling
    scroll_enabled: bool = True
    scroll_step_norm: float = 0.015
    scroll_lines_per_step: int = 2
    scroll_invert: bool = False

    # platform / demo
    platform: str = "laptop"  # 'laptop' or 'android'
    draw_debug: bool = True
    enable_mouse_demo: bool = False


# Utilities
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


def _angle(a, b, c) -> float:
    bax, bay, baz = a[0]-b[0], a[1]-b[1], a[2]-b[2]
    bcx, bcy, bcz = c[0]-b[0], c[1]-b[1], c[2]-b[2]
    dot = bax*bcx + bay*bcy + baz*bcz
    mag1 = math.sqrt(bax*bax + bay*bay + baz*baz)
    mag2 = math.sqrt(bcx*bcx + bcy*bcy + bcz*bcz)
    if mag1 == 0 or mag2 == 0:
        return 0.0
    cosang = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cosang))


def _dist(a, b) -> float:
    return math.sqrt(sum((a[i] - b[i])**2 for i in range(3)))


def _lm_array(hand_landmarks) -> List[Tuple[float, float, float]]:
    return [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]


class GestureEngine:
    def __init__(self, cfg: GestureConfig | None = None):
        self.cfg = cfg or GestureConfig()
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.cfg.max_hands,
            model_complexity=self.cfg.model_complexity,
            min_detection_confidence=self.cfg.detection_confidence,
            min_tracking_confidence=self.cfg.tracking_confidence,
        )

        self.state = "IDLE"
        self.prev_state = "IDLE"
        self._pending_state = None
        self._pending_count = 0

        self.cursor_smooth: Deque[Tuple[float, float]] = deque(maxlen=self.cfg.smooth_window_max)
        self.prev_cursor_display: Optional[Tuple[float, float]] = None
        self.prev_cursor_screen: Optional[Tuple[float, float]] = None

        self.frame_time = time.time()

        self._hold_anchor_y = None
        self._scroll_accum = 0.0

    # helper: finger angles using MCP-PIP-TIP
    def _finger_angles(self, lms):
        idx = _angle(lms[5], lms[6], lms[8])
        mid = _angle(lms[9], lms[10], lms[12])
        ring = _angle(lms[13], lms[14], lms[16])
        pinky = _angle(lms[17], lms[18], lms[20])
        return {"index": idx, "middle": mid, "ring": ring, "pinky": pinky}

    def _is_index_pointing(self, lms) -> Tuple[bool, float]:
        angles = self._finger_angles(lms)
        index_straight = angles["index"] >= self.cfg.angle_extended_deg
        others_folded = (
            angles["middle"] <= self.cfg.angle_folded_deg and
            angles["ring"] <= self.cfg.angle_folded_deg and
            angles["pinky"] <= self.cfg.angle_folded_deg
        )
        # PIP -> TIP vector
        vx = lms[8][0] - lms[6][0]
        vy = lms[8][1] - lms[6][1]
        vz = lms[8][2] - lms[6][2]
        norm = math.sqrt(vx*vx + vy*vy + vz*vz) + 1e-9
        forward_score = -vz / norm  # higher -> more toward camera
        lateral_ratio = abs(vx) / norm
        forward_ok = (forward_score >= self.cfg.forward_dot_thresh and lateral_ratio <= self.cfg.lateral_ratio_thresh)
        # optional additional z-diff
        zdiff_ok = (lms[8][2] - lms[6][2]) <= 0.0  # tip not farther than pip
        is_pointing = bool(index_straight and others_folded and forward_ok and zdiff_ok)
        return is_pointing, float(forward_score)

    def _hold_gesture(self, lms) -> bool:
        pointing, _ = self._is_index_pointing(lms)
        if not pointing:
            return False
        thumb = lms[4]
        mid = lms[12]
        size = max(1e-6, _dist(lms[0], lms[9]))
        pinch = _dist(thumb, mid) / size
        return pinch <= self.cfg.hold_pinch_thresh

    def process(self, frame_bgr) -> Dict:
        h, w = frame_bgr.shape[:2]
        # For performance: mark as not writeable before passing to MediaPipe
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False
        result = self.hands.process(img_rgb)
        img_rgb.flags.writeable = True

        cursor_display = None
        cursor_screen = None
        new_state = "IDLE"
        scroll_amount = 0
        forward_score_val = None

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            lms = _lm_array(hand)

            # Determine state: HOLD first, then pointer
            if self._hold_gesture(lms):
                new_state = "HOLD"
            else:
                pointing, forward_score = self._is_index_pointing(lms)
                forward_score_val = forward_score
                if pointing:
                    new_state = "POINTER"
                    ix, iy, _ = lms[8]
                    # cursor in image coords:
                    # screen coord (non-mirrored) = (ix, iy)
                    # display coord = mirrored if preview_mirror True
                    cursor_screen = (ix, iy)
                    if self.cfg.preview_mirror:
                        cursor_display = (1.0 - ix, iy)
                    else:
                        cursor_display = (ix, iy)

                    # adaptive smoothing using display coords (because we draw on preview)
                    # choose which cursor we smooth for display; smoothing affects both though
                    smoothing_src = cursor_display
                    if self.prev_cursor_display is not None:
                        motion = math.hypot(smoothing_src[0] - self.prev_cursor_display[0],
                                            smoothing_src[1] - self.prev_cursor_display[1])
                    else:
                        motion = 0.0

                    desired_len = self.cfg.smooth_window_min if motion > self.cfg.motion_threshold else self.cfg.smooth_window_max
                    old = list(self.cursor_smooth)
                    self.cursor_smooth = deque(old[-desired_len:], maxlen=desired_len)
                    self.cursor_smooth.append(smoothing_src)
                    sx = sum([p[0] for p in self.cursor_smooth]) / len(self.cursor_smooth)
                    sy = sum([p[1] for p in self.cursor_smooth]) / len(self.cursor_smooth)
                    cursor_display = (sx, sy)

                    # derive screen cursor from display cursor properly:
                    if self.cfg.preview_mirror:
                        # display = 1 - screen => screen = 1 - display
                        cursor_screen = (1.0 - cursor_display[0], cursor_display[1])
                    else:
                        cursor_screen = cursor_display

                    self.prev_cursor_display = cursor_display
                    self.prev_cursor_screen = cursor_screen
                else:
                    # not pointing
                    self.cursor_smooth.clear()
                    self.prev_cursor_display = None
                    self.prev_cursor_screen = None

            # HOLD scroll logic uses screen-space movement (so it controls actual content)
            if self.cfg.scroll_enabled and new_state == "HOLD" and self.prev_cursor_screen is not None:
                if self._hold_anchor_y is None:
                    self._hold_anchor_y = self.prev_cursor_screen[1]
                    self._scroll_accum = 0.0
                # screen coords: y increases downwards; positive dy => hand moved down
                dy = self.prev_cursor_screen[1] - self._hold_anchor_y
                if self.cfg.scroll_invert:
                    dy *= -1
                self._scroll_accum += dy
                step = self.cfg.scroll_step_norm
                while abs(self._scroll_accum) >= step:
                    scroll_amount += self.cfg.scroll_lines_per_step * (1 if self._scroll_accum > 0 else -1)
                    self._scroll_accum -= step * (1 if self._scroll_accum > 0 else -1)
                self._hold_anchor_y = self.prev_cursor_screen[1]
            else:
                self._hold_anchor_y = None
                self._scroll_accum = 0.0

            # draw landmarks and display cursor on the frame we'll show to the user
            if self.cfg.draw_debug:
                mp_draw.draw_landmarks(
                    frame_bgr,
                    hand,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )
                if cursor_display is not None:
                    cx, cy = int(cursor_display[0] * w), int(cursor_display[1] * h)
                    cv2.circle(frame_bgr, (cx, cy), 8, (0, 255, 0), -1)

        # commit state with min_stable_frames debounce
        if new_state == self.state:
            self._pending_state = None
            self._pending_count = 0
        else:
            if self._pending_state == new_state:
                self._pending_count += 1
            else:
                self._pending_state = new_state
                self._pending_count = 1
            if self._pending_count >= self.cfg.min_stable_frames:
                self.prev_state = self.state
                self.state = new_state
                self._pending_state = None
                self._pending_count = 0

        events = []
        if self.prev_state != self.state:
            if self.prev_state != "HOLD" and self.state == "HOLD":
                events.append("HOLD_START")
            elif self.prev_state == "HOLD" and self.state != "HOLD":
                events.append("HOLD_END")

        # fps
        now = time.time()
        fps = 1.0 / (now - self.frame_time) if now > self.frame_time else 0.0
        self.frame_time = now

        # optional laptop demo: move OS cursor using screen coords if requested
        if (self.cfg.platform == "laptop" and self.cfg.enable_mouse_demo and PYAUTO_OK and self.prev_cursor_screen is not None):
            try:
                sw, sh = pyautogui.size()
                px = max(0.0, min(1.0, self.prev_cursor_screen[0]))
                py = max(0.0, min(1.0, self.prev_cursor_screen[1]))
                pyautogui.moveTo(px * sw, py * sh)
                if scroll_amount != 0:
                    pyautogui.scroll(-scroll_amount)
            except Exception:
                pass

        if self.cfg.draw_debug:
            hud = f"State:{self.state} FPS:{fps:.1f} Plat:{self.cfg.platform} PreviewMirror:{self.cfg.preview_mirror}"
            cv2.putText(frame_bgr, hud, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            if forward_score_val is not None:
                cv2.putText(frame_bgr, f"forward:{forward_score_val:.2f}", (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,0), 1)

        # Important: if preview_mirror is True, flip the frame we show to the user (so preview matches common selfie preview)
        display_frame = frame_bgr
        if self.cfg.preview_mirror:
            display_frame = cv2.flip(frame_bgr, 1)

        return {
            "state": self.state,
            "cursor_norm_screen": self.prev_cursor_screen,    # use for system actions
            "cursor_norm_display": self.prev_cursor_display,  # use for overlay drawing on preview
            "events": events,
            "scroll": scroll_amount,
            "fps": fps,
            "frame": display_frame,
        }


# Demo control
def demo(cam_index=0):
    cfg = GestureConfig()
    engine = GestureEngine(cfg)
    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FPS, 30); cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("Demo: ESC/q quit | m toggle preview mirror | s toggle map_to_screen | p toggle platform | c calibrate")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            out = engine.process(frame)
            win = "PalmCtrl v4"
            cv2.imshow(win, out["frame"])
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            elif key == ord('m'):
                cfg.preview_mirror = not cfg.preview_mirror
                print("preview_mirror:", cfg.preview_mirror)
            elif key == ord('s'):
                cfg.map_to_screen = not cfg.map_to_screen
                print("map_to_screen (engine still returns both coords):", cfg.map_to_screen)
            elif key == ord('p'):
                cfg.platform = 'android' if cfg.platform == 'laptop' else 'laptop'
                print("platform:", cfg.platform)
            elif key == ord('c'):
                calibrate_forward(engine)
    finally:
        cap.release(); cv2.destroyAllWindows()


def calibrate_forward(engine: GestureEngine, duration: float = 2.0):
    print("Calibration: point at camera for 2s...")
    cap = cv2.VideoCapture(0)
    t0 = time.time()
    vals = []
    try:
        while time.time() - t0 < duration:
            ok, frame = cap.read()
            if not ok:
                break
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_rgb.flags.writeable = False
            res = engine.hands.process(img_rgb)
            img_rgb.flags.writeable = True
            if res.multi_hand_landmarks:
                lms = _lm_array(res.multi_hand_landmarks[0])
                vx = lms[8][0] - lms[6][0]
                vz = lms[8][2] - lms[6][2]
                norm = math.sqrt(vx*vx + (lms[8][1]-lms[6][1])**2 + vz*vz) + 1e-9
                forward = -vz / norm
                vals.append(forward)
            cv2.waitKey(1)
    finally:
        cap.release()
    if vals:
        avg = sum(vals)/len(vals)
        print(f"Calib avg forward score: {avg:.3f}. If this < forward_dot_thresh, lower the threshold.")
    else:
        print("No hand detected during calibration.")


if __name__ == "__main__":
    demo()
