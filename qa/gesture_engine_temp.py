
"""
PalmCtrl — Gesture Engine (Temp v2)

Updated gestures (per user spec):
- POINTER (cursor mode): Index finger pointing toward camera. Other fingers mostly curled.
- HOLD: While pointing, join THUMB + MIDDLE fingertips (pinch). Remains active while pinched.

Outputs per frame (dict):
{
  "state": "IDLE" | "POINTER" | "HOLD",
  "cursor_norm": (x, y) or None,       # normalized in [0,1], origin = left-top of image
  "events": [...],                      # transitions: "HOLD_START", "HOLD_END"
  "fps": float
}

Notes:
- This is a pure CV/ML module (Dev A). Mapping to Android touch/cursor is Dev B.
- Debug overlay can be toggled; use this file in the Python laptop build or port to Android (MediaPipe Hands / TFLite).

Author: PalmCtrl team (Evan, Nathal, Nael)
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp

try:
    import pyautogui  # optional, for laptop demo only
    PYAUTO_OK = True
except Exception:
    PYAUTO_OK = False


# ---------------------------- Config ---------------------------------
@dataclass
class GestureConfig:
    detection_confidence: float = 0.7
    tracking_confidence: float = 0.6
    max_hands: int = 1

    # Gesture thresholds
    angle_extended_deg: float = 160.0  # finger considered straight if PIP angle > this
    angle_folded_deg: float = 140.0    # finger considered folded if PIP angle < this
    index_toward_camera_zdiff: float = -0.03  # tip_z - pip_z must be less than this (closer to camera)

    # Thumb-middle pinch threshold for HOLD, normalized by hand size
    hold_pinch_thresh: float = 0.22

    # Smoothing
    smooth_window: int = 5             # moving average window for cursor
    min_stable_frames: int = 2         # gesture must persist this many frames to switch

    # Scroll while HOLD
    scroll_enabled: bool = True
    scroll_step_norm: float = 0.015     # normalized y delta per step
    scroll_lines_per_step: int = 2      # event amount per step (tune in Dev B)
    scroll_invert: bool = False         # set True if you want opposite direction

    # Debug / Control
    draw_debug: bool = True
    enable_mouse_demo: bool = False    # if True and pyautogui available, moves OS cursor (laptop demo)


# ---------------------------- Utils ----------------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


def _angle(a, b, c) -> float:
    """Return angle ABC in degrees."""
    import math
    bax = a[0] - b[0]
    bay = a[1] - b[1]
    baz = a[2] - b[2]
    bcx = c[0] - b[0]
    bcy = c[1] - b[1]
    bcz = c[2] - b[2]
    dot = bax * bcx + bay * bcy + baz * bcz
    mag1 = (bax*bax + bay*bay + baz*baz) ** 0.5
    mag2 = (bcx*bcx + bcy*bcy + bcz*bcz) ** 0.5
    if mag1 == 0 or mag2 == 0:
        return 0.0
    cosang = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return float(__import__("math").degrees(__import__("math").acos(cosang)))


def _dist(a, b) -> float:
    """Euclidean distance in 3D."""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return (dx*dx + dy*dy + dz*dz) ** 0.5


def _hand_size(lms) -> float:
    """Scale for normalization based on wrist to middle_mcp distance."""
    wrist = lms[0]
    middle_mcp = lms[9]
    return max(1e-6, _dist(wrist, middle_mcp))


def _lm_array(hand_landmarks) -> List[Tuple[float, float, float]]:
    return [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]


# ---------------------------- Engine ---------------------------------
class GestureEngine:
    def __init__(self, cfg: GestureConfig | None = None):
        self.cfg = cfg or GestureConfig()
        self.hands = mp_hands.Hands(
            max_num_hands=self.cfg.max_hands,
            model_complexity=1,
            min_detection_confidence=self.cfg.detection_confidence,
            min_tracking_confidence=self.cfg.tracking_confidence,
        )

        self.prev_state: str = "IDLE"
        self.state: str = "IDLE"
        self.cursor_smooth: Deque[Tuple[float, float]] = deque(maxlen=self.cfg.smooth_window)
        self.frame_time = time.time()

        # for debounce
        self._pending_state: Optional[str] = None
        self._pending_count = 0

        # for HOLD scrolling
        self._hold_anchor_y: Optional[float] = None
        self._scroll_accum: float = 0.0

    # ---- finger state helpers ----
    def _finger_angles(self, lms: List[Tuple[float, float, float]]) -> Dict[str, float]:
        # Use MCP-PIP-DIP angle for index/middle/ring/pinky
        # Landmark indices: https://google.github.io/mediapipe/solutions/hands#hand-landmark-model
        idx = _angle(lms[5], lms[6], lms[7])    # index PIP angle
        mid = _angle(lms[9], lms[10], lms[11])  # middle PIP angle
        rng = _angle(lms[13], lms[14], lms[15]) # ring PIP angle
        pnk = _angle(lms[17], lms[18], lms[19]) # pinky PIP angle
        return {"index": idx, "middle": mid, "ring": rng, "pinky": pnk}

    def _is_index_pointing(self, lms) -> bool:
        angles = self._finger_angles(lms)
        index_straight = angles["index"] >= self.cfg.angle_extended_deg
        others_folded = (
            angles["middle"] <= self.cfg.angle_folded_deg and
            angles["ring"] <= self.cfg.angle_folded_deg and
            angles["pinky"] <= self.cfg.angle_folded_deg
        )
        # towards camera: tip z closer than pip
        index_tip_z = lms[8][2]; index_pip_z = lms[6][2]
        towards_cam = (index_tip_z - index_pip_z) <= self.cfg.index_toward_camera_zdiff
        return bool(index_straight and others_folded and towards_cam)

    def _hold_gesture(self, lms) -> bool:
        """Thumb + middle pinch while index is pointing (per spec)."""
        if not self._is_index_pointing(lms):
            return False
        thumb_tip = lms[4]
        middle_tip = lms[12]
        size = _hand_size(lms)
        pinch = _dist(thumb_tip, middle_tip) / size
        return pinch <= self.cfg.hold_pinch_thresh

    # ---- main processing ----
    def process(self, frame_bgr) -> Dict:
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.hands.process(frame_rgb)

        cursor_norm = None
        new_state = "IDLE"
        scroll_amount = 0

        if result.multi_hand_landmarks:
            hand_lms = result.multi_hand_landmarks[0]
            lms = _lm_array(hand_lms)

            if self._hold_gesture(lms):
                new_state = "HOLD"
            elif self._is_index_pointing(lms):
                new_state = "POINTER"
                # cursor uses index fingertip
                ix, iy, _ = lms[8]
                cursor_norm = (1.0 - ix, iy)  # flip x for selfie cam feel
                self.cursor_smooth.append(cursor_norm)
                if len(self.cursor_smooth) > 0:
                    sx = sum([p[0] for p in self.cursor_smooth]) / len(self.cursor_smooth)
                    sy = sum([p[1] for p in self.cursor_smooth]) / len(self.cursor_smooth)
                    cursor_norm = (sx, sy)
            else:
                self.cursor_smooth.clear()

            
            # HOLD scroll logic: vertical movement while holding emits scroll steps
            if self.cfg.scroll_enabled and new_state == "HOLD" and cursor_norm is not None:
                if self._hold_anchor_y is None:
                    self._hold_anchor_y = cursor_norm[1]
                    self._scroll_accum = 0.0
                dy = cursor_norm[1] - self._hold_anchor_y
                if self.cfg.scroll_invert:
                    dy *= -1
                self._scroll_accum += dy
                step = self.cfg.scroll_step_norm
                while abs(self._scroll_accum) >= step:
                    scroll_amount += self.cfg.scroll_lines_per_step * (1 if self._scroll_accum > 0 else -1)
                    self._scroll_accum -= step * (1 if self._scroll_accum > 0 else -1)
                # update anchor to current pos so additional movement keeps scrolling
                self._hold_anchor_y = cursor_norm[1]
            else:
                self._hold_anchor_y = None
                self._scroll_accum = 0.0
# Debug draw
            if self.cfg.draw_debug:
                mp_draw.draw_landmarks(
                    frame_bgr,
                    hand_lms,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

                # Draw cursor point
                if cursor_norm is not None:
                    cx, cy = int(cursor_norm[0] * w), int(cursor_norm[1] * h)
                    cv2.circle(frame_bgr, (cx, cy), 8, (0, 255, 0), -1)

        
        # basic state smoothing (commit to a new state only if it persists N frames)
        if new_state == self.state:
            # no change
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
# transition events
        events: List[str] = []
        if self.prev_state != self.state:
            if self.prev_state != "HOLD" and self.state == "HOLD":
                events.append("HOLD_START")
            elif self.prev_state == "HOLD" and self.state != "HOLD":
                events.append("HOLD_END")

        # FPS
        now = time.time()
        fps = 1.0 / (now - self.frame_time) if now > self.frame_time else 0.0
        self.frame_time = now

        
        # Optional: if using laptop demo, map scroll via pyautogui
        if self.cfg.enable_mouse_demo and PYAUTO_OK and scroll_amount != 0:
            # pyautogui: positive = scroll up. We invert so moving hand down scrolls down.
            import pyautogui as pag
            pag.scroll(-scroll_amount)
# Mouse demo (optional)
        if self.cfg.enable_mouse_demo and PYAUTO_OK and cursor_norm is not None:
            import pyautogui as pag
            sw, sh = pag.size()
            pag.moveTo(cursor_norm[0] * sw, cursor_norm[1] * sh)

        # HUD
        if self.cfg.draw_debug:
            hud = f"State: {self.state}  FPS: {fps:.1f}  SCROLL:{scroll_amount}"
            cv2.putText(frame_bgr, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            if self.state == "HOLD":
                cv2.putText(frame_bgr, "HOLD (thumb+middle pinch)", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            elif self.state == "POINTER":
                cv2.putText(frame_bgr, "POINTER (index toward camera)", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        return {"state": self.state, "cursor_norm": cursor_norm, "events": events, "scroll": scroll_amount, "fps": fps, "frame": frame_bgr}

# --------------------------- Demo Loop --------------------------------
def demo(camera_index: int = 0):
    cfg = GestureConfig(draw_debug=True, enable_mouse_demo=False)
    engine = GestureEngine(cfg)
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            out = engine.process(frame)
            cv2.imshow("PalmCtrl — GestureEngine v2 (Temp)", out["frame"])
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    demo()
