"""
PalmCtrl — Gesture Engine (Optimized for Low Latency & High Accuracy)
Focus: fast, stable index finger tracking & precise hold detection
"""

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import math

try:
    import pyautogui
    PYAUTO_OK = True
except Exception:
    PYAUTO_OK = False


# ---------------------------- Config ---------------------------------
@dataclass
class GestureConfig:
    detection_confidence: float = 0.5
    tracking_confidence: float = 0.5
    max_hands: int = 1
    model_complexity: int = 0

    # Geometry thresholds
    angle_extended_deg: float = 155.0
    angle_folded_deg: float = 135.0
    index_toward_camera_zdiff: float = -0.02
    hold_pinch_thresh: float = 0.22

    # Adaptive smoothing
    smooth_window_min: int = 2
    smooth_window_max: int = 5
    motion_threshold: float = 0.05

    # Debug & platform
    draw_debug: bool = True
    enable_mouse_demo: bool = False
    platform: str = "laptop"  # 'laptop' or 'android'


# ---------------------------- Utils ----------------------------------
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
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def _lm_array(hand_landmarks) -> List[Tuple[float, float, float]]:
    return [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]


# ---------------------------- Engine ---------------------------------
class GestureEngine:
    def __init__(self, cfg: GestureConfig | None = None):
        self.cfg = cfg or GestureConfig()
        self.hands = mp_hands.Hands(
            max_num_hands=self.cfg.max_hands,
            model_complexity=self.cfg.model_complexity,
            min_detection_confidence=self.cfg.detection_confidence,
            min_tracking_confidence=self.cfg.tracking_confidence,
        )
        self.state: str = "IDLE"
        self.cursor_smooth: Deque[Tuple[float, float]] = deque(maxlen=self.cfg.smooth_window_max)
        self.prev_cursor = None
        self.frame_time = time.time()

    def _is_index_pointing(self, lms) -> bool:
        idx_angle = _angle(lms[5], lms[6], lms[7])
        mid_angle = _angle(lms[9], lms[10], lms[11])
        ring_angle = _angle(lms[13], lms[14], lms[15])
        pinky_angle = _angle(lms[17], lms[18], lms[19])
        index_tip_z, index_pip_z = lms[8][2], lms[6][2]

        straight = idx_angle >= self.cfg.angle_extended_deg
        folded = (mid_angle < self.cfg.angle_folded_deg and
                  ring_angle < self.cfg.angle_folded_deg and
                  pinky_angle < self.cfg.angle_folded_deg)
        forward = (index_tip_z - index_pip_z) <= self.cfg.index_toward_camera_zdiff

        return straight and folded and forward

    def _hold_gesture(self, lms) -> bool:
        thumb_tip = lms[4]
        mid_tip = lms[12]
        size = _dist(lms[0], lms[9])
        pinch = _dist(thumb_tip, mid_tip) / size
        return pinch <= self.cfg.hold_pinch_thresh

    def process(self, frame_bgr) -> Dict:
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.hands.process(frame_rgb)

        cursor_norm, state = None, "IDLE"
        if result.multi_hand_landmarks:
            lms = _lm_array(result.multi_hand_landmarks[0])

            if self._hold_gesture(lms):
                state = "HOLD"
            elif self._is_index_pointing(lms):
                state = "POINTER"
                ix, iy, _ = lms[8]
                cursor_norm = (1 - ix, iy)

                if self.prev_cursor:
                    motion = _dist((cursor_norm[0], cursor_norm[1], 0),
                                   (self.prev_cursor[0], self.prev_cursor[1], 0))
                    if motion > self.cfg.motion_threshold:
                        self.cursor_smooth = deque(list(self.cursor_smooth)[-self.cfg.smooth_window_min:], 
                                                   maxlen=self.cfg.smooth_window_min)
                    else:
                        self.cursor_smooth = deque(list(self.cursor_smooth)[-self.cfg.smooth_window_max:], 
                                                   maxlen=self.cfg.smooth_window_max)

                self.cursor_smooth.append(cursor_norm)
                sx = sum(p[0] for p in self.cursor_smooth) / len(self.cursor_smooth)
                sy = sum(p[1] for p in self.cursor_smooth) / len(self.cursor_smooth)
                cursor_norm = (sx, sy)
                self.prev_cursor = cursor_norm
            else:
                self.cursor_smooth.clear()

            if self.cfg.draw_debug:
                mp_draw.draw_landmarks(frame_bgr, result.multi_hand_landmarks[0],
                                       mp_hands.HAND_CONNECTIONS,
                                       mp_styles.get_default_hand_landmarks_style(),
                                       mp_styles.get_default_hand_connections_style())
                if cursor_norm:
                    cx, cy = int(cursor_norm[0]*w), int(cursor_norm[1]*h)
                    cv2.circle(frame_bgr, (cx, cy), 8, (0, 255, 0), -1)

        self.state = state
        fps = 1.0 / max(time.time() - self.frame_time, 1e-6)
        self.frame_time = time.time()

        # Mouse demo only for laptop
        if self.cfg.platform == "laptop" and self.cfg.enable_mouse_demo and cursor_norm and PYAUTO_OK:
            sw, sh = pyautogui.size()
            pyautogui.moveTo(cursor_norm[0]*sw, cursor_norm[1]*sh)

        if self.cfg.draw_debug:
            cv2.putText(frame_bgr, f"{state} | FPS:{fps:.1f}", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return {"state": state, "cursor_norm": cursor_norm, "fps": fps, "frame": frame_bgr}


def demo(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    engine = GestureEngine(GestureConfig(draw_debug=True))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out = engine.process(frame)
        cv2.imshow("PalmCtrl — Low Latency Mode", out["frame"])
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    demo()
