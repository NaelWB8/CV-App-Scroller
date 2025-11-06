# gesture_engine_fixed_v5.py
"""
PalmCtrl — Gesture Engine (Fixed v5)
Improvements:
 - min_hand_score gating
 - short-dropout tolerance (keep/predict cursor for N frames)
 - EMA smoothing + velocity prediction to keep cursor stable during brief dropouts
 - tighter debug HUD showing hand_score, forward_score, unseen_count
 - returns cursor_norm_display (for preview overlay) and cursor_norm_screen (for mapping)
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

# ----------------- Config -----------------
@dataclass
class GestureConfig:
    detection_confidence: float = 0.5
    tracking_confidence: float = 0.5
    max_hands: int = 1
    model_complexity: int = 1

    preview_mirror: bool = False
    map_to_screen: bool = True

    # forward pointing
    angle_extended_deg: float = 150.0
    angle_folded_deg: float = 130.0
    forward_dot_thresh: float = 0.4
    lateral_ratio_thresh: float = 0.85

    # hold pinch
    hold_pinch_thresh: float = 0.22

    # smoothing / prediction
    smooth_alpha_pos: float = 0.6       # EMA alpha for position (higher -> more responsive)
    smooth_alpha_vel: float = 0.4       # EMA alpha for velocity
    predict_frames: float = 1.0         # how many frames ahead to predict on dropout
    max_unseen_frames_keep: int = 6     # keep/predict for this many frames when no landmarks
    min_hand_score: float = 0.5         # require mediapipe hand score >= this

    # debounce / scroll
    min_stable_frames: int = 2
    scroll_enabled: bool = True
    scroll_step_norm: float = 0.015
    scroll_lines_per_step: int = 2
    scroll_invert: bool = False

    # debug / platform
    platform: str = "laptop"  # 'laptop' or 'android'
    draw_debug: bool = True
    enable_mouse_demo: bool = False

# ----------------- Utils -----------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

def _angle(a,b,c):
    bax, bay, baz = a[0]-b[0], a[1]-b[1], a[2]-b[2]
    bcx, bcy, bcz = c[0]-b[0], c[1]-b[1], c[2]-b[2]
    dot = bax*bcx + bay*bcy + baz*bcz
    mag1 = math.sqrt(bax*bax + bay*bay + baz*baz)
    mag2 = math.sqrt(bcx*bcx + bcy*bcy + bcz*bcz)
    if mag1==0 or mag2==0: return 0.0
    cos = max(-1.0, min(1.0, dot/(mag1*mag2)))
    return math.degrees(math.acos(cos))

def _dist(a,b):
    return math.sqrt(sum((a[i]-b[i])**2 for i in range(3)))

def _lm_array(hand_landmarks):
    return [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]

# ----------------- Engine -----------------
class GestureEngine:
    def __init__(self, cfg: GestureConfig|None = None):
        self.cfg = cfg or GestureConfig()
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.cfg.max_hands,
            model_complexity=self.cfg.model_complexity,
            min_detection_confidence=self.cfg.detection_confidence,
            min_tracking_confidence=self.cfg.tracking_confidence
        )
        self.state = "IDLE"
        self.prev_state = "IDLE"
        self._pending_state = None
        self._pending_count = 0

        # smoothed position & velocity (screen space, non-mirrored)
        self.pos_ema: Optional[Tuple[float,float]] = None
        self.vel_ema: Optional[Tuple[float,float]] = None

        self.prev_cursor_display: Optional[Tuple[float,float]] = None
        self.prev_cursor_screen: Optional[Tuple[float,float]] = None

        # unseen/dropout handling
        self.unseen_count = 0

        # hold/scroll
        self._hold_anchor_y = None
        self._scroll_accum = 0.0

        self.frame_time = time.time()

    def _finger_angles(self,lms):
        idx = _angle(lms[5], lms[6], lms[8])
        mid = _angle(lms[9], lms[10], lms[12])
        ring = _angle(lms[13], lms[14], lms[16])
        pinky = _angle(lms[17], lms[18], lms[20])
        return {"index":idx, "middle":mid, "ring":ring, "pinky":pinky}

    def _is_index_pointing(self, lms):
        angles = self._finger_angles(lms)
        index_straight = angles["index"] >= self.cfg.angle_extended_deg
        others_folded = (angles["middle"] <= self.cfg.angle_folded_deg and
                         angles["ring"] <= self.cfg.angle_folded_deg and
                         angles["pinky"] <= self.cfg.angle_folded_deg)
        vx = lms[8][0]-lms[6][0]; vy = lms[8][1]-lms[6][1]; vz = lms[8][2]-lms[6][2]
        norm = math.sqrt(vx*vx + vy*vy + vz*vz) + 1e-9
        forward_score = -vz/norm
        lateral_ratio = abs(vx)/norm
        forward_ok = (forward_score >= self.cfg.forward_dot_thresh and lateral_ratio <= self.cfg.lateral_ratio_thresh)
        zdiff_ok = (lms[8][2]-lms[6][2]) <= 0.03  # allow a little tolerance
        return bool(index_straight and others_folded and forward_ok and zdiff_ok), float(forward_score)

    def _hold_gesture(self, lms):
        pointing, _ = self._is_index_pointing(lms)
        if not pointing: return False
        thumb = lms[4]; mid = lms[12]
        size = max(1e-6, _dist(lms[0], lms[9]))
        pinch = _dist(thumb, mid)/size
        return pinch <= self.cfg.hold_pinch_thresh

    def _hand_score(self, result):
        # return MediaPipe handedness score if available (0..1)
        try:
            if result.multi_handedness and len(result.multi_handedness)>0:
                return float(result.multi_handedness[0].classification[0].score)
        except Exception:
            pass
        return 1.0

    def _update_ema_and_predict(self, new_pos:Tuple[float,float], dt:float):
        # new_pos in screen coords (non-mirrored)
        if self.pos_ema is None:
            self.pos_ema = new_pos
            self.vel_ema = (0.0, 0.0)
            return self.pos_ema
        # compute instantaneous velocity
        dx = (new_pos[0] - self.pos_ema[0]) / max(dt,1e-6)
        dy = (new_pos[1] - self.pos_ema[1]) / max(dt,1e-6)
        # EMA update velocity and position
        ax = self.cfg.smooth_alpha_vel
        ay = self.cfg.smooth_alpha_pos
        vx = ax*dx + (1-ax)*self.vel_ema[0]
        vy = ax*dy + (1-ax)*self.vel_ema[1]
        self.vel_ema = (vx, vy)
        px = ay*new_pos[0] + (1-ay)*self.pos_ema[0]
        py = ay*new_pos[1] + (1-ay)*self.pos_ema[1]
        self.pos_ema = (px, py)
        return self.pos_ema

    def _predict_when_unseen(self, dt):
        # dt: seconds since last actual update; use self.vel_ema to predict ahead up to predict_frames
        if self.pos_ema is None or self.vel_ema is None: return None
        # predict using frames as unit: predict_frames * (vel * dt_frame)
        # convert predict_frames (frames) into seconds using last known frame delta (~1/fps)
        # approximate using dt
        pred_secs = self.cfg.predict_frames * dt
        px = self.pos_ema[0] + self.vel_ema[0]*pred_secs
        py = self.pos_ema[1] + self.vel_ema[1]*pred_secs
        # clamp
        px = max(0.0, min(1.0, px))
        py = max(0.0, min(1.0, py))
        return (px, py)

    def process(self, frame_bgr) -> Dict:
        h,w = frame_bgr.shape[:2]
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False
        result = self.hands.process(img_rgb)
        img_rgb.flags.writeable = True

        cursor_display = None
        cursor_screen = None
        new_state = "IDLE"
        scroll_amount = 0
        forward_score_val = None
        hand_score_val = None

        hand_found = False
        # gate on hand score to avoid trusting spurious low-confidence detections
        hand_score_val = self._hand_score(result) if result is not None else 0.0
        if result and result.multi_hand_landmarks and hand_score_val >= self.cfg.min_hand_score:
            hand_found = True
            hand = result.multi_hand_landmarks[0]
            lms = _lm_array(hand)

            # determine state
            if self._hold_gesture(lms):
                new_state = "HOLD"
            else:
                pointing, forward_score = self._is_index_pointing(lms)
                forward_score_val = forward_score
                if pointing:
                    new_state = "POINTER"
                    ix,iy,_ = lms[8]
                    cursor_screen = (ix, iy)  # screen coords (non-mirrored)
                    cursor_display = (1.0-ix, iy) if self.cfg.preview_mirror else (ix, iy)

                    # update EMA & velocity using screen coords (we smooth/predict on screen-space)
                    now = time.time()
                    dt = max(1e-6, now - self.frame_time)
                    # update ema: pos in screen coords
                    ema_pos = self._update_ema_and_predict(cursor_screen, dt)
                    # map ema->display (respect preview_mirror)
                    if self.cfg.preview_mirror:
                        cursor_display = (1.0 - ema_pos[0], ema_pos[1])
                        cursor_screen = ema_pos
                    else:
                        cursor_display = ema_pos
                        cursor_screen = ema_pos
                    self.prev_cursor_display = cursor_display
                    self.prev_cursor_screen = cursor_screen
                else:
                    # not pointing
                    self.prev_cursor_display = None
                    self.prev_cursor_screen = None
                    # clear ema? keep it to allow faster reacquire
            # scroll logic uses screen coords (persist across preview mirroring)
            if self.cfg.scroll_enabled and new_state == "HOLD" and self.prev_cursor_screen is not None:
                if self._hold_anchor_y is None:
                    self._hold_anchor_y = self.prev_cursor_screen[1]
                    self._scroll_accum = 0.0
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

            if self.cfg.draw_debug:
                mp_draw.draw_landmarks(frame_bgr, hand, mp_hands.HAND_CONNECTIONS,
                                      mp_styles.get_default_hand_landmarks_style(),
                                      mp_styles.get_default_hand_connections_style())

                if self.prev_cursor_display is not None:
                    cx,cy = int(self.prev_cursor_display[0]*w), int(self.prev_cursor_display[1]*h)
                    cv2.circle(frame_bgr, (cx,cy), 8, (0,255,0), -1)

        # If no valid hand this frame but we had one recently: hold/predict
        if not hand_found:
            self.unseen_count += 1
            if self.unseen_count <= self.cfg.max_unseen_frames_keep and self.pos_ema is not None:
                # predict using last dt approx (safe fallback)
                now = time.time()
                dt = max(1e-6, now - self.frame_time)
                pred = self._predict_when_unseen(dt)
                if pred is not None:
                    # map to display
                    cursor_screen = pred
                    cursor_display = (1.0 - pred[0], pred[1]) if self.cfg.preview_mirror else pred
                    self.prev_cursor_screen = cursor_screen
                    self.prev_cursor_display = cursor_display
                    # keep state as previous (do not generate HOLD_START/HOLD_END during short dropout)
                    new_state = self.state
            else:
                # beyond tolerance -> treat as lost
                self.pos_ema = None
                self.vel_ema = None
                self.prev_cursor_display = None
                self.prev_cursor_screen = None
                self.unseen_count = 0
                new_state = "IDLE"

        if hand_found:
            self.unseen_count = 0

        # state smoothing (debounce)
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

        # laptop demo OS cursor mapping (use screen coords)
        if (self.cfg.platform == "laptop" and self.cfg.enable_mouse_demo and PYAUTO_OK and self.prev_cursor_screen is not None):
            try:
                sw, sh = pyautogui.size()
                px = max(0.0, min(1.0, self.prev_cursor_screen[0])); py = max(0.0, min(1.0, self.prev_cursor_screen[1]))
                pyautogui.moveTo(px*sw, py*sh)
                if scroll_amount != 0:
                    pyautogui.scroll(-scroll_amount)
            except Exception:
                pass

        if self.cfg.draw_debug:
            hud = f"State:{self.state} FPS:{fps:.1f} hand_score:{hand_score_val:.2f} forward:{(forward_score_val or 0):.2f} unseen:{self.unseen_count}"
            cv2.putText(frame_bgr, hud, (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)

        display_frame = frame_bgr
        if self.cfg.preview_mirror:
            display_frame = cv2.flip(frame_bgr, 1)

        return {
            "state": self.state,
            "cursor_norm_screen": self.prev_cursor_screen,
            "cursor_norm_display": self.prev_cursor_display,
            "events": events,
            "scroll": scroll_amount,
            "fps": fps,
            "frame": display_frame,
        }

# ------------- Demo (same controls) -----------------
def demo(cam_index=0):
    cfg = GestureConfig()
    engine = GestureEngine(cfg)
    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FPS, 30); cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("Demo: ESC/q quit | m toggle preview mirror | p toggle platform | c calibrate")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            out = engine.process(frame)
            cv2.imshow("PalmCtrl v5", out["frame"])
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            elif key==ord('m'):
                cfg.preview_mirror = not cfg.preview_mirror; print("preview_mirror:", cfg.preview_mirror)
            elif key==ord('p'):
                cfg.platform = 'android' if cfg.platform=='laptop' else 'laptop'; print("platform:", cfg.platform)
            elif key==ord('c'):
                calibrate_forward(engine)
    finally:
        cap.release(); cv2.destroyAllWindows()

def calibrate_forward(engine:GestureEngine, duration=2.0):
    # quick calibration helper; same approach as before
    print("Calibration: point index at camera for 2s...")
    cap = cv2.VideoCapture(0)
    t0 = time.time(); vals=[]
    try:
        while time.time()-t0 < duration:
            ok, frame = cap.read()
            if not ok: break
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); img_rgb.flags.writeable=False
            res = engine.hands.process(img_rgb); img_rgb.flags.writeable=True
            if res and res.multi_hand_landmarks:
                lms = _lm_array(res.multi_hand_landmarks[0])
                vx = lms[8][0] - lms[6][0]; vz = lms[8][2] - lms[6][2]
                norm = math.sqrt(vx*vx + (lms[8][1]-lms[6][1])**2 + vz*vz)+1e-9
                vals.append(-vz/norm)
            cv2.waitKey(1)
    finally:
        cap.release()
    if vals:
        avg = sum(vals)/len(vals); print("avg forward:", avg)
    else:
        print("no hand detected")

if __name__=="__main__":
    demo()
