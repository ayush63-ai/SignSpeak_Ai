import cv2
import time
import math
import numpy as np
import mediapipe as mp

# Text-to-speech removed (running without voice output)

# Initialize MediaPipe Hand Landmarker
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

def create_hand_landmarker():
    """Create MediaPipe Hand Landmarker"""
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
        running_mode=VisionRunningMode.VIDEO
    )
    return HandLandmarker.create_from_options(options)

def count_fingers(landmarks):
    """
    Robust finger counting that works for:
    - Left and right hands
    - Palm or back of hand facing the camera
    - Different wrist rotations
    
    Uses joint angles instead of raw image coordinates so that
    half-bent fingers are not mistakenly treated as extended.
    """
    if not landmarks:
        return 0

    def joint_angle(a, b, c):
        """
        Return the angle (in degrees) at joint b formed by points a-b-c.
        For a straight finger this will be close to 180 degrees.
        """
        ba = np.array([a.x - b.x, a.y - b.y, getattr(a, "z", 0.0) - getattr(b, "z", 0.0)])
        bc = np.array([c.x - b.x, c.y - b.y, getattr(c, "z", 0.0) - getattr(b, "z", 0.0)])
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        if norm_ba < 1e-6 or norm_bc < 1e-6:
            return 0.0
        cos_angle = float(np.dot(ba, bc) / (norm_ba * norm_bc))
        # Clamp for numerical stability
        cos_angle = max(-1.0, min(1.0, cos_angle))
        return math.degrees(math.acos(cos_angle))

    # Angle threshold: only count fingers that are very straight
    STRAIGHT_ANGLE_DEG = 160.0

    # Index, Middle, Ring, Pinky: angle at PIP joint (mcp - pip - tip)
    angle_index = joint_angle(landmarks[5], landmarks[6], landmarks[8])
    angle_middle = joint_angle(landmarks[9], landmarks[10], landmarks[12])
    angle_ring = joint_angle(landmarks[13], landmarks[14], landmarks[16])
    angle_pinky = joint_angle(landmarks[17], landmarks[18], landmarks[20])

    is_index_up = angle_index > STRAIGHT_ANGLE_DEG
    is_middle_up = angle_middle > STRAIGHT_ANGLE_DEG
    is_ring_up = angle_ring > STRAIGHT_ANGLE_DEG
    is_pinky_up = angle_pinky > STRAIGHT_ANGLE_DEG

    # ===== Thumb detection (works for palm/back and both hands) =====
    wrist = landmarks[0]
    thumb_mcp = landmarks[2]
    thumb_ip = landmarks[3]
    thumb_tip = landmarks[4]

    # Angle at thumb IP joint (mcp - ip - tip)
    angle_thumb = joint_angle(thumb_mcp, thumb_ip, thumb_tip)

    # Distances from wrist: thumb tip must be noticeably further than MCP
    def dist(a, b):
        return math.sqrt(
            (a.x - b.x) ** 2 +
            (a.y - b.y) ** 2 +
            (getattr(a, "z", 0.0) - getattr(b, "z", 0.0)) ** 2
        )

    dist_tip_wrist = dist(thumb_tip, wrist)
    dist_mcp_wrist = dist(thumb_mcp, wrist)
    THUMB_DIST_THRESH = 0.025  # how much further the tip must be from the wrist

    is_thumb_straight = angle_thumb > STRAIGHT_ANGLE_DEG
    is_thumb_far_enough = dist_tip_wrist > dist_mcp_wrist + THUMB_DIST_THRESH

    # Thumb is considered "up" only when it is both straight and clearly extended
    is_thumb_up = is_thumb_straight and is_thumb_far_enough

    fingers = sum(
        1 for is_up in (
            is_index_up,
            is_middle_up,
            is_ring_up,
            is_pinky_up,
            is_thumb_up,
        ) if is_up
    )

    return fingers

def detect_custom_gesture(landmarks):
    """Detect custom hand gestures before normal finger counting"""
    if not landmarks:
        return None
    
    # MediaPipe landmark indices:
    # Thumb: 2=mcp, 3=ip, 4=tip
    # Index: 5=mcp, 6=pip, 8=tip
    # Middle: 9=mcp, 10=pip, 12=tip
    # Ring: 13=mcp, 14=pip, 16=tip
    # Pinky: 17=mcp, 18=pip, 20=tip
    
    thumb_mcp = landmarks[2]
    thumb_ip = landmarks[3]
    thumb_tip = landmarks[4]
    
    index_pip = landmarks[6]
    index_tip = landmarks[8]
    
    middle_pip = landmarks[10]
    middle_tip = landmarks[12]
    
    ring_pip = landmarks[14]
    ring_tip = landmarks[16]
    
    pinky_pip = landmarks[18]
    pinky_tip = landmarks[20]
    wrist = landmarks[0]
    # --- Orientation-independent detection using projections ---
    # Build normalized hand direction (wrist -> middle_mcp) and perpendicular
    wx, wy = wrist.x, wrist.y
    mx, my = landmarks[9].x, landmarks[9].y
    hx, hy = mx - wx, my - wy
    h_norm = (hx * hx + hy * hy) ** 0.5
    if h_norm < 1e-6:
        # fallback: assume upward in image coords
        ux, uy = 0.0, -1.0
    else:
        ux, uy = hx / h_norm, hy / h_norm

    # side vector (perpendicular)
    sx, sy = -uy, ux

    def proj(pt):
        return (pt.x - wx) * ux + (pt.y - wy) * uy

    def lat(pt):
        return (pt.x - wx) * sx + (pt.y - wy) * sy

    # projection values
    p_index_tip = proj(index_tip)
    p_index_pip = proj(index_pip)
    p_index_mcp = proj(landmarks[5])

    p_middle_tip = proj(middle_tip)
    p_middle_pip = proj(middle_pip)
    p_middle_mcp = proj(landmarks[9])

    p_ring_tip = proj(ring_tip)
    p_ring_pip = proj(ring_pip)
    p_ring_mcp = proj(landmarks[13])

    p_pinky_tip = proj(pinky_tip)
    p_pinky_pip = proj(pinky_pip)
    p_pinky_mcp = proj(landmarks[17])

    p_thumb_tip = proj(thumb_tip)
    p_thumb_ip = proj(thumb_ip)
    p_thumb_mcp = proj(thumb_mcp)

    lat_thumb_tip = lat(thumb_tip)
    lat_thumb_ip = lat(thumb_ip)

    # thresholds (normalized projection units)
    proj_ext_thresh = 0.04   # finger tip must be this far beyond pip to be 'up'
    proj_fold_thresh = 0.03  # pip must be this far beyond tip to be 'folded'
    thumb_long_thresh = 0.03
    thumb_lat_thresh = 0.10

    # Strict per-finger states (orientation independent)
    is_index_up = (p_index_tip > p_index_pip + proj_ext_thresh)
    is_index_folded = (p_index_pip > p_index_tip + proj_fold_thresh)

    is_middle_up = (p_middle_tip > p_middle_pip + proj_ext_thresh)
    is_middle_folded = (p_middle_pip > p_middle_tip + proj_fold_thresh)

    is_ring_up = (p_ring_tip > p_ring_pip + proj_ext_thresh)
    is_ring_folded = (p_ring_pip > p_ring_tip + proj_fold_thresh)

    is_pinky_up = (p_pinky_tip > p_pinky_pip + proj_ext_thresh)
    is_pinky_folded = (p_pinky_pip > p_pinky_tip + proj_fold_thresh)

    # Thumb states using longitudinal and lateral projections
    thumb_long_delta = p_thumb_tip - p_thumb_ip
    thumb_lat_delta = abs(lat_thumb_tip - lat_thumb_ip)

    is_thumb_sideways = thumb_lat_delta > thumb_lat_thresh

    # Image-space checks for thumb up/down to be robust across palm/back and both hands
    # Use y-coordinates (image space: smaller y is up)
    thumb_up_img = (thumb_tip.y < thumb_ip.y) and (thumb_ip.y < thumb_mcp.y)
    thumb_down_img = (thumb_tip.y > thumb_ip.y) and (thumb_ip.y > thumb_mcp.y)

    # Combine longitudinal extension with image vertical direction to confirm up/down
    is_thumb_extended_long = thumb_long_delta > thumb_long_thresh and thumb_up_img
    is_thumb_extended_back = thumb_long_delta < -thumb_long_thresh and thumb_down_img

    # Consider thumb folded if it's neither extended nor sideways (small lateral delta)
    is_thumb_folded = (not is_thumb_extended_long) and (not is_thumb_extended_back) and (thumb_lat_delta < thumb_lat_thresh * 0.6)

    # Ensure mutual exclusivity: if any finger clearly up, treat accordingly
    # 1) Thumbs Up: thumb extended away from wrist (long positive) AND all other fingers folded
    if is_thumb_extended_long and is_index_folded and is_middle_folded and is_ring_folded and is_pinky_folded:
        return "OK Good"

    # 2) Thumbs Down: thumb extended back toward wrist (long negative) AND other fingers folded
    if is_thumb_extended_back and is_index_folded and is_middle_folded and is_ring_folded and is_pinky_folded:
        return "NOT GOOD"

    # 3) Spider-Man: index & pinky up, middle & ring folded, thumb folded (not extended sideways)
    if is_index_up and is_pinky_up and is_middle_folded and is_ring_folded and is_thumb_folded:
        return "Bravo"

    # 4) Call Me: thumb extended sideways, pinky up, other fingers folded
    if is_thumb_sideways and is_pinky_up and is_index_folded and is_middle_folded and is_ring_folded:
        return "Call Me"

    # 5/6) Point Left / Right: only index up, other fingers folded, thumb folded
    only_index_up = is_index_up and is_middle_folded and is_ring_folded and is_pinky_folded and is_thumb_folded
    if only_index_up:
        # Use raw x comparison against wrist.x (works for left/right hands and palm/back)
        if index_tip.x < wrist.x - 0.02:
            return "Go Left"
        if index_tip.x > wrist.x + 0.02:
            return "Go Right"

    # No custom gesture detected
    return None

# ============================================================================
# DOCTOR STRANGE MAGIC RING FEATURE
# ============================================================================

def is_open_palm(landmarks):
    """
    Detect open palm (all fingers extended, thumb out).
    Works for both hands and both palm/back orientations.
    
    Returns: True if open palm detected, False otherwise
    """
    if not landmarks:
        return False
    
    # Finger indices: tip, pip
    fingers_to_check = [
        (8, 6),    # index tip, index pip
        (12, 10),  # middle tip, middle pip
        (16, 14),  # ring tip, ring pip
        (20, 18),  # pinky tip, pinky pip
    ]
    
    # Check all 4 fingers are extended (tip.y < pip.y in image space)
    all_fingers_extended = all(
        landmarks[tip].y < landmarks[pip].y
        for tip, pip in fingers_to_check
    )
    
    if not all_fingers_extended:
        return False
    
    # Check thumb is extended outward (not folded)
    # Using image-space y-coordinates for orientation independence
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    thumb_mcp = landmarks[2]
    
    thumb_extended = (thumb_tip.y < thumb_ip.y) and (thumb_ip.y < thumb_mcp.y)
    
    return all_fingers_extended and thumb_extended


def get_palm_center(landmarks, frame_width, frame_height):
    """
    Calculate palm center from MCP joints (5, 9, 13, 17).
    
    Returns: (center_x, center_y) in pixel coordinates, or None if invalid
    """
    mcp_indices = [5, 9, 13, 17]  # index, middle, ring, pinky MCP
    
    try:
        mcp_points = [landmarks[i] for i in mcp_indices]
        
        # Calculate average position in normalized coordinates
        avg_x = sum(p.x for p in mcp_points) / len(mcp_points)
        avg_y = sum(p.y for p in mcp_points) / len(mcp_points)
        
        # Convert to pixel coordinates
        center_x = int(avg_x * frame_width)
        center_y = int(avg_y * frame_height)
        
        return (center_x, center_y)
    except:
        return None


def get_hand_rotation_angle(landmarks):
    """
    Calculate hand rotation angle using wrist (0) and middle MCP (9).
    
    Returns: angle in degrees (-180 to 180)
    """
    try:
        wrist = landmarks[0]
        middle_mcp = landmarks[9]
        
        # Calculate direction vector from wrist to middle MCP
        dx = middle_mcp.x - wrist.x
        dy = middle_mcp.y - wrist.y
        
        # Calculate angle in degrees
        import math
        angle = math.degrees(math.atan2(dy, dx))
        
        return angle
    except:
        return 0.0


class HUDManager:
    """
    Cinematic Iron Man–style HUD manager.
    Handles all on-screen overlays and animations separately from gesture logic.
    """

    def __init__(self):
        self.start_time = time.time()
        # Per-hand persistent state (indexed by integer hand id)
        self.hand_states = {}

    # ---------- High-level entry point ----------
    def draw(self, frame, hands_info, now):
        """
        Draw global AI overlay and per-hand HUD modes.

        Args:
            frame: BGR image (modified in place)
            hands_info: list of dicts with keys:
                - id: int
                - palm_center: (x, y) or None
                - angle: float degrees
                - fingers: int finger count
            now: current time in seconds
        """
        h, w = frame.shape[:2]

        # Always-on AI overlay
        self._draw_ai_overlay(frame, now)

        combat_active = False

        for hand in hands_info:
            hand_id = hand["id"]
            center = hand.get("palm_center")
            angle = hand.get("angle", 0.0)
            fingers = hand.get("fingers", 0)

            mode = self._update_hand_state(hand_id, center, fingers, now, (w, h))

            if mode in ("combat", "repulsor_burst"):
                combat_active = True

            if center is None:
                continue

            if mode == "palm_scan":
                self._draw_palm_scan(frame, center, angle, hand_id, now)
            elif mode == "target_lock":
                self._draw_target_lock(frame, center, angle, hand_id, now)
            elif mode == "repulsor":
                self._draw_repulsor_charge(frame, center, angle, hand_id, now)
            elif mode == "repulsor_burst":
                self._draw_repulsor_burst(frame, center, angle, hand_id, now)

        # Global combat HUD (screen shake + red border)
        if combat_active:
            self._draw_combat_overlay(frame, now)

    # ---------- Hand state machine ----------
    def _get_state(self, hand_id):
        if hand_id not in self.hand_states:
            self.hand_states[hand_id] = {
                "mode": "idle",
                "last_seen": 0.0,
                "open_palm_start": None,
                "repulsor_charge": 0.0,
                "repulsor_active": False,
                "burst_start": None,
                "last_center": None,
                "last_update": self.start_time,
            }
        return self.hand_states[hand_id]

    def _update_hand_state(self, hand_id, center, fingers, now, frame_size):
        """
        Decide which HUD mode to activate for this hand based on gesture & motion.
        Modes:
            - palm_scan: open palm
            - target_lock: two fingers
            - repulsor: open palm held steady
            - repulsor_burst: triggered when charged repulsor hand makes a fist
            - combat: fist (no repulsor burst)
            - idle: nothing special
        """
        w, h = frame_size
        state = self._get_state(hand_id)
        dt = max(1e-3, now - state["last_update"])
        state["last_seen"] = now

        # Tunables
        steady_pixels = min(w, h) * 0.025  # how much palm can move and still be "steady"
        steady_time = 0.8                  # seconds before repulsor starts charging
        charge_rate = 0.5                  # units per second (2 seconds to reach 100%)
        burst_duration = 0.7               # seconds of burst animation

        # Gesture classification
        is_fist = fingers == 0
        is_two_fingers = fingers == 2
        is_open_palm = fingers == 5

        # Handle existing burst animation regardless of current gesture
        if state["mode"] == "repulsor_burst" and state["burst_start"] is not None:
            if now - state["burst_start"] > burst_duration:
                state["mode"] = "idle"
                state["repulsor_charge"] = 0.0
                state["repulsor_active"] = False
            state["last_update"] = now
            return state["mode"]

        if center is None:
            # If we lost tracking, slowly decay repulsor charge and fall back to idle
            state["repulsor_charge"] = max(0.0, state["repulsor_charge"] - dt * 0.4)
            if now - state["last_seen"] > 0.5:
                state["mode"] = "idle"
                state["open_palm_start"] = None
                state["repulsor_active"] = False
            state["last_update"] = now
            return state["mode"]

        # Fist: either trigger repulsor burst (if charged) or combat mode
        if is_fist:
            if state["repulsor_active"] and state["repulsor_charge"] > 0.7:
                state["mode"] = "repulsor_burst"
                state["burst_start"] = now
            else:
                state["mode"] = "combat"
            state["open_palm_start"] = None
            state["repulsor_active"] = False
            state["last_update"] = now
            return state["mode"]

        # Two fingers → Target lock
        if is_two_fingers:
            state["mode"] = "target_lock"
            state["open_palm_start"] = None
            state["repulsor_active"] = False
            state["last_update"] = now
            return state["mode"]

        # Open palm → Palm scan or Repulsor charge (if steady)
        if is_open_palm:
            if state["open_palm_start"] is None:
                state["open_palm_start"] = now

            last_center = state["last_center"]
            movement = 0.0
            if last_center is not None:
                dx = center[0] - last_center[0]
                dy = center[1] - last_center[1]
                movement = math.hypot(dx, dy)

            state["last_center"] = center

            time_open = now - state["open_palm_start"]
            is_steady = movement < steady_pixels

            if is_steady and time_open > steady_time:
                # Repulsor charging
                state["mode"] = "repulsor"
                state["repulsor_active"] = True
                state["repulsor_charge"] = min(
                    1.0, state["repulsor_charge"] + dt * charge_rate
                )
            else:
                # Palm scan mode while hand is moving or just appeared
                state["mode"] = "palm_scan"
                state["repulsor_active"] = False

            state["last_update"] = now
            return state["mode"]

        # Any other gesture → idle, decay charge
        state["mode"] = "idle"
        state["open_palm_start"] = None
        state["repulsor_active"] = False
        state["repulsor_charge"] = max(0.0, state["repulsor_charge"] - dt * 0.3)
        state["last_update"] = now
        return state["mode"]

    # ---------- Always-on AI overlay ----------
    def _draw_ai_overlay(self, frame, now):
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # Subtle grid lines
        grid_color = (40, 120, 200)  # bluish
        spacing = 80
        for x in range(0, w, spacing):
            cv2.line(overlay, (x, 0), (x, h), grid_color, 1)
        for y in range(0, h, spacing):
            cv2.line(overlay, (0, y), (w, y), grid_color, 1)

        # Corner brackets
        corner_len = 40
        thickness = 3
        corner_color = (0, 200, 255)
        # Top-left
        cv2.line(overlay, (0, 0), (corner_len, 0), corner_color, thickness)
        cv2.line(overlay, (0, 0), (0, corner_len), corner_color, thickness)
        # Top-right
        cv2.line(overlay, (w - corner_len, 0), (w, 0), corner_color, thickness)
        cv2.line(overlay, (w - 1, 0), (w - 1, corner_len), corner_color, thickness)
        # Bottom-left
        cv2.line(overlay, (0, h - corner_len), (0, h), corner_color, thickness)
        cv2.line(overlay, (0, h - 1), (corner_len, h - 1), corner_color, thickness)
        # Bottom-right
        cv2.line(
            overlay,
            (w - corner_len, h - 1),
            (w - 1, h - 1),
            corner_color,
            thickness,
        )
        cv2.line(
            overlay,
            (w - 1, h - corner_len),
            (w - 1, h - 1),
            corner_color,
            thickness,
        )

        # Animated vertical scanning line
        scan_x = int((now * 80) % w)
        cv2.line(overlay, (scan_x, 0), (scan_x, h), (0, 255, 255), 2)

        # Top-left status panel
        panel_w, panel_h = 260, 110
        panel_x, panel_y = 20, 20
        panel_rect = (panel_x, panel_y, panel_x + panel_w, panel_y + panel_h)
        px1, py1, px2, py2 = panel_rect
        cv2.rectangle(overlay, (px1, py1), (px2, py2), (10, 40, 80), -1)

        # Slight opacity so background is visible
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

        # Panel contents (drawn directly for crisp text)
        energy = 82 + 8 * math.sin(now * 0.3)
        armor = 94 + 3 * math.sin(now * 0.2 + 1.2)
        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(
            frame,
            "STARK SYSTEMS HUD",
            (panel_x + 8, panel_y + 22),
            font,
            0.6,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"ENERGY  : {int(energy):3d}%",
            (panel_x + 8, panel_y + 48),
            font,
            0.5,
            (0, 230, 255),
            1,
        )
        cv2.putText(
            frame,
            f"ARMOR   : {int(armor):3d}%",
            (panel_x + 8, panel_y + 70),
            font,
            0.5,
            (0, 230, 180),
            1,
        )
        cv2.putText(
            frame,
            "AI STATUS: ONLINE",
            (panel_x + 8, panel_y + 92),
            font,
            0.5,
            (0, 255, 0),
            1,
        )

    # ---------- Mode renderers ----------
    def _draw_palm_scan(self, frame, center, angle, hand_id, now):
        """Palm Scan mode: circular scanning ring + progress + label."""
        h, w = frame.shape[:2]
        cx, cy = center
        overlay = np.zeros_like(frame)

        # Base ring radius
        base_r = int(min(w, h) * 0.11)
        # Breathing effect
        pulse = 1.0 + 0.06 * math.sin(now * 4.0)
        r = int(base_r * pulse)

        # Rotating radial lines
        num_spokes = 12
        for i in range(num_spokes):
            a = (now * 120 + i * (360 / num_spokes) + hand_id * 15) * math.pi / 180
            inner_r = int(r * 0.45)
            outer_r = int(r * 1.05)
            x1 = int(cx + inner_r * math.cos(a))
            y1 = int(cy + inner_r * math.sin(a))
            x2 = int(cx + outer_r * math.cos(a))
            y2 = int(cy + outer_r * math.sin(a))
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 200, 255), 2)

        # Main circular ring
        cv2.circle(overlay, (cx, cy), r, (255, 180, 0), 2)
        cv2.circle(overlay, (cx, cy), int(r * 0.55), (0, 180, 255), 1)

        # Circular progress (0-100% looping)
        cycle = 2.5
        phase = (now % cycle) / cycle
        end_angle = int(360 * phase)
        cv2.ellipse(
            overlay,
            (cx, cy),
            (r + 6, r + 6),
            0,
            0,
            end_angle,
            (0, 255, 255),
            3,
        )

        # Blend overlay
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)

        # Text label and percentage
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            frame,
            "BIOMETRIC SCAN",
            (cx - 120, cy - r - 25),
            font,
            0.7,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"{int(phase * 100):3d}%",
            (cx - 30, cy + r + 35),
            font,
            0.7,
            (0, 255, 200),
            2,
        )

    def _draw_target_lock(self, frame, center, angle, hand_id, now):
        """Target lock: crosshair, rotating brackets, locked text, pulse."""
        h, w = frame.shape[:2]
        cx, cy = center
        overlay = np.zeros_like(frame)

        # Central crosshair
        size = int(min(w, h) * 0.07)
        color = (0, 0, 255)  # red
        thickness = 2
        cv2.line(overlay, (cx - size, cy), (cx + size, cy), color, thickness)
        cv2.line(overlay, (cx, cy - size), (cx, cy + size), color, thickness)
        cv2.circle(overlay, (cx, cy), int(size * 0.5), color, 2)

        # Rotating brackets
        t = now
        angle_deg = (t * 90 + hand_id * 30) % 360
        rad = math.radians(angle_deg)
        bracket_r = int(size * 1.6)
        for i in range(4):
            a = rad + i * math.pi / 2
            bx = int(cx + bracket_r * math.cos(a))
            by = int(cy + bracket_r * math.sin(a))
            half = int(size * 0.5)
            cv2.rectangle(
                overlay,
                (bx - half, by - half),
                (bx + half, by + half),
                (0, 0, 255),
                2,
            )

        # Pulsing outer circle
        pulse = 1.0 + 0.08 * math.sin(t * 6.0)
        outer_r = int(size * 2.2 * pulse)
        cv2.circle(overlay, (cx, cy), outer_r, (0, 0, 255), 2)

        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        # Text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            frame,
            "TARGET LOCKED",
            (cx - 130, cy - outer_r - 20),
            font,
            0.7,
            (0, 0, 255),
            2,
        )

    def _draw_repulsor_charge(self, frame, center, angle, hand_id, now):
        """Repulsor charge: blue glowing circle + charge percentage."""
        cx, cy = center
        h, w = frame.shape[:2]
        overlay = np.zeros_like(frame)

        state = self._get_state(hand_id)
        charge = state.get("repulsor_charge", 0.0)

        base_r = int(min(w, h) * 0.09)
        r = int(base_r * (1.0 + 0.15 * charge))

        # Multi-layer glow
        for i in range(4):
            ri = int(r * (1.0 + i * 0.18))
            alpha = int(80 - i * 15)
            color = (255, 120 + alpha, 0)  # BGR (neon blue-ish)
            cv2.circle(overlay, (cx, cy), ri, color, 2)

        # Inner core
        cv2.circle(overlay, (cx, cy), int(r * 0.4), (255, 255, 255), -1)
        cv2.circle(overlay, (cx, cy), int(r * 0.6), (255, 200, 80), 2)

        # Rotating hexagon
        sides = 6
        rot = now * 120
        hex_r = int(r * 0.9)
        pts = []
        for i in range(sides):
            a = math.radians(rot + i * (360 / sides))
            pts.append((int(cx + hex_r * math.cos(a)), int(cy + hex_r * math.sin(a))))
        pts = np.array(pts, np.int32)
        cv2.polylines(overlay, [pts], True, (255, 200, 80), 2)

        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Charge text
        font = cv2.FONT_HERSHEY_SIMPLEX
        pct = int(charge * 100)
        cv2.putText(
            frame,
            f"REPULSOR CHARGE {pct:3d}%",
            (cx - 160, cy - r - 25),
            font,
            0.6,
            (255, 220, 120),
            2,
        )

    def _draw_repulsor_burst(self, frame, center, angle, hand_id, now):
        """Energy burst animation after charged repulsor + fist."""
        cx, cy = center
        h, w = frame.shape[:2]
        overlay = np.zeros_like(frame)

        state = self._get_state(hand_id)
        if state["burst_start"] is None:
            return
        t = now - state["burst_start"]
        duration = 0.7
        p = min(1.0, t / duration)

        max_r = int(min(w, h) * 0.6)
        r = int(max_r * p)

        # Expanding energy ring
        cv2.circle(overlay, (cx, cy), r, (255, 255, 255), 4)
        cv2.circle(overlay, (cx, cy), int(r * 0.7), (0, 255, 255), 3)

        # Flash overlay (fade out)
        flash_alpha = 1.0 - p
        flash = np.zeros_like(frame)
        flash[:] = (255, 255, 255)
        cv2.addWeighted(flash, 0.35 * flash_alpha, overlay, 1.0, 0, overlay)

        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            frame,
            "ENERGY BURST",
            (cx - 120, cy - r - 10),
            font,
            0.8,
            (0, 255, 255),
            2,
        )

    def _draw_combat_overlay(self, frame, now):
        """Combat mode: red border glow + flashing warning + subtle screen shake."""
        h, w = frame.shape[:2]

        # Screen shake via small affine transform
        intensity = 4
        dx = int(math.sin(now * 18.0) * intensity)
        dy = int(math.cos(now * 23.0) * intensity)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shaken = cv2.warpAffine(frame, M, (w, h))
        frame[:, :, :] = shaken

        # Red border glow
        overlay = frame.copy()
        thickness = 12
        glow_color = (0, 0, 255)
        cv2.rectangle(overlay, (0, 0), (w - 1, h - 1), glow_color, thickness)

        # Flashing intensity
        alpha = 0.4 + 0.3 * (0.5 * (1 + math.sin(now * 6.0)))
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Warning text with flashing
        font = cv2.FONT_HERSHEY_SIMPLEX
        flash = 0.5 * (1 + math.sin(now * 8.0))
        color = (
            int(50 + 205 * flash),
            int(50 * (1 - flash)),
            int(50 * (1 - flash)),
        )

        cv2.putText(
            frame,
            "COMBAT MODE ACTIVATED",
            (40, int(h * 0.18)),
            font,
            0.9,
            color,
            3,
        )

def draw_magic_ring(frame, center, radius=80, angle=0, glow_intensity=200, frame_num=0):
    """
    Draw an animated, glowing rotating magic ring centered on palm.
    Features: multi-layer rotation, pulsing effect, dynamic glow, magical sparkles.
    
    Args:
        frame: OpenCV image
        center: (x, y) tuple for ring center
        radius: ring radius in pixels
        angle: rotation angle in degrees
        glow_intensity: glow effect intensity (0-255)
        frame_num: current frame number (for animation timing)
    """
    if center is None:
        return
    
    h, w = frame.shape[:2]
    cx, cy = center
    
    # Clamp center to frame bounds
    if cx < 0 or cx >= w or cy < 0 or cy >= h:
        return
    
    # ===== PULSING EFFECT =====
    pulse = 0.9 + 0.1 * np.sin(frame_num * 0.1)
    dynamic_radius = int(radius * pulse)
    
    # Create overlay for glow effects
    overlay = frame.copy()
    
    # ===== OUTER GLOW AURA (Soft diffused) =====
    glow_colors = [
        (20, 60, 150),       # Deep blue-orange
        (0, 120, 200),       # Medium orange
        (50, 160, 255),      # Light orange
        (100, 200, 255),     # Very light orange
    ]
    
    for i, color in enumerate(glow_colors):
        glow_radius = dynamic_radius + (len(glow_colors) - i) * 15
        cv2.circle(overlay, (cx, cy), glow_radius, color, 1)
    
    # Blend glow with frame
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
    
    # ===== INNER ROTATING SPOKES (Fast rotation) =====
    num_spokes = 8
    spoke_angle_offset = angle + frame_num * 3
    
    for i in range(num_spokes):
        spoke_angle_rad = (spoke_angle_offset + (i * 360 / num_spokes)) * np.pi / 180
        
        # Varying spoke lengths for visual interest
        outer_r = dynamic_radius * (0.9 + 0.1 * np.sin(i))
        inner_r = dynamic_radius * 0.5
        
        outer_x = int(cx + outer_r * np.cos(spoke_angle_rad))
        outer_y = int(cy + outer_r * np.sin(spoke_angle_rad))
        
        inner_x = int(cx + inner_r * np.cos(spoke_angle_rad))
        inner_y = int(cy + inner_r * np.sin(spoke_angle_rad))
        
        # Gradient color (bright orange to yellow)
        cv2.line(frame, (inner_x, inner_y), (outer_x, outer_y), (0, 180, 255), 2)
    
    # ===== MIDDLE ROTATING RING (Medium speed) =====
    middle_ring_angle = angle * 0.8 + frame_num * 1.5
    num_segments = 16
    
    for i in range(num_segments):
        segment_angle = (middle_ring_angle + (i * 360 / num_segments)) * np.pi / 180
        segment_radius = dynamic_radius * 0.75
        
        x = int(cx + segment_radius * np.cos(segment_angle))
        y = int(cy + segment_radius * np.sin(segment_angle))
        
        # Bright cyan/yellow color
        brightness = int(150 + 100 * np.sin(frame_num * 0.15 + i * 0.5))
        cv2.circle(frame, (x, y), 2, (brightness, 200, 255), -1)
    
    # ===== MAIN OUTER RING =====
    cv2.circle(frame, (cx, cy), dynamic_radius, (0, 165, 255), 3)
    
    # ===== SECONDARY INNER RING (Slower rotation) =====
    inner_ring_angle = angle * 0.3 + frame_num * 0.8
    inner_radius = int(dynamic_radius * 0.65)
    
    num_inner_segments = 12
    for i in range(num_inner_segments):
        segment_angle = (inner_ring_angle + (i * 360 / num_inner_segments)) * np.pi / 180
        
        x = int(cx + inner_radius * np.cos(segment_angle))
        y = int(cy + inner_radius * np.sin(segment_angle))
        
        cv2.circle(frame, (x, y), 1, (100, 220, 255), -1)
    
    # Draw inner circle outline for depth
    cv2.circle(frame, (cx, cy), inner_radius, (50, 200, 255), 2)
    
    # ===== MAGICAL SPARKLES =====
    num_sparkles = 6
    for i in range(num_sparkles):
        sparkle_angle = (angle + (i * 360 / num_sparkles) + frame_num * 5) * np.pi / 180
        sparkle_r = dynamic_radius + 30
        
        sx = int(cx + sparkle_r * np.cos(sparkle_angle))
        sy = int(cy + sparkle_r * np.sin(sparkle_angle))
        
        # Twinkling effect
        sparkle_brightness = int(150 + 100 * np.sin(frame_num * 0.2 + i))
        if 0 <= sx < w and 0 <= sy < h:
            cv2.circle(frame, (sx, sy), 2, (255, sparkle_brightness, 0), -1)
            cv2.circle(frame, (sx, sy), 4, (255, sparkle_brightness, 0), 1)
    
    # ===== CENTER CORE (Pulsing bright center) =====
    center_brightness = int(200 + 55 * np.sin(frame_num * 0.12))
    cv2.circle(frame, (cx, cy), 6, (0, center_brightness, 255), -1)
    cv2.circle(frame, (cx, cy), 8, (50, center_brightness, 255), 2)


# Ring animation state (smooth rotation)
ring_angle = 0.0
RING_ROTATION_SPEED = 8.0  # degrees per frame

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Create hand landmarker
hand_landmarker = create_hand_landmarker()

print("Show hand gestures to the camera")
print("Press 'q' or ESC to quit")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # Convert BGR to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    
    gesture = "Show Your Hand"
    
    # Run hand detection
    frame_count += 1
    detection_result = hand_landmarker.detect_for_video(mp_image, frame_count)
    
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            # Draw landmarks
            for landmark in hand_landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            
            # ===== MAGIC RING DETECTION & DRAWING =====
            if is_open_palm(hand_landmarks) and count_fingers(hand_landmarks) == 5:
                # Calculate palm center and hand rotation
                palm_center = get_palm_center(hand_landmarks, w, h)
                hand_angle = get_hand_rotation_angle(hand_landmarks)
                
                # Update ring animation (smooth rotation)
                ring_angle = hand_angle
                
                # Draw the magic ring with animation
                draw_magic_ring(frame, palm_center, radius=80, angle=ring_angle, frame_num=frame_count)
                
                # Update gesture display
                gesture = "✨ Magic Ring ✨"
            else:
                # Check for custom gestures first
                custom_gesture = detect_custom_gesture(hand_landmarks)
                
                if custom_gesture:
                    gesture = custom_gesture
                else:
                    # Count fingers for normal gestures
                    fingers = count_fingers(hand_landmarks)
                    
                    # Determine gesture based on finger count
                    if fingers == 0:
                        gesture = "Fist✊"
                    elif fingers == 1:
                        gesture = "One Finger👆"
                    elif fingers == 2:
                        gesture = "Victory✌️"
                    elif fingers == 3:
                        gesture = "Three Fingers🤟"
                    elif fingers == 4:
                        gesture = "Four Fingers🖐️"
                    elif fingers == 5:
                        gesture = "Open Hand✋"
    
    # Display text
    cv2.putText(frame, gesture, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.putText(frame, "Press Q/ESC to quit", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    # (voice output removed)
    
    # Show frame
    cv2.imshow("SignSpeak AI - Gesture Recognition", frame)
    
    # Check if window is closed
    if cv2.getWindowProperty("SignSpeak AI - Gesture Recognition", cv2.WND_PROP_VISIBLE) < 1:
        print("Window closed by user")
        break
    
    # Check for exit key
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:  # 'q' or ESC
        print("Closing application...")
        break

cap.release()
cv2.destroyAllWindows()
print("Application closed successfully")
