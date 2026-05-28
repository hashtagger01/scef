
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import os
import urllib.request


MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("\n" + "="*60)
    print(" INITIAL SETUP: Downloading Google's hand_landmarker.task model...")
    print(" This file (~5.6 MB) is required by the modern MediaPipe Tasks API.")
    print(" Downloading to local directory. This will only happen once...")
    print("="*60 + "\n")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("✔ Model file downloaded successfully!\n")
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        print("Please check your internet connection and try running again.")
        exit(1)


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (9, 10), (10, 11), (11, 12),           # Middle
    (13, 14), (14, 15), (15, 16),          # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (5, 9), (9, 13), (13, 17)              # Knuckle connections
]



def get_diamond():
    """Returns vertices and edges for a 3D Diamond (Double Cone). Used for 0 fingers."""
    vertices = np.array([
        [0, 0, 1.2],    # 0: Top apex
        [0, 0, -1.2],   # 1: Bottom apex
        [1, 0, 0],      # 2: Right
        [0, 1, 0],      # 3: Bottom
        [-1, 0, 0],     # 4: Left
        [0, -1, 0]      # 5: Top
    ], dtype=np.float32)
    
    edges = [
        (0, 2), (0, 3), (0, 4), (0, 5),  # Connect base to top apex
        (1, 2), (1, 3), (1, 4), (1, 5),  # Connect base to bottom apex
        (2, 3), (3, 4), (4, 5), (5, 2)   # Connect the base ring
    ]
    return vertices, edges

def get_pyramid():
    """Returns vertices and edges for a 3D Square Pyramid. Used for 1 finger."""
    vertices = np.array([
        [-1, -1, -0.8], # 0: Bottom-left
        [1, -1, -0.8],  # 1: Bottom-right
        [1, 1, -0.8],   # 2: Top-right
        [-1, 1, -0.8],  # 3: Top-left
        [0, 0, 1.2]     # 4: Apex (tip)
    ], dtype=np.float32)
    
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Square base edges
        (0, 4), (1, 4), (2, 4), (3, 4)   # Side edges to apex
    ]
    return vertices, edges

def get_cube():
    """Returns vertices and edges for a standard 3D Cube. Used for 2 fingers."""
    vertices = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], # Bottom face (0-3)
        [-1, -1, 1],  [1, -1, 1],  [1, 1, 1],  [-1, 1, 1]   # Top face (4-7)
    ], dtype=np.float32)
    
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face edges
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top face edges
        (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical connecting edges
    ]
    return vertices, edges

def get_octahedron():
    """Returns vertices and edges for a 3D Octahedron. Used for 3 fingers."""
    vertices = np.array([
        [-1, 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1], # Mid-plane ring (0-3)
        [0, 1.4, 0],                                    # Top apex (4)
        [0, -1.4, 0]                                    # Bottom apex (5)
    ], dtype=np.float32)
    
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Mid-plane square base
        (0, 4), (1, 4), (2, 4), (3, 4),  # Connect mid-plane to top
        (0, 5), (1, 5), (2, 5), (3, 5)   # Connect mid-plane to bottom
    ]
    return vertices, edges

def get_prism():
    """Returns vertices and edges for a 3D Hexagonal Prism. Used for 4 fingers."""
    vertices = []
    edges = []
    
    for i in range(6):
        angle = i * np.pi / 3  
        x = np.cos(angle)
        y = np.sin(angle)
        vertices.append([x, y, -1])  # Even indices: bottom ring
        vertices.append([x, y, 1])   # Odd indices: top ring
        
    vertices = np.array(vertices, dtype=np.float32)
    
    for i in range(6):
        # Connect bottom ring: (0, 2), (2, 4), ...
        edges.append((2 * i, 2 * ((i + 1) % 6)))
        # Connect top ring: (1, 3), (3, 5), ...
        edges.append((2 * i + 1, 2 * ((i + 1) % 6) + 1))
        # Connect bottom to top: (0, 1), (2, 3), (4, 5)...
        edges.append((2 * i, 2 * i + 1))
        
    return vertices, edges

def get_sphere():
    """Generates vertices and edges for a beautiful 3D wireframe grid Sphere. Used for 5 fingers."""
    vertices = []
    edges = []
    
    lat_steps = 5  # Number of latitude rings (horizontal)
    lon_steps = 10 # Number of longitude divisions (vertical)
    
    # Generate ring vertices
    for i in range(1, lat_steps):
        lat = np.pi * i / lat_steps - np.pi / 2  # Latitude angle
        r = np.cos(lat)  # Radius at this latitude
        z = np.sin(lat)  # Z-height at this latitude
        for j in range(lon_steps):
            lon = 2 * np.pi * j / lon_steps  # Longitude angle
            x = r * np.cos(lon)
            y = r * np.sin(lon)
            vertices.append([x, y, z])
            
    # Add top and bottom poles
    vertices.append([0, 0, 1])   # North Pole: index N-2
    vertices.append([0, 0, -1])  # South Pole: index N-1
    vertices = np.array(vertices, dtype=np.float32)
    N = len(vertices)
    
    # Connect edges
    num_rings = lat_steps - 1
    for r in range(num_rings):
        start = r * lon_steps
        for j in range(lon_steps):
            # Connect horizontal latitude lines
            edges.append((start + j, start + (j + 1) % lon_steps))
            # Connect vertical longitude lines to the next ring
            if r < num_rings - 1:
                edges.append((start + j, start + lon_steps + j))
                
    # Connect poles to outer rings
    north_pole_idx = N - 2
    south_pole_idx = N - 1
    top_ring_start = (num_rings - 1) * lon_steps
    bottom_ring_start = 0
    
    for j in range(lon_steps):
        edges.append((top_ring_start + j, north_pole_idx))
        edges.append((bottom_ring_start + j, south_pole_idx))
        
    return vertices, edges

# Map shape types to names and functions
SHAPE_MAP = {
    0: ("DIAMOND (0 fingers)", get_diamond),
    1: ("PYRAMID (1 finger)", get_pyramid),
    2: ("CUBE (2 fingers)", get_cube),
    3: ("OCTAHEDRON (3 fingers)", get_octahedron),
    4: ("PRISM (4 fingers)", get_prism),
    5: ("SPHERE (5 fingers)", get_sphere)
}



def rotate_x(points, angle):
    """Rotates a set of 3D points around the X-axis by a given angle in degrees."""
    rad = np.radians(angle)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, cos_a, -sin_a],
        [0, sin_a, cos_a]
    ], dtype=np.float32)
    return np.dot(points, rotation_matrix.T)

def rotate_y(points, angle):
    """Rotates a set of 3D points around the Y-axis by a given angle in degrees."""
    rad = np.radians(angle)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    rotation_matrix = np.array([
        [cos_a, 0, sin_a],
        [0, 1, 0],
        [-sin_a, 0, cos_a]
    ], dtype=np.float32)
    return np.dot(points, rotation_matrix.T)

def rotate_z(points, angle):
    """Rotates a set of 3D points around the Z-axis by a given angle in degrees."""
    rad = np.radians(angle)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    rotation_matrix = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ], dtype=np.float32)
    return np.dot(points, rotation_matrix.T)



def main():
    # Setup OpenCV webcam window
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Set custom frame dimensions for higher quality (e.g. HD 720p)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Configure the modern MediaPipe Hand Landmarker Tasks API
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,  # Highly optimized synchronous mode
        num_hands=1
    )

    # 3D Shape telemetry/state variables
    angle_x, angle_y, angle_z = 0.0, 0.0, 0.0
    rot_speed_x, rot_speed_y = 1.0, 1.5  # Default slow rotation speeds
    
   
    smooth_x, smooth_y = 640.0, 360.0    # Smoothed shape coordinates (centered initially)
    smooth_scale = 1.0                    # Smoothed scale (size) multiplier
    
    # Display statistics
    prev_time = time.time()
    fps = 0
    current_shape_idx = 2  # Start with Cube
    
    print("\n" + "="*60)
    print(" 3D HOLOGRAM GESTURE CONTROLLER - RUNNING SUCCESSFULLY!")
    print("="*60)
    print(" Controls:")
    print(" - MOVE HAND: Controls the 3D position of the shape.")
    print(" - PINCH (Thumb + Index): Scales the size of the shape.")
    print(" - MOVE HAND L/R or U/D: Accelerates the Y/X-axis rotation.")
    print(" - TILT HAND: Rotates the shape around its Z-axis.")
    print(" - MOVE HAND HIGH/LOW: Dynamically changes the color of the shape.")
    print(" - SHOW FINGERS: Changes the shape geometry:")
    print("   - Fist (0 fingers)  => DIAMOND")
    print("   - 1 Finger (Index) => PYRAMID")
    print("   - 2 Fingers        => CUBE")
    print("   - 3 Fingers        => OCTAHEDRON")
    print("   - 4 Fingers        => PRISM")
    print("   - 5 Fingers        => SPHERE")
    print(" - Press 'q' or 'ESC' in the webcam window to exit.")
    print("="*60 + "\n")

    # Open the Landmarker in a context manager
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            success, frame = cap.read()
            if not success:
                print("Error: Failed to read from webcam.")
                break

            # Mirror the frame horizontally so it acts like a mirror
            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape

            # Convert frame to MediaPipe Image object (required by Tasks API)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Perform hand detection synchronously in VIDEO mode using timestamp in milliseconds
            timestamp_ms = int(time.time() * 1000)
            results = landmarker.detect_for_video(mp_image, timestamp_ms)

            # Draw a subtle high-tech futuristic grid background on the screen border
            cv2.rectangle(frame, (10, 10), (w - 10, h - 10), (100, 100, 100), 1, cv2.LINE_AA)
            
            # Determine shape parameters
            pinch_detected = False
            fingers_count = 5 # Default to open hand

            if results.hand_landmarks:
                # Get the first detected hand's landmarks list
                hand_landmarks = results.hand_landmarks[0]

                # --- A. CUSTOM SLEEK GLOWING HAND DRAWING ---
                # 1. Translate all landmarks to pixel points
                pixel_points = []
                for lm in hand_landmarks:
                    px = int(lm.x * w)
                    py = int(lm.y * h)
                    pixel_points.append((px, py))

                # 2. Draw high-tech connections
                for conn in HAND_CONNECTIONS:
                    pt1 = pixel_points[conn[0]]
                    pt2 = pixel_points[conn[1]]
                    cv2.line(frame, pt1, pt2, (0, 255, 255), 2, cv2.LINE_AA) # Futuristic Cyan

                # 3. Draw glowing joints
                for pt in pixel_points:
                    cv2.circle(frame, pt, 4, (255, 0, 128), -1) # Magenta core

                # --- B. EXTRACT CRITICAL LANDMARKS (Objects with x, y, z) ---
                # Get Wrist (0), Thumb Tip (4), Index Tip (8), Index MCP (5), Middle MCP (9), Pinky MCP (17)
                wrist = hand_landmarks[0]
                thumb_tip = hand_landmarks[4]
                index_tip = hand_landmarks[8]
                index_mcp = hand_landmarks[5]
                middle_mcp = hand_landmarks[9]
                pinky_mcp = hand_landmarks[17]

                # Convert landmark points to pixel values
                wx, wy = pixel_points[0]
                tx, ty = pixel_points[4]
                ix, iy = pixel_points[8]
                mx, my = pixel_points[9]

                # --- C. SHAPE POSITION TRACKING (Centered at Middle Finger MCP) ---
                # Apply 0.7 smoothing (EMA) to prevent hand jitter
                smooth_x = smooth_x * 0.7 + mx * 0.3
                smooth_y = smooth_y * 0.7 + my * 0.3

                # --- D. PINCH GESTURE FOR SIZE/SCALE ---
                # Measure pixel distance between thumb tip and index tip
                pinch_dist = np.hypot(tx - ix, ty - iy)
                # Normalize the pinch distance by hand physical length (wrist to middle finger base)
                # This ensures size is invariant to hand distance from the camera
                hand_length = np.hypot(wx - mx, wy - my)
                hand_length = max(1.0, hand_length)  # Prevent division by zero
                norm_pinch_dist = pinch_dist / hand_length

                # Map normalized pinch to scale (Pinch Close = small scale, Pinch Wide = large scale)
                # Normal range: 0.1 (extremely close pinch) to 1.5+ (wide open)
                target_scale = norm_pinch_dist * 2.2
                target_scale = np.clip(target_scale, 0.3, 3.5) # Bound scale to reasonable limits
                smooth_scale = smooth_scale * 0.8 + target_scale * 0.2

                # If fingers are very close, highlight the pinch with a neon green connector
                if norm_pinch_dist < 0.35:
                    pinch_detected = True
                    cv2.line(frame, (tx, ty), (ix, iy), (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.circle(frame, (tx, ty), 6, (0, 255, 0), -1)
                    cv2.circle(frame, (ix, iy), 6, (0, 255, 0), -1)
                else:
                    cv2.line(frame, (tx, ty), (ix, iy), (255, 255, 0), 1, cv2.LINE_AA)

                # --- E. PALM TILT FOR Z-AXIS ROTATION ---
                # Calculate angle of the hand vector (wrist to middle finger base)
                dx = mx - wx
                dy = my - wy
                # Arctan2 gives angle in radians. Add 90 deg so that upright vertical hand is 0 deg.
                hand_angle = np.degrees(np.arctan2(dy, dx)) + 90
                angle_z = hand_angle

                # --- F. DYNAMIC SPIN CONTROL ---
                # Modulate continuous auto-rotation speed based on hand's distance from the screen center
                # Hand on right = rotate Y right, Hand on left = rotate Y left
                # Hand on top = rotate X up, Hand on bottom = rotate X down
                rot_speed_y = ((smooth_x - w/2) / (w/2)) * 6.0
                rot_speed_x = ((smooth_y - h/2) / (h/2)) * 6.0

                # --- G. FINGER COUNTING FOR SHAPE SELECTOR ---
                # 1. Thumb extension check
                # Check if distance from thumb tip to pinky MCP is greater than thumb IP to pinky MCP
                px, py = pinky_mcp.x, pinky_mcp.y
                d_tip = np.hypot(thumb_tip.x - px, thumb_tip.y - py)
                d_ip = np.hypot(hand_landmarks[3].x - px, hand_landmarks[3].y - py)
                thumb_open = d_tip > d_ip

                # 2. Four finger extension checks (if tip is higher than PIP joint)
                fingers_tips = [8, 12, 16, 20]
                fingers_pips = [6, 10, 14, 18]
                fingers_open = []
                for tip, pip in zip(fingers_tips, fingers_pips):
                    # Y decreases as you go UP in screen coordinates
                    is_open = hand_landmarks[tip].y < hand_landmarks[pip].y
                    fingers_open.append(is_open)

                # Sum up all open fingers
                fingers_count = int(thumb_open) + sum(fingers_open)
                current_shape_idx = np.clip(fingers_count, 0, 5)

            else:
                # If no hand detected, drift the shape smoothly back to screen center and reset scale
                smooth_x = smooth_x * 0.95 + (w / 2) * 0.05
                smooth_y = smooth_y * 0.95 + (h / 2) * 0.05
                smooth_scale = smooth_scale * 0.95 + 1.2 * 0.05
                # Revert to gentle auto-spin
                rot_speed_x = 0.5
                rot_speed_y = 1.0
                angle_z = angle_z * 0.95  # Slowly untilt

            # --- H. COLOR CONTROL BASED ON HAND HEIGHT ---
            # Map hand height (smooth_y) to a color hue (vertical slider behavior)
            # As you move your hand to the top (y near 0), color shifts.
            hue = int((1.0 - (smooth_y / h)) * 179)
            hue = np.clip(hue, 0, 179)
            # Convert HSV color to standard BGR for OpenCV drawing
            hsv_color = np.uint8([[[hue, 255, 255]]])
            bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
            color_tuple = (int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2]))

            # --- I. ROTATE & PROJECT THE 3D FIGURE ---
            # 1. Update continuous rotation angles
            angle_x = (angle_x + rot_speed_x) % 360
            angle_y = (angle_y + rot_speed_y) % 360

            # 2. Retrieve shape vertices & edges based on the gesture shape selector
            shape_name, shape_func = SHAPE_MAP[current_shape_idx]
            base_vertices, edges = shape_func()

            # 3. Apply scale (controlled by hand pinch)
            # We also scale the base vertices slightly (e.g. base width of 80 pixels)
            shape_pixel_size = 90.0 * smooth_scale
            scaled_vertices = base_vertices * shape_pixel_size

            # 4. Apply 3D Rotations (X, Y, then Z)
            rotated = rotate_x(scaled_vertices, angle_x)
            rotated = rotate_y(rotated, angle_y)
            rotated = rotate_z(rotated, angle_z)

            # 5. Project 3D points onto 2D screen coordinates using perspective projection
            # We put the shape at some distance D in front of the camera to give a 3D perspective depth effect
            camera_distance = 300.0  # Simulated distance of camera from shape in pixels
            focal_length = 350.0     # Focal length projection factor
            
            # Calculate shifted Z coordinate (depth) of each vertex
            # rotated[:, 2] is the Z value of the vertex. Shift it forward.
            z_coords = rotated[:, 2] + camera_distance
            # Safety check: avoid division by zero if depth is negative or near zero
            z_coords = np.where(z_coords <= 10.0, 10.0, z_coords)

            # Project vertices: X_proj = (X / Z) * focal_length + center_x
            proj_x = (rotated[:, 0] / z_coords) * focal_length + smooth_x
            proj_y = (rotated[:, 1] / z_coords) * focal_length + smooth_y

            # 6. Draw glowing wireframe lines
            # Drawing a thick line with the color first, then a thin white line on top creates a beautiful neon laser glow!
            for edge in edges:
                pt1_x, pt1_y = int(proj_x[edge[0]]), int(proj_y[edge[0]])
                pt2_x, pt2_y = int(proj_x[edge[1]]), int(proj_y[edge[1]])
                
                # Draw only if points are reasonably within screen limits to prevent massive line drawing issues
                if -w < pt1_x < 2*w and -h < pt1_y < 2*h and -w < pt2_x < 2*w and -h < pt2_y < 2*h:
                    # Laser Glow (Thick)
                    cv2.line(frame, (pt1_x, pt1_y), (pt2_x, pt2_y), color_tuple, 5, cv2.LINE_AA)
                    # Laser Core (Thin white)
                    cv2.line(frame, (pt1_x, pt1_y), (pt2_x, pt2_y), (255, 255, 255), 1, cv2.LINE_AA)

            # Draw a glowing center dot at the hand's tracked center
            cv2.circle(frame, (int(smooth_x), int(smooth_y)), 5, (255, 255, 255), -1)
            cv2.circle(frame, (int(smooth_x), int(smooth_y)), 8, color_tuple, 1, cv2.LINE_AA)

            # --- J. DRAW THE FUTURISTIC HUD OVERLAY ---
            # 1. Top status header
            cv2.rectangle(frame, (0, 0), (w, 45), (15, 15, 15), -1)  # HUD bar background
            cv2.line(frame, (0, 45), (w, 45), (0, 255, 255), 1, cv2.LINE_AA)  # Cyan separating line
            
            cv2.putText(frame, "SCEF PYTHON PROGRAM", (20, 28), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            
            # Calculate FPS
            curr_time = time.time()
            fps = int(1.0 / (curr_time - prev_time))
            prev_time = curr_time
            cv2.putText(frame, f"SYS_FPS: {fps}", (w - 150, 28), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

            # 2. Left Telemetry Panel (Hologram parameters)
            cv2.rectangle(frame, (20, 60), (320, 240), (10, 10, 10), -1) # Panel Background
            cv2.rectangle(frame, (20, 60), (320, 240), color_tuple, 1, cv2.LINE_AA) # Panel border (color matches shape!)
            
            cv2.putText(frame, "TELEMETRY DATA", (30, 85), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.line(frame, (30, 92), (310, 92), (100, 100, 100), 1)
            
            cv2.putText(frame, f"ACTIVE SHAPE : {shape_name}", (35, 115), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"SCALE SIZE   : {smooth_scale:.2f}x", (35, 135), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"TRACKING POS : X:{int(smooth_x)} Y:{int(smooth_y)}", (35, 155), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            
            rot_status = f"SPIN X:{rot_speed_x:+.1f} Y:{rot_speed_y:+.1f} Z:{angle_z:+.1f}"
            cv2.putText(frame, rot_status, (35, 175), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

            # Display Pinch Alert Status
            pinch_txt = "PINCH DETECTED (SIZING)" if pinch_detected else "TRACKING COORD (IDLE)"
            pinch_color = (0, 255, 0) if pinch_detected else (150, 150, 150)
            cv2.putText(frame, f"GESTURE STATE: {pinch_txt}", (35, 195), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, pinch_color, 1, cv2.LINE_AA)
            
            # Display Finger Count State
            cv2.putText(frame, f"FINGERS SHOWN: {fingers_count}", (35, 215), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

            # 3. Bottom instructions banner
            cv2.rectangle(frame, (20, h - 70), (w - 20, h - 20), (10, 10, 10), -1)
            cv2.rectangle(frame, (20, h - 70), (w - 20, h - 20), (100, 100, 100), 1, cv2.LINE_AA)
            guide_text1 = "anguthe or phli ungli ko touch karne pe shape change hoga "
            guide_text2 = "hant upar neeche karne se ya aage peeche karne se ye fast or slow hoga aur iska colour change hoga left right karne se "
            cv2.putText(frame, guide_text1, (35, h - 52), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, guide_text2, (35, h - 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 255, 255), 1, cv2.LINE_AA)

            # Render current frame in the window
            cv2.imshow("ye 3d hologram ka program hai ", frame)

            # Handle keypress events
            key = cv2.waitKey(1) & 0xFF
            # Quit on 'q' or 'ESC' keys
            if key == ord('q') or key == 27:
                break

    # Clean up and release system resources
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed down successfully. Resources released.")

if __name__ == "__main__":
    main()
