3D Hologram Hand Gesture Controller 🌌🖐️
An interactive, high-tech computer vision application that tracks your hand movements and projects a 3D holographic wireframe onto your webcam feed. You can manipulate its 3D coordinates, size, rotation speed, color hue, and shape geometry using natural hand gestures.

🚀 How to Run the Code
To launch the gesture controller, make sure you are in the project folder C:\Users\abhin\.gemini\antigravity\scratch\hand_3d_control. You can run it easily in one of the following ways:

Option A: From VS Code Terminal (PowerShell)
Just type and run:

.\run.ps1
Option B: From Command Prompt (cmd)
run.bat
Option C: From File Explorer
Simply double-click the run.bat file inside your hand_3d_control folder!

Note: Make sure your webcam is plugged in and not currently in use by other applications like Teams or Zoom.

🔮 Telemetry & Interactive Gestures
Here is a breakdown of the holographic controls you can perform in front of your camera:

Feature	Hand Gesture	Visualization Effect
3D Position	Move Hand	The 3D hologram is anchored and moves with your hand (tracked at the center of your palm) using a smoothing filter to avoid jitter.
Size / Scale	Pinch (Thumb + Index)	Bring your thumb and index finger together to shrink the shape; pull them apart to enlarge it. This is normalized using hand length, so moving closer/further from your camera won't cause unexpected resizing!
Shape Geometry	Finger Counting	Show a specific number of fingers to swap the geometry model instantly:
• 0 fingers (Fist) ✊ ➔ Diamond (Double Cone)
• 1 finger (Index) ☝️ ➔ Square Pyramid
• 2 fingers ✌️ ➔ Cube
• 3 fingers 🤟 ➔ Octahedron
• 4 fingers 🖖 ➔ Hexagonal Prism
• 5 fingers (Open Palm) 🖐️ ➔ Glowing Wireframe Sphere
Rotation Speed	Move Hand off-center	The shape automatically rotates. Moving your hand further to the right/left accelerates horizontal spin; moving it up/down accelerates vertical spin.
Roll Rotation	Tilt Hand	Tilting your palm horizontally (clockwise/counter-clockwise) rotates the 3D shape along its Z-axis in exact sync with your hand!
Color Spectrum	Hand Vertical Height	Raising your hand higher on the screen shifts the color hue dynamically through a glowing HSV rainbow spectrum.
🛠️ Code Structure & How It Works
3D Projection: The script defines mathematical coordinate points of vertices and connecting edges. Instead of requiring massive gaming engines like Pygame or OpenGL, it performs perspective calculation in pure NumPy: 
Proj
X
=
X
Z
+
Distance
×
FocalLength
+
Hand
X
 This ensures high efficiency, keeping the script single-file and extremely simple to understand!
Neon Glow Lines: By drawing a thick, highly saturated color line followed by a thin white line centered exactly on top, OpenCV renders a beautiful high-tech "neon laser" glow effect.
Cyberpunk HUD: Renders live telemetry data, including active frame rate (FPS), hand position coordinates, active gesture, scale factors, and an instructional overlay, to give you an immersive interface.
🛑 How to Exit
Simply press q or ESC while focused on the webcam display window to shut down the tracking engine and close the application.
