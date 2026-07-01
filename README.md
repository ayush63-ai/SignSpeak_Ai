# SignSpeak AI

SignSpeak AI is an advanced hand gesture recognition and augmented reality application. Built with **OpenCV** and **MediaPipe**, it identifies hand landmarks in real-time through a webcam feed to perform complex gesture tracking.

## Features
- **Robust Finger Counting**: Counts fingers independently of hand orientation (palm or back facing the camera) using joint angle analysis.
- **Custom Gestures**: Detects specific hand configurations such as:
  - OK Good (Thumbs Up)
  - NOT GOOD (Thumbs Down)
  - Spider-Man
  - Call Me
  - Go Left / Go Right
- **Cinematic AR HUD**: An interactive, Iron Man-style Heads-Up Display (HUD) overlay that reacts to your gestures:
  - **Biometric Palm Scan**: Triggered when presenting an open, moving palm.
  - **Target Lock**: Activated by extending exactly two fingers.
  - **Repulsor Charge**: Holding a steady open palm charges a repulsor blast.
  - **Repulsor Burst**: Clenching a fist after charging unleashes a visual energy burst.
  - **Combat Mode**: Clenching a fist triggers screen shake and red alert borders.

## Requirements
- Python 3.x
- OpenCV (`cv2`)
- NumPy
- MediaPipe

## Files
- `SignSpeak.py`: The main application script containing gesture detection logic and HUD rendering.
- `hand_landmarker.task`: The MediaPipe Hand Landmarker model file.
