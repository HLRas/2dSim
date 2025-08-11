# Differential Drive Parking Simulator

A 2D top-down simulator of a differential-drive robot that finds the closest parking space, plans a path using A* on a grid, and follows it using a pure-pursuit controller. Includes headless mode for CI and remote environments.

## Features
- Differential-drive kinematics with individual left/right wheel speeds
- A* grid pathfinding to the closest parking space on the right side of the map
- Pure-pursuit tracking controller with adaptive lookahead near goal
- Pygame visualization; headless fallback that saves an image

## Requirements
- Python 3.9+
- Dependencies in `requirements.txt`:
  - `pygame-ce`
  - `numpy`

## Quickstart
```bash
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

- Controls: press `R` to reset, `ESC` to quit.
- The robot will select the nearest parking spot and drive into it.

## Headless Mode
If no display is detected, the simulator runs offscreen and saves the last frame to `parking_result.png` in the working directory. This is useful on CI or servers without a GUI.

To force headless:
```bash
export SDL_VIDEODRIVER=dummy
python main.py
```

## Continuous Integration
A GitHub Actions workflow (`.github/workflows/sim.yml`) runs the simulator headlessly and uploads `parking_result.png` as a build artifact for PRs.