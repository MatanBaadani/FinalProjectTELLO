# FinalProjectTELLO

## Overview
**FinalProjectTELLO** is an autonomous drone project using the DJI Tello.  
The drone flies indoors, exploring rooms autonomously while searching for a known target — a recognized face of Elon Musk.  
Additionally, the drone constructs a map showing the optimal route from a known user's position (another known face) to the target.

The system combines computer vision, depth estimation, and autonomous navigation, powered by several open-source libraries and models.

## Features
- Real-time face detection and recognition (known faces: Elon Musk and user)
- Indoor autonomous navigation
- Depth estimation for detecting open doors and accessible paths
- Dynamic mapping from user location to target

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/MatanBaadani/FinalProjectTELLO.git
    cd FinalProjectTELLO
    ```

2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up your DJI Tello drone and ensure it is connected via Wi-Fi to your computer.

## Usage

1. Start the program:
    ```bash
    python depth-anything-main/depth+map+target_user.py
    ```

2. Follow the on-screen instructions. The drone will:
   - Take off and begin exploring.
   - Search for the known user and target faces.
   - Build and print a navigation map from the user to the target once both are detected.

## Sources and Credits

This project uses the following open-source resources:
- [DJITelloPy](https://github.com/damiafuentes/DJITelloPy) - Python library for controlling DJI Tello drones (MIT License)
- [CV-Zone](https://github.com/cvzone/cvzone) - Computer vision helper library (MIT License)
- [Depth-Anything](https://github.com/LiheYoung/Depth-Anything) - Depth estimation model (Apache License 2.0)

Thank you to the developers and the open-source community!

## License

This project is licensed under the MIT License - see the [LICENSE](drone-Detection/LICENSE) file for details.
