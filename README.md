# FinalProjectTELLO

# Autonomous Indoor Drone Navigation Using Computer Vision

This project was developed as part of my final project for the B.Sc. in Electrical Engineering at Tel Aviv University.

##  Project Overview

The goal of this project is to develop an **autonomous drone** capable of navigating a **closed indoor environment** using **only computer vision**, with no external sensors or GPS.

The drone is tasked with navigating from a known **user** to a known **target** within a household environment. It must:
- Recognize the user and the target using facial recognition.
- Detect **open doors** within the environment to determine valid passages.
- Navigate autonomously through open paths while avoiding obstacles.
- Generate a **live map** of the house as it explores, showing:
  - The scanned route in **red**
  - The final optimized path from user to target in **blue**

## Technologies Used

- **Face Recognition**  
  Used to identify both the user and the target from the drone’s camera feed.

- **Depth Anything**  
  A real-time depth estimation model used to convert RGB frames to depth maps efficiently.

- **YOLOv10 Object Detection**  
  A custom-trained model was used to detect **open doors**.  
  The model was trained on **depth images** of doors instead of RGB images, as open doors are clearly distinguishable in depth maps by a black rectangle (indicating deep background behind the doorframe).

## 🖼️ Example: Open Door in Depth Map

In the example below, you can clearly see how an open door appears as a black rectangle in the depth map, making it easily detectable.

![Open door in depth map](https://github.com/MatanBaadani/FinalProjectTELLO/blob/main/depth_model_3.jpg?raw=true)

## Sources and Credits

This project uses the following open-source resources:
- [DJITelloPy](https://github.com/damiafuentes/DJITelloPy) - Python library for controlling DJI Tello drones (MIT License)
- [CV-Zone](https://github.com/cvzone/cvzone) - Computer vision helper library (MIT License)
- [Depth-Anything](https://github.com/LiheYoung/Depth-Anything) - Depth estimation model (Apache License 2.0)

Thank you to the developers and the open-source community!

## License

This project is licensed under the MIT License - see the [LICENSE](drone-Detection/LICENSE) file for details.
