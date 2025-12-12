# Object-tracking-with-dlib

**Real-time Object Tracking using Dlib and MobileNet SSD**
Cool object size estimator with just OpenCV and python
All thanks to Adrian Rosebrock (from **[pyimagesearch](https://pyimagesearch.com/)** for making great tutorials. This project is inspired from his blog: **[Object tracking with dlib](https://pyimagesearch.com/2018/10/22/object-tracking-with-dlib/)**. I have included the author's code and the one i wrote my self as well.

# Key Points
1. Steps involved:
     1. Initial Detection: Use a robust Deep Learning model (MobileNet SSD) to find the target object (e.g., 'person', 'car') in the very first frame.
     2. Tracker Seeding: Initialize Dlib's Correlation Tracker with the bounding box coordinates obtained from the initial detection.
     3. Continuous Tracking: Use the Dlib Correlation Tracker to update the object's position and size in subsequent frames without running the slower Deep Learning model again.
     4. Visualization: Draw the updated bounding box and the object's class label on the frame for real-time output.
     5. Cleanup: Properly release the video stream and destroy windows upon completion.
2. Assumptions:
     1. The target object (specified by the user via command-line arguments) is present and visible in the starting frames of the video.
     2. The video stream contains relatively smooth motion suitable for Dlib's correlation tracking algorithm.
     3. The necessary pre-trained Caffe model files (.prototxt and .caffemodel) are available in the specified path.
  
# Requirements: (Tested Versions)
1. python (3.7.3)
2. opencv-python (4.1.0)
3. numpy (1.61.4)
4. imutils (0.5.2)

# How to Run:
1. Download Model Files: Ensure you have the necessary Caffe model files (MobileNetSSD_deploy.prototxt and MobileNetSSD_deploy.caffemodel) available in a local directory (e.g., mobilenet_ssd/).
2. Run the Script: Execute the track_object.py script from your terminal, specifying the input video, model files, and the class you wish to track.
₩₩₩bash
python track_object.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
--model mobilenet_ssd/MobileNetSSD_deploy.caffemodel \
--video path/to/input/video.mp4 \
--label person

# Results:
The results show decent real-time performance, successfully maintaining the bounding box on the target object across video frames after the initial detection.
This system effectively demonstrates the efficiency of the hybrid approach:
- Initial Frame: The detection process is slower due to the MobileNet SSD model loading and execution.
- Subsequent Frames: Tracking is significantly faster and more stable, relying on the low-cost Dlib Correlation Tracker rather than repeated deep learning inference.

# The limitations:
The slight imperfections, such as minor drifting or temporary loss of tracking during rapid changes in object scale or quick occlusion, are primarily due to:
- Tracker Type Limitations: The Correlation Tracker is optimized for speed but can be less robust than full re-detection or more complex trackers in highly dynamic scenes.
- Environmental Factors: Rapid camera motion or complex backgrounds in the input video stream can occasionally challenge the tracker's stability.
