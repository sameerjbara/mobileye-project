# mobileye-project
Description: The Mobileye Traffic Light Detection System is a computer vision-based project aimed at detecting traffic lights (TFL) in video frames using both image processing techniques and neural networks. The project analyzes video frames to identify the locations and states (red or green) of traffic lights, ensuring real-time detection for use in autonomous vehicle technologies.

Key Components:

Traffic Light Detection: The system uses image processing to detect traffic light candidates in a given video clip. The project applies convolution operations, filtering, and color thresholding to identify potential red and green traffic lights.
Neural Networks: The neural network model is designed to enhance detection accuracy, learning from traffic light characteristics (shape, size, and color) in various conditions.
Workflow:
Data Preprocessing: The system processes each frame from a video to identify regions that may contain traffic lights. The raw pixel data is cleaned and filtered to enhance traffic light signals.
TFL Detection: Detection involves identifying candidates for red and green traffic lights. The algorithm utilizes image filters, thresholds, and candidate generation techniques based on the intensity of red and green pixels.
Post-Processing: After candidate generation, post-processing algorithms ensure that detected candidates are indeed traffic lights by checking consistency across multiple frames and validating the detection with further criteria.
 
# Team members : 
Sameer Jbara - Dvir Meir Perkin - Yaniv Sonino - Yinon Tzomi
