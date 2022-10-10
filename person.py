from imageai.Detection import ObjectDetection
from imageai.Detection import VideoObjectDetection
import os
import cv2

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(execution_path , "yolo.h5"))
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "obj.jpeg"), output_image_path=os.path.join(execution_path , "imagenew.jpg"), minimum_percentage_probability=30)

# for eachObject in detections:
#     print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
#     print("--------------------------------")


execution_path = os.getcwd()

camera = cv2.VideoCapture(0)


detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(execution_path , "yolo.h5"))
detector.loadModel()
video_path = detector.detectObjectsFromVideo(camera_input=camera,
    output_file_path=os.path.join(execution_path, "camera_detected_video")
    , frames_per_second=20, log_progress=True, minimum_percentage_probability=30)

######################### Abubakr######################

# car_cascade = detector.setModelPath(os.path.join(execution_path , "yolo.h5"))
# ret, frames = camera.read()
# gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
# video_path = detector.detectObjectsFromVideo(camera_input=camera,
#     output_file_path=os.path.join(execution_path, "camera_detected_video")
#     , frames_per_second=20, log_progress=True, minimum_percentage_probability=30)
# cv2.imshow('video2', video_path)
