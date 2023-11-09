#Sheekar Banerjee------>> AI Engineering Lead

import socket
import time
import math

import cv2
import numpy as np
import yolov5
#from vinacts_vision import process_image
from nopose_vision import process_image


HEADERSIZE = 1000


s= socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((socket.gethostname(), 1230))
s.listen(5)



# Load the YOLOv5 model
model_path = 'yolov5s.pt'
# device = "cpu"  # for cpu
device = 0  #for gpu
yolov5 = yolov5.YOLOv5(model_path,device,load_on_init=True)

# Load the video
video = cv2.VideoCapture(0)
#video = cv2.VideoCapture("side2.mp4")

clientsocket, address = s.accept()
print(f"Connection from {address} has been established!")

msg = "Welcome to the Server!"
msg = f'{len(msg):<{HEADERSIZE}}' + msg


clientsocket.send(bytes(msg, "utf-8"))

# Get the video's width, height, and frames per second (fps)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))
# Create a VideoWriter object to save the video
#output_file = 'output_video.mp4'  # Specify the output video file name
#video_writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))



# Process each frame of the video
while True:
  # Read the next frame
  success, frame = video.read()
  if not success:
    break


  # Perform object detection on the frame
  results = yolov5.predict(frame, size = 640, augment=False)
  detections = results.pred[0]


  # Check whether the bounding box centroids are inside the ROI
  for detection in detections:    
    xmin    = detection[0]
    ymin    = detection[1]
    xmax    = detection[2]
    ymax    = detection[3]
    score   = detection[4]
    class_id= detection[5]
    centroid_x = int(xmin + xmax) // 2
    centroid_y =  int(ymin + ymax) // 2

    #Threshold score
    if score >= 0.6:  
      if class_id == 0:
        #color = (240, 32, 160)  #purple
        color = (0, 0, 255)

        markers = cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 1)

        #padding
        padding = 25
        person = frame[int(ymin):int(ymax), int(xmin):int(xmax)]

        try:  ### mediapipe
            b = process_image(person)
            msg = f"The landmark is: {b}"
            msg = f'{len(msg):<{HEADERSIZE}}' + msg
            clientsocket.send(bytes(msg, "utf-8"))
            time.sleep(0.01)
            
        except:
            pass

      else:
        pass


  # Display the frame
  cv2.imshow("Vinacts Vision", frame)
  #video_writer.write(frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break



# Release the video capture object
video.release()
#video_writer.release()

cv2.destroyAllWindows()