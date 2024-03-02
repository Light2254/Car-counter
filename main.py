import cv2
import math
import numpy as np
import os
import torch
import cvzone
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from ultralytics import YOLO
from Sort import *


def checkDevice():
    # Test cuda availability
    try:
        torch.cuda.is_available()
    except:
        device = 'cpu'
    else:
        device = 'cuda:0'
    finally:
        print('Running on %s' % device)
        return device
    
def checkVideo(videoPath):
    if not os.path.exists(videoPath):
        print('Video not found')
        exit()
    else:
        video = cv2.VideoCapture(videoPath)
        return video



def draw_boxes(img, className, pred, color=(255, 0, 255)):
    global detections
    for result in pred:
        for box in result.boxes:
            # Get the coordinates of the box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # Convert to int
            w, h = x2 - x1, y2 - y1
            # Get the confidence score
            conf = math.ceil(box.conf[0] * 100) / 100
            # Get the predicted class label
            cls = className[int(box.cls[0])]

            if (cls == 'car' or cls == 'truck' or cls == 'bus') and conf > 0.3:          
                currentArray = np.array([x1,y1,x2,y2,conf]) #Tracking
                detections = np.vstack((detections , currentArray))
    return img
    

def main(videoPath, modelName):
    global detections,count
    device = checkDevice()  # Check device for running the model
    model = YOLO(modelName).to(device)  # Load model
    video = checkVideo(videoPath)  # Load video

    # Load Mask
    mask = cv2.imread('mask.png')
    resized_mask = cv2.resize(mask,(1280,720))

    # Load Graphics
    graphics = cv2.imread('graphics.png',cv2.IMREAD_UNCHANGED)  
    
    # class list for COCO dataset
    classes = ["person", "bicycle", "car", "motorbike", "airplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed",
              "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]  
    # Tracking
    tracker=Sort(max_age=20, min_hits=3, iou_threshold=0.3)

    # Line
    limits = [415,297,673,297]

    # Count
    count=[] 

    # Loop
    while True:
        success, frame = video.read()  # Read frame
        if not success:
            break

        
        # Put Mask
        img=cv2.bitwise_and(frame, resized_mask)
        # Put Graphics
        frame = cvzone.overlayPNG(frame, graphics, (0,0))
        # Detect
        results = model(img,verbose=False) # result list of detections
        # Create empty array
        detections=np.empty((0,5))

        # Draw
        frame = draw_boxes(frame, classes, results)

        # Tracking
        result_tracker=tracker.update(detections)

        # Draw Line
        cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

        for result in result_tracker:
            x1 , y1 , x2 , y2 , id = result
            x1 , y1 , x2 , y2 = int(x1) , int(y1) , int(x2) , int(y2)
            #print(result)
            w , h =x2 - x1 , y2 - y1
            # Draw the box
            cvzone.cornerRect(frame , (x1 , y1 , w, h) , l = 9 ,rt =2 , colorR=(255, 0, 0))
            # Draw the label
            cvzone.putTextRect(frame , f'{int(id)}' , (max(0, x1), max(35, y1)), scale = 2 , thickness = 3 , offset = 10)
            # Draw the center of object
            cx, cy = x1+w//2, y1+h//2
            cv2.circle(frame, (cx,cy), 5, (255, 0, 0), cv2.FILLED)
            # Count if cross the line
            if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[3] + 15 :
                if count.count(id) == 0:
                    count.append(id)
                    cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)
        
        # Show number
        cv2.putText(frame , str(len(count)) , (255 , 100) , cv2.FONT_HERSHEY_PLAIN , 5 , (50 , 50 ,255), 8 )
        # Show
        cv2.imshow('frame', frame)
        cv2.waitKey(1)

        # Break the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # close all windows
    cv2.destroyAllWindows()
    os.system('cls')


if __name__ == '__main__': 
    videoPath = 'cars.mp4'
    modelName = 'yolov8n.pt'
    main(videoPath, modelName)