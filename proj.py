# project

# import libraries
import cv2
import numpy as np 
import argparse
import time

# Import YOLO weights
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg" )

classes = []

with open("coco.names", 'r') as f:
    classes = [line.strip() for line in f.readlines()]
    
layer_names = net.getLayerNames()    
outer_layers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]

img = cv2.imread('1.jpg',1)
img = cv2.resize(img,(416,416))

# Converting img into blob to extract features from it
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416,), (0, 0, 0), True, crop = False)

# give blob_image as input 
net.setInput(blob)
outs = net.forward(outer_layers)
height, width, channel = img.shape

# Iterating over all the grid cells of the ouput and chechking confidence
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            
            # Drawing Rectangle 
            cv2.rectangle(img, (int(center_x - w/2), int(center_y - h/2)),
                          (int(center_x + w/2), int(center_y + h/2), (0, 255, 0), 5)
                          
while(1):
    cv2.imshow('image',img)
    if cv2.waitKey(0) & 0xFF == 27:
        break
cv2.destroyAllWindows()
