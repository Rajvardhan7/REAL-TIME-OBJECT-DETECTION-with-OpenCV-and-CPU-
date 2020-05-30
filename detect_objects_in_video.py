# project- real time object detection

# import libraries
import cv2
import numpy as np 
import argparse
import time
import math

# Import YOLO weights
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg" )

classes = []

with open("coco.names", 'r') as f:
    classes = [line.strip() for line in f.readlines()]
    
layer_names = net.getLayerNames()    
outer_layers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]

# Read Image
img = cv2.imread('1.jpg',1)
img = cv2.resize(img,(416,416))

# Converting img into blob to extract features from it
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416,), (0, 0, 0), True, crop = False)

# give blob_image as input 
net.setInput(blob)
outs = net.forward(outer_layers)
height, width, channel = img.shape

# Iterating over all the grid cells of the ouput and chechking confidence
class_ids = []
confidences = []
boxes = []

# This loop will will extract features from image and store them in above 
# defined empty lists
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
            
            # Rectangle Coordinates
            x = int(center_x - w/2)
            y = int(center_y - h/2)
            
            # Saving values for later use
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)
            
# Non-max suppression
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)            

# For differnt colored boxes
colors = np.random.uniform(0, 255, size = (len(classes), 3))


for i in range(len(boxes)):
    x, y, h, w = boxes[i]
    label = classes[class_ids[i]]
    confidence = confidences[i]
    print(label)
    
    # chossing only those boxes left after NMS
    if i in indexes:
        # choosing color
        color = colors[i]
        
        # Drawing Rectangle 
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        
        # text over rectangle
        font = cv2.FONT_HERSHEY_PLAIN
        text = str(round(confidence, 2)) + " " +  label
        cv2.putText(img, text, (x, y-5), font, 1, color, 1)
        
    

            
                          
while(1):
    cv2.imshow('image',img)
    if cv2.waitKey(0) & 0xFF == 27:
        break
cv2.destroyAllWindows()

