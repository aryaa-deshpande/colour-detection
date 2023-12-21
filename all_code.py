# MAIN CODE THAT DOES CAR DETECTION AND SAVES IMAGES OF THE CROPPED CARS DETECTED IN A FOLDER

import cv2
import torch
import datetime


# yolov5 model for object detection
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# path to the sample video 
video_path = "input2.mp4"
cap = cv2.VideoCapture(video_path)

# output folder where the captured frames will be saved 
output_folder = 'C:\\Users\\aryaa\\OneDrive\\Desktop\\VSCode\\projects\\traffic_violation_detection\\cars2'

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # applying the model for object detection to the video frame by frame 
    results = model(frame)
    cv2.rectangle(frame,(642,151), (1274,220), (0,255,255),2) #for the zebra crossing 

    for detection in results.pred[0]:
        class_id, conf, bbox = detection[5], detection[4], detection[:4]

        # class 2 represents cars 
        if int(class_id) == 2 or int(class_id) == 3 or int(class_id) == 7: 

            # finding coordinates of all the cars detected 
            x, y, w, h = map(int, bbox)
            center_x = int(x + w / 2)
            center_y = int(y + h / 2)
            center_coordinates = (center_x,center_y)
                        
            # drawing bounding boxes for cars crossing the zebra crossing 
            # corrdinates of the zebra crossing can be edited depending on the video
            if (center_coordinates[0] in range(1500,1584) or center_coordinates[0] in range(800,1400)) :
                cv2.rectangle(frame, (x, y), (w, h), (255, 255, 0), 2)
                                
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                output_path = f"{output_folder}/{timestamp}.jpg"

                # saving just the cars around which bounding boxes are drawn
                cv2.imwrite(output_path, frame[y:h,x:w])

                       
    cv2.imshow("Car Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release() 
cv2.destroyAllWindows()


