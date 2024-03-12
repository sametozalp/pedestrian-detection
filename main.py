import cv2
import imutils

cap = cv2.VideoCapture("pedestrian-detection/pedestrian.mp4")

while True:
    ret, frame = cap.read()
    
    if ret == False:
        break
    
    frame = imutils.resize(frame, 400)
    
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(hog.getDefaultPeopleDetector())
    
    coordinate, _ = hog.detectMultiScale(frame, winStride=(4,4), padding=(8,8), scale=1.05)
    
    color = (0,255,0)
    thickness = 5
    
    for(x,y,w,h) in coordinate:
        cv2.rectangle(frame, (x,y), (x+w,y+h), color, thickness)
    
    cv2.imshow("frame", frame)
    
    cv2.waitKey(30)
    