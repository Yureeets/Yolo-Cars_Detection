import cv2
import cvzone
from ultralytics import YOLO
from sort import *

cap_front = cv2.VideoCapture('../videos/cars.mp4')

model = YOLO("../Yolo-Weights/yolov8l.pt")

classes_to_detect = ['car', 'truck', 'motorbike', 'bicycle', 'bus']
dict_of_classes = model.names

seen_id = []

count_values = 0

limits = [423, 297, 673, 297]

mask = cv2.imread('mask.png')

tracker = Sort(max_age=20, min_hits=4)

while True:
    ret, frame = cap_front.read()
    imageRegion = cv2.bitwise_and(frame, mask)
    results = model(imageRegion, stream=True)

    detections = np.empty((0, 5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = int(box.conf[0] * 100)
            # Class Name
            cls = int(box.cls[0])
            currentClass = dict_of_classes[cls]
            if currentClass in classes_to_detect and conf >= 0.3:
                # cvzone.cornerRect(frame, (x1, y1, w, h), l=9, t=2)
                # cvzone.putTextRect(frame, f'{dict_of_classes[cls]}-{conf}', (x1, y1), thickness=2, scale=1, offset=3)

                curr_array = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, curr_array))

    resultTracker = tracker.update(detections)
    cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (255, 255, 0))

    for result in resultTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1

        cx, cy = x1 + w // 2, y1 + h // 2

        if limits[0] < cx < limits[2] and limits[1] - 20 < cy < limits[1] + 20:
            if id not in seen_id:
                count_values += 1
                seen_id.append(id)


        cvzone.cornerRect(frame, (x1, y1, w, h), l=9, t=2, rt=2, colorR=(255, 0, 0))
        cvzone.putTextRect(frame, f'{id}', (x1, y1), thickness=2, scale=1, offset=3)
    cvzone.putTextRect(frame, f'Number of cars: {count_values}', (20, 700), thickness=2, scale=1, offset=3)

    key = cv2.waitKey(1)

    cv2.imshow("Camera", frame)
