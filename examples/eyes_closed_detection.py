import cv2
import sys
from os import path   
sys.path.append(path.dirname(path.dirname(__file__)))
from base_ai.face.face_mesh import FaceMesh

cap = cv2.VideoCapture(0)
face_mesh = FaceMesh()
last_eyes_closed_flag = False
eyes_closed_count = 0

while cap.isOpened():
  
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image = cv2.flip(image, 1)
    results = face_mesh.inference(image)
    image = face_mesh.draw(image, results)
    
    cur_eyes_closed_flag = face_mesh.eyes_closed_detection(results, image.shape, threshold=0.22)
    if cur_eyes_closed_flag and not last_eyes_closed_flag:
        eyes_closed_count += 1
    last_eyes_closed_flag = cur_eyes_closed_flag
    cv2.putText(image, 'eyes closed:{}'.format(eyes_closed_count), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow('Eyes Closed Detection', image)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
