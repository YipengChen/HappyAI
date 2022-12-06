import cv2
import sys
from os import path   
sys.path.append(path.dirname(path.dirname(__file__)))
from base_ai.face.face_mesh import FaceMesh

cap = cv2.VideoCapture(0)
face_mesh = FaceMesh()
last_mouth_opened_flag = False
mouth_opened_count = 0

while cap.isOpened():
  
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image = cv2.flip(image, 1)
    results = face_mesh.inference(image)
    image = face_mesh.draw(image, results)
    
    cur_mouth_opened_flag = face_mesh.mouth_opened_detection(results, image.shape, threshold=0.15)
    if cur_mouth_opened_flag and not last_mouth_opened_flag:
        mouth_opened_count += 1
    last_mouth_opened_flag = cur_mouth_opened_flag

    cv2.putText(image, 'mouth opended:{}'.format(cur_mouth_opened_flag), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
    cv2.putText(image, 'mouth opended count:{}'.format(mouth_opened_count), (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
    
    cv2.imshow('Face Mesh', image)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
