import cv2
import sys
from os import path   
sys.path.append(path.dirname(path.dirname(__file__)))
from base_ai.face.face_mesh import FaceMesh

cap = cv2.VideoCapture(0)
face_mesh = FaceMesh()

while cap.isOpened():
  
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image = cv2.flip(image, 1)
    results = face_mesh.inference(image)
    x, y, z = face_mesh.calculation_head_pose(results, image.shape)
    print(x, y, z)

    nose_location = face_mesh.get_nose_location(results, image.shape)
    p1 = (int(nose_location[0]), int(nose_location[1]))
    p2 = (int(p1[0] + y*10), int(p1[1] - x*10))

    cv2.line(image, p1, p2, (255, 0, 0), 3)

    cv2.imshow('Face Mesh', image)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
