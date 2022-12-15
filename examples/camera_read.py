import cv2
import sys
import time

camera_id = 0
if len(sys.argv) >= 2:
    camera_id = int(sys.argv[1])
cap = cv2.VideoCapture(camera_id)

print('宽:{}'.format(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
print('高:{}'.format(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print('帧率:{}'.format(cap.get(cv2.CAP_PROP_FPS)))
# print(cap.get(cv2.CAP_PROP_FOURCC))

queue_size = 10
time_queue = []
while cap.isOpened():
  
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image = cv2.flip(image, 1)

    cur_time = time.time()
    time_queue.append(cur_time)

    if len(time_queue) > queue_size:
        last_time = time_queue.pop(0)
        fps = round(queue_size/(cur_time-last_time), 2)
        print('平均读取帧率为{}'.format(fps))
        image = cv2.putText(image, 'FPS:'+str(fps), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Camera Read', image)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
