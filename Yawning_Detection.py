import cv2
import dlib
import numpy as np
from pygame import mixer
import time

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
cap = cv2.VideoCapture(0)
yawn_count = 0
count_yawn = 0
elapsed = 0
start=0
# local_time = time.ctime(start)
# print("Local time:", local_time)
while True:

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        # nose = landmarks.parts()[27]
        # cv2.circle(frame, (nose.x, nose.y), 2, (255, 0, 0), 3)
        for point in landmarks.parts():
            cv2.circle(frame, (point.x, point.y), 2, (255, 0, 0), 3)
        lip_up = landmarks.parts()[62].y
        lip_down = landmarks.parts()[66].y

        if lip_down - lip_up > 10:

            # print("Yawn")
            yawn_count += 1
            if yawn_count == 1:
                start = time.time()

        # print(yawn_count)
            # mixer.init()
            # mixer.music.load("alarm.wav")
            # mixer.music.play()

        else:
            # print("Close")
            yawn_count=0
            elapsed = time.time() - start
            start=0
            if 5 < elapsed < 30:
                count_yawn += 1
                print(count_yawn)
        # print(yawn_count)
    if ret:
        cv2.imshow("My Screen", frame)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

