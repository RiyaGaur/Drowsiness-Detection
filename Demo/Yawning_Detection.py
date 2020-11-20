from scipy.spatial import distance
from imutils import face_utils
import cv2
import dlib
from pygame import mixer
import time

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../Files/shape_predictor_68_face_landmarks.dat")
cap = cv2.VideoCapture(0)
counter = 0  # Counter to initialize yawn_timer
count_yawn = 0  # Total number of yawns
elapsed = 0
timer_start = 0
yawn_timer = 0

#Extract indexes of facial landmarks for mouth
(Start, End) = face_utils.FACIAL_LANDMARKS_IDXS['mouth']


while True:

    ret, frame = cap.read()  # Reading frames from the video (Web cam)
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        mouth = landmarks[Start:End]

        # Use hull to remove convex contour discrepencies and draw eye shape around eyes
        mouthHull = cv2.convexHull(mouth)
        # cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

        # nose = landmarks.parts()[27] ( Finding the center )
        # cv2.circle(frame, (nose.x, nose.y), 2, (255, 0, 0), 3)
        # for point in landmarks.parts():
        #     cv2.circle(frame, (point.x, point.y), 2, (255, 0, 0), 3)

        lip_up = landmarks[62]
        lip_down = landmarks[66]

        dist = distance.euclidean(lip_up, lip_down)
        # print(dist)

        if dist > 25:

            # print("Yawn")
            counter += 1
            if counter > 100:
                count_yawn += 1
                counter = 0


        if count_yawn == 0:
            timer_start = time.time()
            # print(start2)

        timer_end = time.time() - timer_start
        # print(timer_end)

        if 1800 < timer_end < 1820:
            count_yawn = 0

        if count_yawn == 5:
            mixer.init()
            mixer.music.load("../Files/alarm.wav")
            mixer.music.play()

    if ret:
        cv2.imshow("My Screen", frame)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print(count_yawn)
