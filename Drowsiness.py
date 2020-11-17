from scipy.spatial import distance
from imutils import face_utils
import cv2
import dlib
from pygame import mixer
import time


#Initialize Pygame and load music
mixer.init()
mixer.music.load('Files/alarm.wav')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("Files/shape_predictor_68_face_landmarks.dat")
cap = cv2.VideoCapture(0)

#Extract indexes of facial landmarks for mouth
(Start, End) = face_utils.FACIAL_LANDMARKS_IDXS['mouth']

counter = 0  # Counter to initialize yawn_timer
count_yawn = 0  # Total number of yawns
elapsed = 0
timer_start = 0
yawn_timer = 0



#Extract indexes of facial landmarks for left and right eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

#This function calculates and return eye aspect ratio
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])

    ear = (A+B) / (2*C)
    return ear

#Minimum threshold of eye aspect ratio below which alarm is triggerd
EYE_ASPECT_RATIO_THRESHOLD = 0.3
time_start=0
#Minimum consecutive frames for which eye ratio is below threshold for alarm to be triggered
EYE_ASPECT_RATIO_CONSEC_FRAMES = 100

#COunts no. of consecutuve frames below threshold value
COUNTER = 0


while(True):

    #Read each frame and flip it, and convert to grayscale
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detect facial points through detector function
    faces = detector(gray, 0)

    #Detect faces through haarcascade_frontalface_default.xml
    # face_rectangle = face_cascade.detectMultiScale(gray, 1.3, 5)

    #Draw rectangle around each face detected
    # for (x,y,w,h) in face_rectangle:
        # cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    #Detect facial points
    for face in faces:

        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        #Get array of coordinates of leftEye and rightEye
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        #Calculate aspect ratio of both eyes
        leftEyeAspectRatio = eye_aspect_ratio(leftEye)
        rightEyeAspectRatio = eye_aspect_ratio(rightEye)

        eyeAspectRatio = (leftEyeAspectRatio + rightEyeAspectRatio) / 2

        #Use hull to remove convex contour discrepencies and draw eye shape around eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        # cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        # cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        #Detect if eye aspect ratio is less than threshold
        if(eyeAspectRatio < EYE_ASPECT_RATIO_THRESHOLD):
            COUNTER += 1
            #If no. of frames is greater than threshold frames,
            if COUNTER >= EYE_ASPECT_RATIO_CONSEC_FRAMES:
                mixer.music.play(-1)
                cv2.putText(frame, "You are Sleepy", (150,300), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0,0,255), 2)

        else:
            mixer.music.stop()
            COUNTER = 0



        # YAWNING

        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        mouth = landmarks[Start:End]

        # Use hull to remove convex contour discrepencies and draw eye shape around eyes
        mouthHull = cv2.convexHull(mouth)

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
                print(count_yawn)

        if count_yawn == 0:
            timer_start = time.time()
            # print(start2)

        timer_end = time.time() - timer_start
        # print(timer_end)

        if 1800 < timer_end < 1820:
            count_yawn = 0

        if count_yawn == 5:
            mixer.music.play(-1)
            cv2.putText(frame, "You are Yawning", (150, 300), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 2)
        else:
            mixer.music.stop()

    #Show video feed
    cv2.imshow('Video', frame)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break

cap.release()
cv2.destroyAllWindows()

