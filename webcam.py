import cv2
import mediapipe as mp
import numpy as np
import threading
import tensorflow as tf
import os
import h5py
label = "Warmup...."
n_time_steps = 5
lm_list = []

action = {}
pathSave = 'DATA'
threadsold = 0.8

for dirpath, dirnames, filenames in os.walk(pathSave):
    count=0
    for dirname in dirnames:
        print(dirname)
        action[f"{count}"] = dirname
        count+=1
        

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

with h5py.File("AImodel.h5", "r") as f:
    model = tf.keras.models.load_model(f, compile=False)

cap = cv2.VideoCapture(0)

def make_landmark_timestep(results):
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm


def draw_landmark_on_image(mpDraw, results, img):
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = img.shape
        #print(id, lm)
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
    return img


def draw_class_on_image(label, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
    fontScale = 1
    fontColor = (0, 255, 0)
    thickness = 2
    lineType = 2
    cv2.putText(img, label,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    return img


def detect(model, lm_list):
    global label
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    #print(lm_list.shape)
    results = model.predict(lm_list)
    # print(results[0][0])
    
    if max(results[0]) > threadsold:
        max_index = results[0].argmax()
        label = action[f"{max_index}"]
        print(label)
        print(max_index)
    else:
        label = "Warmup...."
    return label


i = 0
warmup_frames = 60

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    i = i + 1
    if i > warmup_frames:
        #print("Start detect....")

        if results.pose_landmarks:
            c_lm = make_landmark_timestep(results)

            lm_list.append(c_lm)
            if len(lm_list) == n_time_steps:
                # predict
                t1 = threading.Thread(target=detect, args=(model, lm_list,))
                t1.start()
                lm_list = []

            img = draw_landmark_on_image(mpDraw, results, img)

    img = draw_class_on_image(label, img)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break