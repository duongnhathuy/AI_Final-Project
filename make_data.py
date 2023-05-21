import cv2
import mediapipe as mp
import pandas as pd
import os
from threading import Thread
import threading

# Khởi tạo thư viện mediapipe
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils
def make_landmark_timestep(results):
    #print(results.pose_landmarks.landmark)
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm

def draw_landmark_on_image(mpDraw, results, img):
    # Vẽ các đường nối
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

    # Vẽ các điểm nút
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = img.shape
        #print(id, lm)
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
    return img

def readVideo(path):
    lm_list = []
    cap = cv2.VideoCapture(path)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            #xử lý nhận dạng
            results = pose.process(frame)
            if results.pose_landmarks:
                # Ghi nhận thông số khung xương
                lm = make_landmark_timestep(results)
                lm_list.append(lm)
                #print(lm)
                # Vẽ khung xương lên ảnh
                frame = draw_landmark_on_image(mpDraw, results, frame)
            cv2.imshow(path, frame)  
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    print(lm_list)
    cap.release()
    cv2.destroyAllWindows()
    return lm_list
    
def fileSave(path, item, file, count):
    print(path +'/' + file)
    list = readVideo(f"{path}/{item}/{file}")
    df= pd.DataFrame(list)
    if not os.path.exists(f"DATA/{item}"):
        os.makedirs(f"DATA/{item}")
    df.to_csv(f"DATA/{item}/{count['index']}.txt")
    count["index"]= count["index"]+1

def loopFolder(path, item):
    count = {"index":0}
    for file in files:
       fileSave(path,item, file, count)
       
# Đọc ảnh từ dataset
path = 'E:\\NĂM 3\\HOC KY 2\\AI\\AI_FinalProject\\CK\\dataset'
items = os.listdir(path)
folders = [f for f in items if os.path.isdir(os.path.join(path, f))]

pathSave = 'DATA'

if not os.path.exists(pathSave):
    os.mkdir(pathSave)
for item in folders:
    print(item)
    folder_path = os.path.join(path, item)
    files = os.listdir(folder_path)
    t = threading.Thread(target=loopFolder, args=(path,item,))
    t.start()
    t.join()
    
            




