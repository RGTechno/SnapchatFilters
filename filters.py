import cv2
import numpy as np
import pandas as pd
# import os

cap = cv2.VideoCapture(0)

eyes_cascade = cv2.CascadeClassifier(
    "./Train/third-party/frontalEyes35x16.xml")
nose_cascade = cv2.CascadeClassifier("./Train/third-party/Nose18x15.xml")

# image_path = "./Train/Jamie_Before.jpg"

# img_before = cv2.imread(image_path)

# test_img = cv2.imread("./Test/Before.png")

# df = pd.read_csv("./Test/sample.csv")

# sample_img = df.values
# print(sample_img)

# cv2.imshow("Sample",sample_img)

# img_before = cv2.cvtColor(img_before,cv2.COLOR_BGR2BGRA)

# final_image = img_before.copy()
# final_image = test_img.copy()

glasses = cv2.imread("./Train/glasses.png", cv2.IMREAD_UNCHANGED)
mustache = cv2.imread("./Train/mustache.png", cv2.IMREAD_UNCHANGED)

# glasses = cv2.cvtColor(glasses,cv2.COLOR_BGR2RGBA)

# print(glasses)

# cv2.imshow("Glasses",glasses)
# cv2.imshow("mustache",mustache)

# print("Image Before shape: ", test_img.shape)
# print("Glasses shape: ", glasses.shape)

skip = 0

while True:

    ret, frame = cap.read()

    # print(frame.shape)

    if ret == False:
        continue

    eyes = eyes_cascade.detectMultiScale(frame, 1.3, 5)
    nose = nose_cascade.detectMultiScale(frame, 1.3, 5)

    for x, y, w, h in eyes:
        # cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        if skip % 10 == 0:
            glasses = cv2.resize(glasses, (w, h))
        for i in range(glasses.shape[0]):
            for j in range(glasses.shape[1]):
                if glasses[i, j, 3] > 0:
                    frame[y+i, x+j, :] = glasses[i, j, :-1]

    for x, y, w, h in nose:
        # cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        if skip % 10 == 0:
            mustache = cv2.resize(mustache, (w+20, h-20))
        yoffset = 45
        xoffset = 10
        for i in range(mustache.shape[0]):
            for j in range(mustache.shape[1]):
                if mustache[i, j, 3] > 0:
                    frame[y+yoffset+i, x-xoffset +j, :] = mustache[i, j, :-1]

    cv2.imshow("Image After", frame)
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


# cv2.imshow("Image Before",img_before)

# print("Glasses shape: ",glasses.shape)


# print(eyes)
