import cv2
import numpy as np
import sqlite3
import os


def insertOrUpdate(id, name, yearofbirth):
    conn = sqlite3.connect(
        'E:/UIT-HKI-2022-2023/HeThongNhung/face-python-pc/face recognition-ok/datacopy.db')

    query = "SELECT * FROM People WHERE ID=" + id
    cusror = conn.execute(query)

    isRecordExist = 0

    for row in cusror:
        isRecordExist = 1

    if (isRecordExist == 0):
        query = "INSERT INTO People(Id,Name, YearOfBirth) Values(" + \
            id + ",'" + name + "','" + yearofbirth + "')"
    else:
        query = "UPDATE People SET Name='" + name + \
            "' YearOfBirth='" + yearofbirth + "' WHERE ID=" + id

    conn.execute(query)
    conn.commit()
    conn.close()


# load tv
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # captureDevice = camera

# insert to db
id = input("Nhập ID: ")
name = input("Nhập Name: ")
yearofbirth = input("Nhập năm sinh: ")
insertOrUpdate(id, name, yearofbirth)

sampleNum = 0

while (True):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if not os.path.exists('dataSet'):
            os.makedirs('dataSet')

        sampleNum += 1

        cv2.imwrite('dataSet/User.' + str(id) + '.' +
                    str(sampleNum) + ' .jpg', gray[y: y + h, x: x + w])

    cv2.imshow('frame', frame)
    cv2.waitKey(1)

    if sampleNum > 200:
        break

cap.release()
cv2.destroyAllWindows()
