from flask import Flask, render_template, Response, redirect, url_for, request
import cv2
import io

from picamera.array import PiRGBArray
from picamera import PiCamera

import numpy as np
from PIL import Image
import os
import sqlite3

app = Flask(__name__)


def get_db_connection():
    conn = sqlite3.connect('http://127.0.0.1:8000/static/recognizer/trainningData.yml')
    conn.row_factory = sqlite3.Row
    return conn


def gen_frames(camera, face_cascade):  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result



@app.route('/')
def index():
    camera = cv2.VideoCapture()

    camera.open("http://192.168.107.140:8160/stream.mjpg")
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return Response(gen_frames(camera, face_cascade), mimetype='multipart/x-mixed-replace; boundary=frame')

# get data from sqlite by ID
def getProfile(id):
    conn = get_db_connection()
    query = "SELECT * FROM People WHERE ID="+str(id)
    cursor = conn.execute(query)
    conn.close()

    profile = None

    for row in cursor:
        profile = row

    conn.close()
    return profile



def nameFaceCamera(camera, face_cascade):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('http://192.168.107.22:8000/static/recognizer/trainningData.yml')
    fontface = cv2.FONT_HERSHEY_SIMPLEX
    while (True):
        # camera read
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            roi_gray = gray[y:y+h, x:x+w]

            id, confidence = recognizer.predict(roi_gray)

            if confidence < 90:
                profile = getProfile(id)
                if (profile != None):
                    # cv2.putText(frame, id, (10,30), fontface, 1, (0, 0, 255), 2)
                    cv2.putText(frame, str(
                        profile[1])+", Year: " + str(profile[2]), (x+10, y+h+30), fontface, 1, (0, 255, 0), 2)

            else:
                cv2.putText(frame, "No", (x+10, y+h+30),
                            fontface, 1, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

    return Response(gen_frames(camera, face_cascade), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/face')
def video_face():
    camera = cv2.VideoCapture()

    camera.open("http://192.168.107.140:8160/stream.mjpg")
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return Response(nameFaceCamera(camera, face_cascade), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run('0.0.0.0', 5050, debug=False)