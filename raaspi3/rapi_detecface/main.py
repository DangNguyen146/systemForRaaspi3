from flask import Flask, Response
import cv2

from PIL import Image
import os
import sqlite3


app = Flask(__name__)


def get_db_connection():  # open file and read database
    conn = sqlite3.connect('./sql/useropendoor.db')
    conn.row_factory = sqlite3.Row
    return conn


def gen_frames(camera, face_cascade):  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        # Look for faces in the image using the loaded cascade file
        faces = face_cascade.detectMultiScale(gray)

        for (x, y, w, h) in faces:
            # Draw a rectangle around every found face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        if not success:
            break
        else:
            # display a frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


# get data from sqlite by ID
def getProfile(id):
    conn = sqlite3.connect('./sql/useropendoor.db')
    query = "SELECT * FROM useropendoor WHERE ID="+str(id)
    cursor = conn.execute(query)

    profile = None

    for row in cursor:
        profile = row

    conn.close()
    return profile


def nameFaceCamera(camera, face_cascade):
    # use Local Binary Patterns Histograms
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('./recognizer/trainningData.yml')  # Load a trainer file
    fontface = cv2.FONT_HERSHEY_SIMPLEX
    while (True):
        # camera read
        ret, frame = camera.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

        # Look for faces in the image using the loaded cascade file
        faces = face_cascade.detectMultiScale(gray)

        for (x, y, w, h) in faces:  # Draw a rectangle around every found face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            roi_gray = gray[y:y+h, x:x+w]

            id, confidence = recognizer.predict(roi_gray)
            print(id)

            if confidence < 90:  # Check if confidence is less them 90 ==> "0" is perfect match
                profile = getProfile(id)
                print(profile)
                if (profile != None):
                    # cv2.putText(frame, id, (10,30), fontface, 1, (0, 0, 255), 2)
                    cv2.putText(frame, str(
                        confidence+"%, " + profile[1])+", Age: " + str(profile[2]), (x+10, y+h+30), fontface, 1, (0, 255, 0), 2)

            else:
                cv2.putText(frame, "No", (x+10, y+h+30),
                            fontface, 1, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

    return Response(gen_frames(camera, face_cascade), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    camera = cv2.VideoCapture()

    # initialize the camera and grab a reference to the raw camera capture
    camera.open("http://192.168.107.140:8160/stream.mjpg")
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")  # Load a cascade file for detecting faces
    return Response(gen_frames(camera, face_cascade), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/face')
def video_face():
    camera = cv2.VideoCapture()

    # initialize the camera and grab a reference to the raw camera capture
    camera.open("http://192.168.107.140:8160/stream.mjpg")
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")  # Load a cascade file for detecting faces
    return Response(nameFaceCamera(camera, face_cascade), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run('0.0.0.0', 5050, debug=False)
