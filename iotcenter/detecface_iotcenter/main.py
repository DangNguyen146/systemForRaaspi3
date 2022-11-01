from flask import Flask, render_template, Response, redirect, url_for, request, json
import cv2
import sqlite3
import io
import os
import shutil
import threading
from PIL import Image
import numpy as np

app = Flask(__name__)

def get_db_connection():
    conn = sqlite3.connect('./static/sql/useropendoor.db')
    conn.row_factory = sqlite3.Row
    return conn


def recieveFace(camera, face_cascade, iduser):
    sampleNum = 0
    while (True):
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if not os.path.exists('dataSet/'+iduser):
                os.makedirs('dataSet/'+iduser)

            sampleNum += 1
            cv2.imwrite('dataSet/'+iduser+'/'+iduser+'.' +
                        str(sampleNum) + ' .jpg', gray[y: y + h, x: x + w])

        print("-----------------------Da chup "+str(sampleNum)+"--------------------")
        if sampleNum > 50:
            break

def absolute_file_paths(directory):
    path = os.path.abspath(directory)
    return [entry.path for entry in os.scandir(path) if entry.is_file()]

def getImageWithId(path):
    # get the path of all the files in the folder
    imagePaths = absolute_file_paths(path)

    faces = []
    IDs = []
    for imagePath in imagePaths:
        faceImg = Image.open(imagePath).convert('L')
        faceNp = np.array(faceImg, 'uint8')

        print(faceNp)
        print(os.path.split(imagePath)[1].split('.')[1])
        # split to get ID of the image
        ID = int(os.path.split(imagePath)[1].split('.')[1])

        faces.append(faceNp)
        IDs.append(ID)
    return IDs, faces

def traninngFace(iduser):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    path = './dataSet/'+iduser+'/'
    Ids, faces = getImageWithId(path)

    # trainning
    recognizer.train(faces, np.array(Ids))

    if not os.path.exists('recognizer'):
        os.makedirs('recognizer')
    recognizer.save('./static/recognizer/trainningData.yml')


@app.route('/manageruser')
def manageruser():
    conn = get_db_connection()
    useropendoor = conn.execute('SELECT * FROM useropendoor').fetchall()
    conn.close()
    return render_template('manageruser.html', useropendoor=useropendoor)


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')



@app.route('/manageruser/adduser', methods = ['POST', 'GET'])
def adduser():
    if request.method == 'POST':
        fullname = request.form['fullname']
        age = request.form['age']
        email = request.form['email']
        if not fullname:
            return render_template('index.html')
        elif not age:
            return render_template('index.html')
        elif not email:
            return render_template('index.html')
        else:
            conn = get_db_connection()
            conn.execute('INSERT INTO useropendoor (fullname, age, email) VALUES (?, ?, ?)',
                         (fullname, age, email))
            useropendoor = conn.execute('SELECT * FROM useropendoor').fetchall()
            conn.commit()
            conn.close()
        conn = get_db_connection()
        useropendoor = conn.execute('SELECT * FROM useropendoor').fetchall()
        conn.close()
        return render_template('manageruser.html', useropendoor=useropendoor)
    else:
        return render_template('adduser.html')

@app.route('/manageruser/addimage')
def addimage():
    args = request.args
    iduser = args.get("id", default="", type=str)

    try:
        shutil.rmtree("./dataSet/"+ iduser)
    except OSError as e:
        print(e)

    return render_template("addimage.html",iduser=iduser)

@app.route('/manageruser/addimage/start')
def addimageStart():
    args = request.args
    iduser = args.get("id", default="", type=str)
    
    camera = cv2.VideoCapture()
    camera.open("http://192.168.107.140:8160/stream.mjpg")

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    recieveFace(camera, face_cascade, iduser)
    traninngFace(iduser)

    conn = get_db_connection()
    useropendoor = conn.execute('SELECT * FROM useropendoor').fetchall()
    conn.close()
    return render_template('manageruser.html', useropendoor=useropendoor)


@app.route('/delete')
def delete():
    args = request.args
    iduser = args.get("id", default="", type=str)
    conn = get_db_connection()
    useropendoor = conn.execute('DELETE FROM useropendoor WHERE id='+iduser)
    useropendoor = conn.execute('SELECT * FROM useropendoor').fetchall()
    conn.commit()
    conn.close()
    try:
        shutil.rmtree("./dataSet/"+ iduser)
    except OSError as e:
        print(e)
    return render_template('manageruser.html', useropendoor=useropendoor)


if __name__ == '__main__':
    app.run('0.0.0.0', 8000, debug=False)