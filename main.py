from time import sleep

import cv2
import os
import pickle
import numpy as np
from PIL import Image

label_ids = []
x_train, y_labels = 0, 0
recognizer = 0
face_cascade = cv2.CascadeClassifier(
    'C:\\Users\\adjakobs\\Desktop\\test\\venv\\lib\\site-packages\\cv2\\data\\haarcascade_frontalface_alt2.xml')


def detect(cam):
    cap = cv2.VideoCapture(cam)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, minNeighbors=5)  # scaleFactor

        # x,y = start of the faces frame(top left)
        # w,h = width and height of frame
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            # img_item = 'my_image.png'
            # cv2.imwrite(img_item, roi_gray)

            # Draw a rectangle around detected faces
            color = (255, 0, 0)  # BGR (opencv default)
            stroke = 2  # line thickness
            end_cord_x = x + w
            end_cord_y = y + h
            cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

        # Display the resulting frame
        cv2.imshow('frame', frame)

        # Press q to quit/turn off camera
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def train():
    # Define image directory
    image_dir = "C:\\Users\\adjakobs\\Desktop\\test\\images"

    # Initialize training set and labels
    global x_train, y_labels, label_ids

    x_train = []
    y_labels = []

    # Create a dictionary to convert labels into numeric
    current_id = 0
    label_ids = {}

    # ret, frame = cap.read()
    global gray
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # print(gray)
    index = 0
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith('png') or file.endswith('jpg') or file.endswith('jpeg'):
                path = os.path.join(root, file)
                label = os.path.basename(root)  # os.path.dirname(path)
                # path = path.replace('\\', '/')
                # print(label, path)

                if label not in label_ids:
                    label_ids[label] = current_id
                    current_id += 1
                id_ = label_ids[label]
                print(current_id)

                pil_image = Image.open(path).convert('L')
                # Resize image (if necessary)
                size = (550, 550)
                final_image = pil_image.resize(size, Image.ANTIALIAS)

                image_array = np.array(final_image, 'uint8')  # change pil_image to final_image if you resize the image

                gray = cv2.cvtColor(cv2.resize(cv2.imread(path), size), cv2.COLOR_BGR2GRAY)

                faces = face_cascade.detectMultiScale(gray, minNeighbors=5)

                # Detect faces in the image
                for (x, y, w, h) in faces:
                    roi = image_array[y:y + h, x:x + w]
                    x_train.append(roi)
                    y_labels.append(id_)
        index = index + 1
    if y_labels != [] and x_train != []:
        print(y_labels)
        print(x_train)
    else:
        print("shit's fucked")

    recognizer.train(x_train, np.array(y_labels))
    recognizer.save("trainer.yml")
    # Save data label to a pickle file

    with open('labels.pickle', 'wb') as f:
        pickle.dump(label_ids, f)


def algorithm():
    global recognizer
    print("Choose method. 'f' for Fisherface, 'e' for Eigenface, or 'l' for LBPH:")
    type = input()
    if type == 'f':
        recognizer = cv2.face.FisherFaceRecognizer_create()
    elif type == 'e':
        recognizer = cv2.face.EigenFaceRecognizer_create()
    elif type == 'l':
        recognizer = cv2.face.LBPHFaceRecognizer_create()


def recognise(cam):
    recognizer.read('trainer.yml')

    # Inverse label
    # label = {v: k for k, v in label_ids.items()}
    # for index in label:
    #    print(label[index])
    # print(label[0], label[1], label[2])
    label = {}
    with(open('labels.pickle', 'rb')) as openfile:
        try:
            test = pickle.load(openfile)
            for k, v in test.items():
                label.update({int(v): k})
        except EOFError:
            ()

    for index in label:
        print(label[index])
    print(label)
    # Initialize web cam
    cap = cv2.VideoCapture(0)

    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        global gray
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, minNeighbors=5)
        print(faces)
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            id_, conf = recognizer.predict(roi_gray)

            # img_item = 'my_image.png'
            # cv2.imwrite(img_item, roi_gray)
            print(conf)
            if conf >= 45:
                print(id_)
                print(label[id_])

                # Draw labels on the Haar Cascade rectangle
                font = cv2.FONT_HERSHEY_SIMPLEX
                name = label[id_]
                color = (0, 0, 255)
                stroke = 2
                cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_8)

            # Draw a rectangle around detected faces
            color = (255, 0, 0)  # BGR (opencv default)
            stroke = 4  # line thickness
            end_cord_x = x + w
            end_cord_y = y + h
            cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

            ## Optional: create rectangles for eyes
            # = eye_cascade.detectMultiScale(roi_gray)
            # for (ex, ey, ew, eh) in eyes:
            #    cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)

        # Display the resulting frame
        cv2.imshow('frame', frame)

        # Press q to quit/turn off camera
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def generate(cam, name, number):
    cap = cv2.VideoCapture(cam)
    retval, image = cap.read()
    global gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("Capturing image in: 3", end="")
    sleep(0.5)
    print(", 2", end="")
    sleep(0.5)
    print(", 1")
    sleep(0.5)

    if not os.path.isdir('./images/' + name):
        os.mkdir('./images/' + name)
    for i in range(number):
        retval, image = cap.read()
        sleep(0.2)

        fileindex = i
        file = "./images/" + name + "/" + name + "_" + str(fileindex) + ".png"

        while os.path.isfile(file):
            fileindex = fileindex + 1
            file = "./images/" + name + "/" + name + "_" + str(fileindex) + ".png"
            print(file, os.path.isfile(file))
        cv2.imwrite(file, image)
    print("Data generated.")
    # print(image)


if __name__ == "__main__":
    algorithm()

    print("Choose camera '0', or '1':")
    cam = int(input())
    print("Center your face in front of camera.")
    retval, image = cv2.VideoCapture(cam).read()
    print(image)
    print()
    global gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    done = False
    while not done:
        print("Enter 'g', to generate, 't' to train, 'd' to detect, 'r' to recognise, or 'q' to quit:")
        op = input()
        if op == 'g':
            print("This will generate training data.")
            print("Name of subject? ")
            name = input()
            print("How many test images? ")
            number = int(input())
            generate(cam, name, number)
        elif op == 't':
            print("This will train the model on the training data.")
            train()
        elif op == 'd':
            print("This will show detected faces.")
            detect(cam)
        elif op == 'r':
            print("This will recognise previously learned subjects.")
            recognise(cam)
        elif op == 'q':
            done = True
        else:
            print("Unknown operation.")
    print("Exiting...")
