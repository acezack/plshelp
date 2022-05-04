"""
Additional libraries needed to run:
    pip install pillow
    pip install opencv-contrib-python
    pip install opencv-python
"""

from time import sleep
import cv2
import os
import pickle
import numpy as np
from PIL import Image

gray = 0
label_ids = []
x_train, y_labels = 0, 0
recognizer = 0


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")

# Since we are testing different algorithms
def algorithm():
    global recognizer
    print("Choose algorithm: 'f' for Fisherface, 'e' for Eigenface, or 'l' for LBPH:")
    type = ""
    while type != 'f' and type != 'e' and type != 'l':
        type = input()
        if type == 'f':
            recognizer = cv2.face.FisherFaceRecognizer_create()
        elif type == 'e':
            recognizer = cv2.face.EigenFaceRecognizer_create()
        elif type == 'l':
            recognizer = cv2.face.LBPHFaceRecognizer_create()
        else:
            print("Unknown operation, please enter 'f' for Fisherface, 'e' for Eigenface, or 'l' for LBPH:")


def detect(cam_in):
    cap = cv2.VideoCapture(cam_in)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        global gray
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, minNeighbors=5)  # scaleFactor

        # x,y = start of the faces frame(top left)
        # w,h = width and height of frame
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

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


def generate(cam_in):
    cap = cv2.VideoCapture(cam_in)
    retval, image = cap.read()
    global gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    print("Name of subject? ")
    name = input()

    if not os.path.isdir('./images/' + name):
        os.mkdir('./images/' + name)
    else:
        print("Subject name already exists. Enter 'a' to add new images of subject, or 'q' to return.")
        cont = False
        while not cont:
            op = input()
            if op == 'a':
                cont = True
            elif op == 'q':
                return
            else:
                print("Unknown operation.")

    print("How many test images? ")
    number = int(input())
    print("How much time between each image (in seconds, e.g. '0.5')?")
    time = float(input())
    print("Capturing image in: 3", end="")
    sleep(1)
    print(", 2", end="")
    sleep(1)
    print(", 1")
    sleep(1)
    for i in range(number):
        retval, image = cap.read()
        fileindex = i
        file = "./images/" + name + "/" + name + "_" + str(fileindex) + ".png"

        while os.path.isfile(file):
            fileindex += 1
            file = "./images/" + name + "/" + name + "_" + str(fileindex) + ".png"
            # print(file, os.path.isfile(file))
        cv2.imwrite(file, image)
        cv2.imshow('frame', image)
        # Press q to quit/turn off camera
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
        if number > 1:
            sleep(time)
        print("\rImage {} of {} taken.".format(i + 1, number), end="")
    print("\nData generated.")

    cap.release()
    cv2.destroyAllWindows()


def train():
    # Define image directory
    image_dir = "./images"

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

    subjects = [f.name for f in os.scandir(image_dir) if f.is_dir()]
    print("Found {} subjects: ".format(len(subjects)), end="")
    cleaned = ""
    for subject in subjects:
        cleaned = cleaned + subject + ", "
    cleaned = cleaned[0:len(cleaned) - 2] + ".\nCommencing training..."
    print(cleaned, end="")
    for root, dirs, files in os.walk(image_dir):

        index = 0
        subject = os.path.basename(root)
        for file in files:
            index = index + 1
            print("\rTraining on subject '{}' is {:.0%} complete.".format(subject, index / len(files)), end="")
            if file.endswith('png') or file.endswith('jpg') or file.endswith('jpeg'):
                path = os.path.join(root, file)
                label = os.path.basename(root)

                if label not in label_ids:
                    label_ids[label] = current_id
                    current_id += 1
                id_ = label_ids[label]

                pil_image = Image.open(path).convert('L')
                # Resize image (if necessary)
                size = (550, 550)
                final_image = pil_image.resize(size, Image.ANTIALIAS)

                image_array = np.array(final_image, 'uint8')  # change pil_image to final_image if you resize the image

                gray = cv2.cvtColor(cv2.resize(cv2.imread(path), size), cv2.COLOR_BGR2GRAY)
                #print(len(gray))
                faces = face_cascade.detectMultiScale(gray, minNeighbors=5)

                cv2.PCA

                # Detect faces in the image
                for (x, y, w, h) in faces:
                    print(w,h)
                    roi = image_array[y:y + h, x:x + w]
                    x_train.append(roi)
                    y_labels.append(id_)
        print()
    if y_labels != [] and x_train != []:
        print("Training complete.\n")
    else:
        print("\nshit's fucked, ABORT, ABORT\n")

    recognizer.train(x_train, np.array(y_labels))
    recognizer.save("trainer.yml")
    # Save data label to a pickle file

    with open('labels.pickle', 'wb') as f:
        pickle.dump(label_ids, f)


def recognise(cam_in):
    recognizer.read('trainer.yml')

    label = {}
    with(open('labels.pickle', 'rb')) as openfile:
        try:
            test = pickle.load(openfile)
            for k, v in test.items():
                label.update({int(v): k})
        except EOFError:
            ()

    accuracy = {}
    # Initialize web cam
    cap = cv2.VideoCapture(cam_in)

    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("Below the average conf will be displayed in this format: ('subject' : 'average conf' : 'current conf')")
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        global gray
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            id_, conf = recognizer.predict(roi_gray)

            name = label[id_]
            try:
                accuracy[name]
            except KeyError:
                accuracy[name] = [conf]

            accuracy[name].append(conf)
            if conf >= 45:

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

            out = ""
            for subject in accuracy:
                out = out + "({} : {:.0f} : {:.0f}), ".format(subject,
                                                              np.mean(accuracy[subject]),
                                                              accuracy[subject][-1])
            print("\r" + out[0:len(out) - 2], end="")

        cv2.imshow('frame', frame)

        # Press q to quit/turn off camera
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    print()


def display_help():
    print("\n'g' will generate training data. You will be asked to input the name of the subject, how many "
          "pictures to capture, and the delay between each picture (measured in seconds).")
    print("'t' will use the previously generated data to train the model upon. This will go through and train "
          "on each picture.")
    print("'d' is only used to see if the algorithm will detect a face at all.")
    print("'r' will use the live input from a camera combined with the trained model and try to label each "
          "face in the live feed with the corresponding label from the model.")
    print("'h' is how you got here.")
    print("'q' to quit. Does this need further explanation?\n")


if __name__ == "__main__":
    algorithm()
    print("Choose camera '0', or '1':")

    cam = ""
    while isinstance(cam, str):
        try:
            cam = int(input())
        except ValueError:
            print("Only enter a singular number. E.g. '0', '1', or '22'.")

    done = False
    while not done:
        print("Enter 'g', to generate, 't' to train, 'd' to detect, 'r' to recognise, 'h' for additional info,"
              " or 'q' to quit:")
        op = input()
        if op == 'g':
            print("This will generate training data.")
            generate(cam)
        elif op == 't':
            print("This will train the model on the training data.")
            train()
        elif op == 'd':
            print("This will show detected faces.")
            print("Press 'q' while the window is focused to exit.")
            detect(cam)
        elif op == 'r':
            print("This will recognise previously learned subjects.")
            recognise(cam)
        elif op == 'h':
            display_help()
        elif op == 'q':
            done = True
        else:
            print("Unknown operation.")
    print("Exiting...")
