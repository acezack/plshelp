"""
Additional libraries needed to run:
    pip install pillow
    pip install opencv-contrib-python
    pip install opencv-python
    pip install cmake
    pip install dlib
    pip install imutils

You will also need to change line 68 in 'imutils/face_utils/facealigner.py' from
    M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
to
    M = cv2.getRotationMatrix2D((int(eyesCenter[0]), int(eyesCenter[1])), angle, scale)
"""
import io

import pandas as pd
import datetime
import dlib
import math
import shutil
from time import sleep
import cv2
import os
import pickle
import numpy as np
from PIL import Image
import shutup
import imutils
from imutils import face_utils

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
fa = face_utils.FaceAligner(predictor, desiredFaceWidth=256)

#shutup.please()

cam_inp = ""
gray = 0
label_ids = []
x_train, y_labels = 0, 0
recognizer = 0
algo_choice = ""
cap = None

print(cv2.__file__)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")


# Since we are testing different algorithms
def algorithm():
    global recognizer
    global algo_choice
    print("Choose algorithm: 'f' for Fisherface, 'e' for Eigenface, or 'l' for LBPH:")
    algo_choice = ""
    while algo_choice != 'f' and algo_choice != 'e' and algo_choice != 'l':
        algo_choice = input()
        if algo_choice == 'f':
            recognizer = cv2.face.FisherFaceRecognizer_create(num_components=-1)
        elif algo_choice == 'e':
            recognizer = cv2.face.EigenFaceRecognizer_create()
        elif algo_choice == 'l':
            recognizer = cv2.face.LBPHFaceRecognizer_create()
        else:
            print("Unknown operation, please enter 'f' for Fisherface, 'e' for Eigenface, or 'l' for LBPH:")


def cam():
    print("Choose camera '0', or '1':")
    global cam_inp
    cam_inp = ""
    while isinstance(cam_inp, str):
        try:
            cam_inp = int(input())
        except ValueError:
            print("Only enter a singular number. E.g. '0', '1', or '22'.")
    print("Initialising camera, this might take a while, please hold...")

    start = datetime.datetime.now()
    global cap
    cap = cv2.VideoCapture(cam_inp)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    end = datetime.datetime.now()
    print("Camera initialised. It took {:.2f} seconds.".format((end - start).total_seconds()))


def detect_faces(image_in):
    # image_in = imutils.resize(image_in, width=800)
    gray_in = cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY)
    rects_in = detector(gray_in, 2)

    results = []
    # print(rects_in)
    for rect in rects_in:
        results.append(cv2.cvtColor(fa.align(image_in, gray_in, rect), cv2.COLOR_BGR2GRAY))
    return results, rects_in


def face_detect():
    print("This will show detected faces.")
    print("Press 'q' while the window is focused to exit.")

    global cap

    while True:
        # Capture frame-by-frame
        ret, image = cap.read()

        results, rects = detect_faces(image)

        inside_index = 0
        for i in range(len(results)):
            coord = rects[inside_index]
            x = coord.left()
            y = coord.top()
            x2 = coord.right()
            y2 = coord.bottom()
            # Draw a rectangle around detected faces
            color = (255, 0, 0)  # BGR (opencv default)
            stroke = 4  # line thickness
            end_cord_x = x2
            end_cord_y = y2
            cv2.rectangle(image, (x, y), (end_cord_x, end_cord_y), color, stroke)
            inside_index += 1
        cv2.imshow('frame', image)

        # Press q to quit/turn off camera
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cv2.destroyAllWindows()


def generate_cli():
    print("Name of subject? Can not be 'q'. Enter existing subject name to add new data to that subject.")
    name = input()
    while name == "q":
        print("'q' is not allowed.")
        print("Name of subject? (can not be 'q')")
        name = input()
    if not os.path.isdir("./images/" + name):
        os.mkdir("./images/" + name)
        os.mkdir("./processed/" + name)
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
    number = ""
    while isinstance(number, str):
        try:
            number = int(input())
        except ValueError:
            print("Only enter a singular number. E.g. '0', '1', or '22'.")
    print("How much time between each image (in seconds, e.g. '0.5')?")
    delay = ""
    while isinstance(delay, str):
        try:
            delay = float(input())
        except ValueError:
            print("Only enter a singular number. E.g. '0.2', '1.5', or '5'.")
    return name, number, delay


def generate_display_countdown(j_in):
    font = cv2.FONT_HERSHEY_SIMPLEX
    countdown = str(math.ceil(j_in / 24))
    color = (255, 255, 255)
    stroke = 10
    global cap
    retval, image = cap.read()
    w = len(image[0])
    h = len(image)
    if j_in != 1:
        cv2.putText(image, countdown, (int(w / 2) - 40, int(h / 2) + 50), font, 5, color, stroke,
                    cv2.LINE_8)
    else:
        cv2.rectangle(image, (int(w / 2 / 2), int(h / 2 / 2)),
                      (int(w / 2 + w / 2 / 2), int(h / 2 + h / 2 / 2)), color, 10000)
    cv2.imshow('frame', image)
    sleep(0.017)


def generate():
    print("This will generate training data.")
    print_subjects()

    name, number_of_images, delay = generate_cli()

    retval, image = cap.read()
    cv2.imshow('frame', image)
    for i in range(number_of_images):
        fileindex = i
        file = "./images/" + name + "/" + name + "_" + str(fileindex) + ".png"

        while os.path.isfile(file):
            fileindex += 1
            file = "./images/" + name + "/" + name + "_" + str(fileindex) + ".png"
        # Press q to quit/turn off camera
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
        if number_of_images > 1:
            if delay >= 2:
                for j in range(int(delay) * 24, 0, -1):
                    generate_display_countdown(j)
                    if cv2.waitKey(20) & 0xFF == ord('q'):
                        break
            else:
                retval, image = cap.read()
                cv2.imshow('frame', image)

                if cv2.waitKey(20) & 0xFF == ord('q'):
                    break
                sleep(delay)
        retval, image = cap.read()
        cv2.imwrite(file, image)
        print("\rImage {} of {} taken.".format(i + 1, number_of_images), end="")
    print("\nData generated.")

    cv2.destroyAllWindows()


def print_subjects():
    if not is_subjects_populated():
        print("No subjects found.")
        return
    subjects = [f.name for f in os.scandir("./images") if f.is_dir()]
    if len(subjects) == 1:
        print("Found {} subject: ".format(len(subjects)), end="")
    else:
        print("Found {} subjects: ".format(len(subjects)), end="")
    cleaned = ""
    for subject in subjects:
        cleaned = cleaned + subject + ", "
    cleaned = cleaned[0:len(cleaned) - 2] + "."
    print(cleaned)


def get_subjects():
    if not is_subjects_populated():
        return None
    return [f.name for f in os.scandir("./images") if f.is_dir()]


def is_subjects_populated():
    if [f.name for f in os.scandir("./images") if f.is_dir()] == []:
        return False
    else:
        return True


def delete():
    if not is_subjects_populated():
        print_subjects()
        return
    else:
        subjects = get_subjects()
    print("Enter 'a' to delete all training data, or 'd' to delete individual training data for a subject.")
    op = input()
    while op != 'a' and op != 'd':
        print("Please enter 'a' or 'd'.")
        op = input()

    if op == 'a':
        shutil.rmtree("./images")
        shutil.rmtree("./processed")
        print("All data deleted.")
        os.mkdir("./images")
        os.mkdir("./processed")
    elif op == 'd':
        print("Type subject's name to delete all data of that subject, or 'q' to quit.")
        print_subjects()
        deleted = False
        while not deleted:
            sub = input()
            if sub == "q":
                print("Exiting...")
                return
            for i in range(len(subjects)):
                if sub == subjects[i]:
                    shutil.rmtree("./images/" + sub)
                    shutil.rmtree("./processed/" + sub)
                    print("All data for subject '" + sub + "' deleted.")
                    deleted = True


def training_complete(x_train_in, y_labels_in, bad_input_in):
    if y_labels_in != [] and x_train_in != []:
        print("Training complete.")
        if len(bad_input_in) == 1:
            print("No face(s) detected in image: " + bad_input_in[0] + ". Disregarding input.")
        elif len(bad_input_in) != 0:
            print("No face(s) detected in images: ")
            result = ""
            for bad in bad_input_in:
                print(bad)
            print("Disregarding input.")
    else:
        print("\nshit's fucked, ABORT, ABORT\n")

    global algo_choice
    if algo_choice == 'f' and len(get_subjects()) == 1:
        print("Need more than one subject to train FisherFace model.")
        return
    recognizer.train(x_train_in, np.array(y_labels_in))

    if algo_choice == "f":
        recognizer.save("f")
    elif algo_choice == "e":
        recognizer.save("e")
    elif algo_choice == "l":
        print("should save")
        recognizer.save("l")

    # Save data label to a pickle file
    with open('labels.pickle', 'wb') as f:
        pickle.dump(label_ids, f)


def train():
    if not is_subjects_populated():
        print_subjects()
        return
    print("This will train the model on the training data.")
    # Define image directory
    image_dir = "./images"
    shutil.rmtree("./processed")
    os.mkdir("./processed")

    print("Enter 'a' to train on all images found, 'n' to select n number of images to train on, or 'q' to quit.")
    inp = input()
    while inp != 'a' and inp != 'n' and inp != 'q':
        print("Enter 'a', 'n', or 'q'.")
        inp = input()

    number = ""
    if inp == 'n':
        print("How many training images? ")
        while isinstance(number, str):
            try:
                number = int(input())
            except ValueError:
                print("Only enter a singular number. E.g. '0', '1', or '22'.")

    # Initialize training set and labels
    global x_train, y_labels, label_ids
    x_train, y_labels, label_ids, current_id, bad_input = [], [], {}, 0, []

    print_subjects()

    for root, dirs, files in os.walk(image_dir):

        index = 0
        subject = os.path.basename(root)
        if not os.path.isdir("./processed/" + subject) and subject != "images":
            os.mkdir("./processed/" + subject)

        if number == "":
            limit = len(files)
        elif number != 0 and subject != "images":
            limit = min(len(files), number)

            print("Will use", limit, "images for subject '" + subject + "'.")
        else:
            limit = len(files)
        for file in files:
            index = index + 1
            print("\rTraining on subject '{}' is {:.0%} complete.".format(subject, index / limit), end="")
            if file.endswith('png') or file.endswith('jpg') or file.endswith('jpeg'):
                path = os.path.join(root, file)
                label = os.path.basename(root)

                if label not in label_ids:
                    label_ids[label] = current_id
                    current_id += 1
                id_ = label_ids[label]

                image = cv2.imread(path)
                grays, _ = detect_faces(image)
                for cur_gray in grays:
                    cv2.imwrite("./processed/" + subject + "/" + file, cur_gray)
                    x_train.append(cur_gray)
                    y_labels.append(id_)
            if index == limit:
                break
        print()

    training_complete(x_train, y_labels, bad_input)


def recognise_init():
    # Initialise and read the previously trained model.
    model_list = ["1t_f", "2t_f", "5t_f", "10t_f",
                  "1t_e", "2t_e", "5t_e", "10t_e",
                  "1t_l", "2t_l", "5t_l", "10t_l"]
    print("Available models:\n", model_list)
    model = ""
    while model not in model_list:
        try:
            model = input()
        except FileNotFoundError:
            model = input()
    global recognizer, algo_choice
    recognizer.read("./models/" + model + "_model.yml")
    print(algo_choice)
    """
    if algo_choice == "f":
        if not os.path.isfile("f"):
            print("No model for FisherFace, run 't' in the main menu.")
            return False
        recognizer.read("f")
    elif algo_choice == "e":
        if not os.path.isfile("e"):
            print("No model for Eigenface, run 't' in the main menu.")
            return False
        recognizer.read("e")
    elif algo_choice == "l":
        if not os.path.isfile("l"):
            print("No model for LBPH, run 't' in the main menu.")
            return False
        recognizer.read("l")
    """
    # Initialise and read the names of the subjects previously trained on.
    label = {}
    with(open('labels.pickle', 'rb')) as openfile:
        try:
            test = pickle.load(openfile)
            for k, v in test.items():
                label.update({int(v): k})
        except EOFError:
            ()
    return label, model


def recognise():
    print("This will recognise previously learned subjects.")

    accuracy = {}
    """
    Dict to save the conf of each subject detected per frame.
    Will be structured like this:
    {
        'subject1': [
            75,
            73,
            73,
            76,
            67,
            ...
        ],
        'subject2': [
            99,
            75,
            87,
            67,
            ...
        ],
        ...
    }
    """
    label, model = recognise_init()
    if not label:
        return

    print("Below the average conf will be displayed in this format: ('subject' : 'average conf' : 'latest conf')")
    delay = []
    frame = 0
    test = []
    while True:
        start = datetime.datetime.now()

        # Capture frame-by-frame
        ret, image = cap.read()

        results, rects = detect_faces(image)

        per_frame = []
        inside_index = 0
        for rect in results:

            id_, conf = recognizer.predict(rect)
            try:
                name = label[id_]
            except KeyError:
                print(id_)
                continue
            # name = label[id_]
            # print(name, conf)
            per_face_in_frame = [name, conf]
            try:
                accuracy[name]
            except KeyError:
                accuracy[name] = [conf]

            accuracy[name].append(conf)

            per_frame.append(per_face_in_frame)

            out = "\rFrame: " + str(frame) + "\t"
            for subject in accuracy:
                out = out + "({} : {:.0f} : {:.0f}), ".format(
                    subject,
                    np.mean(accuracy[subject]),
                    accuracy[subject][-1])
            print("\r" + out[0:len(out) - 2] + ". ", end="")

            coord = rects[inside_index]
            x = coord.left()
            y = coord.top()
            x2 = coord.right()
            y2 = coord.bottom()

            font = cv2.FONT_HERSHEY_COMPLEX
            name = label[id_]
            color = (0, 0, 255)
            stroke = 2
            cv2.putText(image, name, (x, y - 5), font, 0.8, color, stroke, cv2.LINE_8)

            # Draw a rectangle around detected faces
            color = (255, 0, 0)  # BGR (opencv default)
            stroke = 4  # line thickness
            end_cord_x = x2
            end_cord_y = y2
            cv2.rectangle(image, (x, y), (end_cord_x, end_cord_y), color, stroke)
            inside_index += 1
        cv2.imshow('frame', image)

        # Press q to quit/turn off camera
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

        end = datetime.datetime.now()
        delay.append((end - start).total_seconds())
        # temp = [test2, (end - start).total_seconds()]
        test.append(per_frame)
        # print("Delay since last frame was {:.2f} seconds.".format((end - start).total_seconds()), end="")
        frame += 1

    # When everything done, release the capture
    cv2.destroyAllWindows()
    output = []
    print(frame)
    for index in range(frame):
        print("frame: {}\ndelay: {:.2f}".format(index, delay[index]))
        if len(test[index]) == 0:
            output.append([str(index),
                           delay[index],
                           model])
        else:
            for sub in test[index]:
                print("\tname: ", sub[0], "\n\t\tconf: ", sub[1])
                output.append([str(index),
                               delay[index],
                               model,
                               sub[0],
                               str(sub[1])])
    print(output)
    df = pd.DataFrame(output)
    print(df)
    #print(pd.DataFrame(output))
    #pd.concat([df, pd.DataFrame(output)], ignore_index=True)
    #df
    df.to_csv("output.csv", index=False, mode="a", header=False)
    # df.to_csv(output)
    # df.to_csv(df, sep=',', columns=["algo", "frame", "name", "conf"])
    print("\nMean delay was {:.2f}".format(np.mean(delay)))


def display_help():
    print("\n'g' will generate training data. You will be asked to input the name of the subject, how many "
          "pictures to capture, and the delay between each picture (measured in seconds).")
    print("'d' will let you delete all generated data, or all generated data for a specific subject.")
    print("'t' will use the previously generated data to train the model upon. This will go through and train "
          "on each picture.")
    print("'r' will use the live input from a camera combined with the trained model and try to label each "
          "face in the live feed with the corresponding label from the model.")
    print("'p' will print existing subjects.")
    print("'a' will change the algorithm to use.")
    print("'c' will change the camera input.")
    print("'f' is only used to see if the algorithm will detect a face at all, useful for checking if the input "
          "is valid.")
    print("'h' is how you got here.")
    print("'q' to quit. Does this need further explanation?")


if __name__ == "__main__":
    algorithm()
    cam()

    if not os.path.isdir('./images'):
        os.mkdir("./images")
    if not os.path.isdir('./processed'):
        os.mkdir("./processed")
    done = False
    while not done:
        print("\nEnter 'g', to generate, "
              "'d' to delete generated data, "
              "'t' to train, "
              "'r' to recognise, "
              "'p' to print existing subjects, \n"
              "'a' to change algorithm, "
              "'c' to change camera input, "
              "'f' for face detection, "
              "'h' for additional info,"
              " or 'q' to quit:")
        op = input()
        if op == 'g':
            generate()
        elif op == 'd':
            delete()
        elif op == 't':
            train()
        elif op == 'r':
            recognise()
        elif op == 'p':
            print_subjects()
        elif op == 'a':
            algorithm()
        elif op == 'c':
            cam()
        elif op == 'f':
            face_detect()
        elif op == 'h':
            display_help()
        elif op == 'q':
            done = True
        else:
            print("Unknown operation.")

    cv2.destroyAllWindows()
    print("Exiting...")
