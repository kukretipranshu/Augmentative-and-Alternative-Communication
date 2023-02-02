from cvzone.HandTrackingModule import HandDetector
import string
import numpy as np
import os
import cv2
import imutils
import mediapipe as mp
import uuid

# Setting up hand points variable and landmark drawing variable

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# Processing the input image
def process_image(img):

    # Converting the input to grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(gray_image)

    # Returning the detected hands to call function
    return results

# Drawing landmark connections
def draw_hand_connections(img, results):
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape

                # Finding the coordinates of each landmark
                cx, cy = int(lm.x * w), int(lm.y * h)

                # Printing each landmark ID and coordinates
                # on the terminal
                print(id, cx, cy)

                # Drawing the landmark connections
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

        return img


def main():

    # Creating Necessary Directory Structure for the Image Dataset

    if not os.path.exists("dataset"): # if dataset directory doesn't exist 
        os.makedirs("dataset") # create dataset directory
    if not os.path.exists("dataset/train_set"): # if dataset/train_set directory doesn't exist 
        os.makedirs("dataset/train_set") # create dataset/train_set directory
    if not os.path.exists("dataset/test_set"): # if dataset/test_set directory doesn't exist 
        os.makedirs("dataset/test_set") # create dataset/test_set directory
        
    # Creating Folders for Numeric / Numbers
    for i in range(10):
        if not os.path.exists("dataset/train_set/" + str(i)):
            os.makedirs("dataset/train_set/"+str(i))
        if not os.path.exists("dataset/test_set/" + str(i)):
            os.makedirs("dataset/test_set/"+str(i))


    # Creating Folders for Alphabets A-Z
    for i in string.ascii_uppercase:
        if not os.path.exists("dataset/train_set/" + i):
            os.makedirs("dataset/train_set/"+i)
        if not os.path.exists("dataset/test_set/" + i):
            os.makedirs("dataset/test_set/"+i)
        

    # Initializing things
    cap = cv2.VideoCapture(0)
    flag = -2 

    # Creating the Dataset 

    mode = 'train_set'
    directory = 'dataset/'+mode+'/'

    while True:

        # frame = (Flipped Image) and non_flipped = (Original Image)
        _, frame = cap.read()


        # flipping the image around Y-axis
        frame = cv2.flip(frame, 1)

        # Getting count of each Number and Alphabet
        count = {
                'zero': len(os.listdir(directory+"/0")),
                'one': len(os.listdir(directory+"/1")),
                'two': len(os.listdir(directory+"/2")),
                'three': len(os.listdir(directory+"/3")),
                'four': len(os.listdir(directory+"/4")),
                'five': len(os.listdir(directory+"/5")),
                'six': len(os.listdir(directory+"/6")),
                'seven': len(os.listdir(directory+"/7")),
                'eight': len(os.listdir(directory+"/8")),
                'nine': len(os.listdir(directory+"/9")),
                'a': len(os.listdir(directory+"/A")),
                'b': len(os.listdir(directory+"/B")),
                'c': len(os.listdir(directory+"/C")),
                'd': len(os.listdir(directory+"/D")),
                'e': len(os.listdir(directory+"/E")),
                'f': len(os.listdir(directory+"/F")),
                'g': len(os.listdir(directory+"/G")),
                'h': len(os.listdir(directory+"/H")),
                'i': len(os.listdir(directory+"/I")),
                'j': len(os.listdir(directory+"/J")),
                'k': len(os.listdir(directory+"/K")),
                'l': len(os.listdir(directory+"/L")),
                'm': len(os.listdir(directory+"/M")),
                'n': len(os.listdir(directory+"/N")),
                'o': len(os.listdir(directory+"/O")),
                'p': len(os.listdir(directory+"/P")),
                'q': len(os.listdir(directory+"/Q")),
                'r': len(os.listdir(directory+"/R")),
                's': len(os.listdir(directory+"/S")),
                't': len(os.listdir(directory+"/T")),
                'u': len(os.listdir(directory+"/U")),
                'v': len(os.listdir(directory+"/V")),
                'w': len(os.listdir(directory+"/W")),
                'x': len(os.listdir(directory+"/X")),
                'y': len(os.listdir(directory+"/Y")),
                'z': len(os.listdir(directory+"/Z"))
                }
        

        # Printing the count in each set to the screen

        cv2.putText(frame, "0 : "+str(count['zero']), (10, 70), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "1 : "+str(count['one']), (10, 80), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "2 : "+str(count['two']), (10, 90), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "3 : "+str(count['three']), (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "4 : "+str(count['four']), (10, 110), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "5 : "+str(count['five']), (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "6 : "+str(count['six']), (10, 130), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "7 : "+str(count['seven']), (10, 140), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "8 : "+str(count['eight']), (10, 150), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "8 : "+str(count['nine']), (10, 160), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "a : "+str(count['a']), (10, 170), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "b : "+str(count['b']), (10, 180), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "c : "+str(count['c']), (10, 190), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "d : "+str(count['d']), (10, 200), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "e : "+str(count['e']), (10, 210), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "f : "+str(count['f']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "g : "+str(count['g']), (10, 230), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "h : "+str(count['h']), (10, 240), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "i : "+str(count['i']), (10, 250), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "j : "+str(count['j']), (10, 260), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "k : "+str(count['k']), (10, 270), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "l : "+str(count['l']), (10, 280), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "m : "+str(count['m']), (10, 290), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "n : "+str(count['n']), (10, 300), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "o : "+str(count['o']), (10, 310), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "p : "+str(count['p']), (10, 320), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "q : "+str(count['q']), (10, 330), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "r : "+str(count['r']), (10, 340), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "s : "+str(count['s']), (10, 350), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "t : "+str(count['t']), (10, 360), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "u : "+str(count['u']), (10, 370), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "v : "+str(count['v']), (10, 380), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "w : "+str(count['w']), (10, 390), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "x : "+str(count['x']), (10, 400), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "y : "+str(count['y']), (10, 410), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "z : "+str(count['z']), (10, 420), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        
        # Drawing Rectangle
        cv2.rectangle(frame, (330, 10), (630, 310), (0,0,255) ,2)

        # Setting Region of Interest
        region_of_interest = frame[12:307, 332:628]

        new_image = region_of_interest

        results = process_image(region_of_interest)
        draw_hand_connections(new_image, results)

        # Applying Canny Edge Detection
        new_image = cv2.Canny(new_image,80,150)
        new_image = cv2.resize(new_image, (300,300))

        non_flipped = cv2.flip(new_image, 1)
        

        cv2.imshow("Frame", frame)
        cv2.imshow("Canny Edge Window with Hand Landmarks", new_image)

        # cv2.imshow("non-flipped", non_flipped)

        flag = cv2.waitKey(10)
        if flag & 0xFF == 27: # ordinal value of Esc key
            break
        if flag & 0xFF == ord('0'):
            cv2.imwrite(directory+'0/'+'0.'+'{}.jpg'.format(str(uuid.uuid1())), new_image)
            cv2.imwrite(directory+'0/'+'0.'+'{}.jpg'.format(str(uuid.uuid1())), non_flipped)
        if flag & 0xFF == ord('1'):
            cv2.imwrite(directory+'1/'+'1.'+'{}.jpg'.format(str(uuid.uuid1())), new_image)
            cv2.imwrite(directory+'1/'+'1.'+'{}.jpg'.format(str(uuid.uuid1())), non_flipped)
        if flag & 0xFF == ord('2'):
            cv2.imwrite(directory+'2/'+'2.'+'{}.jpg'.format(str(uuid.uuid1())), new_image)
            cv2.imwrite(directory+'2/'+'2.'+'{}.jpg'.format(str(uuid.uuid1())), non_flipped)      
        if flag & 0xFF == ord('3'):
            cv2.imwrite(directory+'3/'+'3.'+'{}.jpg'.format(str(uuid.uuid1())), new_image)
            cv2.imwrite(directory+'3/'+'3.'+'{}.jpg'.format(str(uuid.uuid1())), non_flipped)
        if flag & 0xFF == ord('4'):
            cv2.imwrite(directory+'4/'+'4.'+'{}.jpg'.format(str(uuid.uuid1())), new_image)
            cv2.imwrite(directory+'4/'+'4.'+'{}.jpg'.format(str(uuid.uuid1())), non_flipped)
        if flag & 0xFF == ord('5'):
            cv2.imwrite(directory+'5/'+'5.'+'{}.jpg'.format(str(uuid.uuid1())), new_image)
            cv2.imwrite(directory+'5/'+'5.'+'{}.jpg'.format(str(uuid.uuid1())), non_flipped)
        if flag & 0xFF == ord('6'):
            cv2.imwrite(directory+'6/'+'6.'+'{}.jpg'.format(str(uuid.uuid1())), new_image)
            cv2.imwrite(directory+'6/'+'6.'+'{}.jpg'.format(str(uuid.uuid1())), non_flipped)
        if flag & 0xFF == ord('7'):
            cv2.imwrite(directory+'7/'+'7.'+'{}.jpg'.format(str(uuid.uuid1())), new_image)
            cv2.imwrite(directory+'7/'+'7.'+'{}.jpg'.format(str(uuid.uuid1())), non_flipped)
        if flag & 0xFF == ord('8'):
            cv2.imwrite(directory+'8/'+'8.'+'{}.jpg'.format(str(uuid.uuid1())), new_image)
            cv2.imwrite(directory+'8/'+'8.'+'{}.jpg'.format(str(uuid.uuid1())), non_flipped)
        if flag & 0xFF == ord('9'):
            cv2.imwrite(directory+'9/'+'9.'+'{}.jpg'.format(str(uuid.uuid1())), new_image)
            cv2.imwrite(directory+'9/'+'9.'+'{}.jpg'.format(str(uuid.uuid1())), non_flipped)
        if flag & 0xFF == ord('a'):
            cv2.imwrite(directory+'A/'+'A.'+'{}.jpg'.format(str(uuid.uuid1())), new_image)
            cv2.imwrite(directory+'A/'+'A.'+'{}.jpg'.format(str(uuid.uuid1())), non_flipped)
        if flag & 0xFF == ord('b'):
            cv2.imwrite(directory+'B/'+'B.'+'{}.jpg'.format(str(uuid.uuid1())), new_image)
            cv2.imwrite(directory+'B/'+'B.'+'{}.jpg'.format(str(uuid.uuid1())), non_flipped)
        if flag & 0xFF == ord('c'):
            cv2.imwrite(directory+'C/'+'C.'+'{}.jpg'.format(str(uuid.uuid1())), new_image)
            cv2.imwrite(directory+'C/'+'C.'+'{}.jpg'.format(str(uuid.uuid1())), non_flipped)
        if flag & 0xFF == ord('d'):
            cv2.imwrite(directory+'D/'+'D.'+'{}.jpg'.format(str(uuid.uuid1())), new_image)
            cv2.imwrite(directory+'D/'+'D.'+'{}.jpg'.format(str(uuid.uuid1())), non_flipped)
        if flag & 0xFF == ord('e'):
            cv2.imwrite(directory+'E/'+'E.'+'{}.jpg'.format(str(uuid.uuid1())), new_image)
            cv2.imwrite(directory+'E/'+'E.'+'{}.jpg'.format(str(uuid.uuid1())), non_flipped)
        if flag & 0xFF == ord('f'):
            cv2.imwrite(directory+'F/'+'F.'+'{}.jpg'.format(str(uuid.uuid1())), new_image)
            cv2.imwrite(directory+'F/'+'F.'+'{}.jpg'.format(str(uuid.uuid1())), non_flipped)
        if flag & 0xFF == ord('g'):
            cv2.imwrite(directory+'G/'+'G.'+'{}.jpg'.format(str(uuid.uuid1())), new_image)
            cv2.imwrite(directory+'G/'+'G.'+'{}.jpg'.format(str(uuid.uuid1())), non_flipped)
        if flag & 0xFF == ord('h'):
            cv2.imwrite(directory+'H/'+'H.'+'{}.jpg'.format(str(uuid.uuid1())), new_image)
            cv2.imwrite(directory+'H/'+'H.'+'{}.jpg'.format(str(uuid.uuid1())), non_flipped)
        if flag & 0xFF == ord('i'):
            cv2.imwrite(directory+'I/'+'I.'+'{}.jpg'.format(str(uuid.uuid1())), new_image)
            cv2.imwrite(directory+'I/'+'I.'+'{}.jpg'.format(str(uuid.uuid1())), non_flipped)
        if flag & 0xFF == ord('j'):
            cv2.imwrite(directory+'J/'+'J.'+'{}.jpg'.format(str(uuid.uuid1())), new_image)
            cv2.imwrite(directory+'J/'+'J.'+'{}.jpg'.format(str(uuid.uuid1())), non_flipped)
        if flag & 0xFF == ord('k'):
            cv2.imwrite(directory+'K/'+'K.'+'{}.jpg'.format(str(uuid.uuid1())), new_image)
            cv2.imwrite(directory+'K/'+'K.'+'{}.jpg'.format(str(uuid.uuid1())), non_flipped)
        if flag & 0xFF == ord('l'):
            cv2.imwrite(directory+'L/'+'L.'+'{}.jpg'.format(str(uuid.uuid1())), new_image)
            cv2.imwrite(directory+'L/'+'L.'+'{}.jpg'.format(str(uuid.uuid1())), non_flipped)
        if flag & 0xFF == ord('m'):
            cv2.imwrite(directory+'M/'+'M.'+'{}.jpg'.format(str(uuid.uuid1())), new_image)
            cv2.imwrite(directory+'M/'+'M.'+'{}.jpg'.format(str(uuid.uuid1())), non_flipped)
        if flag & 0xFF == ord('n'):
            cv2.imwrite(directory+'N/'+'N.'+'{}.jpg'.format(str(uuid.uuid1())), new_image)
            cv2.imwrite(directory+'N/'+'N.'+'{}.jpg'.format(str(uuid.uuid1())), non_flipped)
        if flag & 0xFF == ord('o'):
            cv2.imwrite(directory+'O/'+'O.'+'{}.jpg'.format(str(uuid.uuid1())), new_image)
            cv2.imwrite(directory+'O/'+'O.'+'{}.jpg'.format(str(uuid.uuid1())), non_flipped)
        if flag & 0xFF == ord('p'):
            cv2.imwrite(directory+'P/'+'P.'+'{}.jpg'.format(str(uuid.uuid1())), new_image)
            cv2.imwrite(directory+'P/'+'P.'+'{}.jpg'.format(str(uuid.uuid1())), non_flipped)
        if flag & 0xFF == ord('q'):
            cv2.imwrite(directory+'Q/'+'Q.'+'{}.jpg'.format(str(uuid.uuid1())), new_image)
            cv2.imwrite(directory+'Q/'+'Q.'+'{}.jpg'.format(str(uuid.uuid1())), non_flipped)
        if flag & 0xFF == ord('r'):
            cv2.imwrite(directory+'R/'+'R.'+'{}.jpg'.format(str(uuid.uuid1())), new_image)
            cv2.imwrite(directory+'R/'+'R.'+'{}.jpg'.format(str(uuid.uuid1())), non_flipped)
        if flag & 0xFF == ord('s'):
            cv2.imwrite(directory+'S/'+'S.'+'{}.jpg'.format(str(uuid.uuid1())), new_image)
            cv2.imwrite(directory+'S/'+'S.'+'{}.jpg'.format(str(uuid.uuid1())), non_flipped)
        if flag & 0xFF == ord('t'):
            cv2.imwrite(directory+'T/'+'T.'+'{}.jpg'.format(str(uuid.uuid1())), new_image)
            cv2.imwrite(directory+'T/'+'T.'+'{}.jpg'.format(str(uuid.uuid1())), non_flipped)
        if flag & 0xFF == ord('u'):
            cv2.imwrite(directory+'U/'+'U.'+'{}.jpg'.format(str(uuid.uuid1())), new_image)
            cv2.imwrite(directory+'U/'+'U.'+'{}.jpg'.format(str(uuid.uuid1())), non_flipped)
        if flag & 0xFF == ord('v'):
            cv2.imwrite(directory+'V/'+'V.'+'{}.jpg'.format(str(uuid.uuid1())), new_image)
            cv2.imwrite(directory+'V/'+'V.'+'{}.jpg'.format(str(uuid.uuid1())), non_flipped)
        if flag & 0xFF == ord('w'):
            cv2.imwrite(directory+'W/'+'W.'+'{}.jpg'.format(str(uuid.uuid1())), new_image)
            cv2.imwrite(directory+'W/'+'W.'+'{}.jpg'.format(str(uuid.uuid1())), non_flipped)
        if flag & 0xFF == ord('x'):
            cv2.imwrite(directory+'X/'+'X.'+'{}.jpg'.format(str(uuid.uuid1())), new_image)
            cv2.imwrite(directory+'X/'+'X.'+'{}.jpg'.format(str(uuid.uuid1())), non_flipped)
        if flag & 0xFF == ord('y'):
            cv2.imwrite(directory+'Y/'+'Y.'+'{}.jpg'.format(str(uuid.uuid1())), new_image)
            cv2.imwrite(directory+'Y/'+'Y.'+'{}.jpg'.format(str(uuid.uuid1())), non_flipped)
        if flag & 0xFF == ord('z'):
            cv2.imwrite(directory+'Z/'+'Z.'+'{}.jpg'.format(str(uuid.uuid1())), new_image)
            cv2.imwrite(directory+'Z/'+'Z.'+'{}.jpg'.format(str(uuid.uuid1())), non_flipped)      
        
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()