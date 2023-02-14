# Importing necesssary libraries

import cv2
from image_preprocessing import process_image, draw_hand_connections, img_processing
import numpy as np
from PIL import Image, ImageTk
import mediapipe as mp
import tkinter as tk
from keras.models import model_from_json
import operator
import time
import sys
import os
import matplotlib.pyplot as plt
from spellchecker import SpellChecker
from string import ascii_uppercase


mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

class App:
    
    def __init__(self):

        self.json_file = open('sign_model.json','r')
        self.model_json = self.json_file.read()
        self.json_file.close()
        
        # Loaded model from JSON file
        self.loaded_model = model_from_json(self.model_json)

        # Loaded the Saved Weights on to the saved model
        self.loaded_model.load_weights('sign_model.h5')

        print('Model Loaded Successfully')

        self.root = tk.Tk()

        # Text on the Tab
        self.root.title('Hand Sign Language Detection GUI App')
        self.root.iconbitmap('favicons/favicon.ico')
        self.root.configure(background='#3C6478')
        width= self.root.winfo_screenwidth()
        height= self.root.winfo_screenheight()
        #setting tkinter window size
        self.root.geometry("%dx%d" % (width, height))
        self.root.protocol('WM_DELETE_WINDOW', self.close_window)

        # Text on the Window (Main Heading)
        self.main_heading = tk.Label(text="Sign Language To Text", font=('Ubuntu', 20, 'bold'))
        self.main_heading.place(x=60, y=0)

        # Side Heading
        self.side_heading = tk.Label(text="Reference Signs", font=('Ubuntu', 20, 'bold'))
        self.heading.place(x=1200, y=0)

        # Reference Image
        self.temp_ref_img = Image.open('pics/isl_alpha2.jpg')
        self.temp_ref_img = self.temp_ref_img.resize((350,750))
        self.ref_img = ImageTk.PhotoImage(self.temp_ref_img)
        self.ref_img_view = tk.Label(image=self.ref_img)
        self.ref_img_view.place(x=1150, y=40)
        
        '''--------------------------------------------------
        # without filter (Original)
        self.unfiltered_camera_view = tk.Label(self.root).place(x=40, y=100, width=800, height=600)

        # with filter (Canny edge)
        self.canny_camera_view = tk.Label(self.root).place(x=840, y=100, width=400, height=400)

        -----------------------------------------------------'''

        
        

        
        

        # Character text, font and its position
        self.character_label = tk.Label(self.root)
        self.character_label.place(x=40, y=750)              
        self.character_label.config(text="Predicted Character :", font=("Ubuntu", 40, "bold"))
        # Predicted character label and place
        self.pred_char = tk.Label(self.root)  
        self.pred_char.place(x=500, y=750)    
        
        # Word text, font and its position
        self.word_label = tk.Label(self.root)
        self.word_label.place(x=40, y=800)       
        self.word_label.config(text="Word :", font=("Ubuntu", 40, "bold"))
        # Predicted character label and place
        self.pred_word = tk.Label(self.root)  
        self.pred_word.place(x=350, y=800)    

        # Sentence text, font and its position
        self.sentence_label = tk.Label(self.root)
        self.sentence_label.place(x=40, y=850)
        self.sentence_label.config(text="Sentence :", font=("Ubuntu", 40, "bold"))
        # Predicted Sentence label and place
        self.pred_sentence = tk.Label(self.root)  # Sentence
        self.pred_sentence.place(x=480, y=850)


        # self.btn1 = tk.Button(self.root, command=self.action1).place(x=40, y=900)
        # self.btn2 = tk.Button(self.root, command=self.action2).place(x=240, y=900)
        # self.btn3 = tk.Button(self.root, command=self.action3).place(x=440, y=900)
        # self.btn4 = tk.Button(self.root, command=self.action4).place(x=640, y=900)
        # self.btn5 = tk.Button(self.root, command=self.action5).place(x=840, y=900)


        self.str = ""
        self.word = ""
        self.character = ""

        self.videocapture_loop()


    def videocapture_loop(self):
        
        self.cap = cv2.VideoCapture(0)

        success, frame = self.cap.read()

        if success:
            frame = cv2.flip(frame,1)

            cv2.rectangle(frame, (330, 10), (630, 310), (0,0,255) ,2)
            region_of_interest = frame[12:307, 332:628]
            self.left_image = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=self.left_image)
            self.unfiltered_camera_view.imgtk = imgtk
            self.unfiltered_camera_view.config(image=imgtk)


    
    def close_window(self):
        print('Exiting from Application')
        self.root.destroy()
        self.cap.release()
        cv2.destroyAllWindows()




print('Starting the Application')
obj = App()
obj.root.mainloop()


