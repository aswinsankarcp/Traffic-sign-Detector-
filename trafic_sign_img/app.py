from tkinter import *
import cv2
import numpy as np
from keras.models import load_model
# from tensorflow.keras.models import load_model
from PIL import Image, ImageTk
import os
from gtts import gTTS
from playsound import playsound

model = load_model('Traffic.h5')

classes = {1: 'Speed limit (20km/h)',
           2: 'Speed limit (30km/h)',
           3: 'Speed limit (50km/h)',
           4: 'Speed limit (60km/h)',
           5: 'Speed limit (70km/h)',
           6: 'Speed limit (80km/h)',
           7: 'End of speed limit (80km/h)',
           8: 'Speed limit (100km/h)',
           9: 'Speed limit (120km/h)',
           10: 'No passing',
           11: 'No passing veh over 3.5 tons',
           12: 'Right-of-way at intersection',
           13: 'Priority road',
           14: 'None',
           15: 'Stop',
           16: 'No vehicles',
           17: 'Veh > 3.5 tons prohibited',
           18: 'No entry',
           19: 'General caution',
           20: 'Dangerous curve left',
           21: 'Dangerous curve right',
           22: 'Double curve',
           23: 'Bumpy road',
           24: 'Slippery road',
           25: 'Road narrows on the right',
           26: 'Road work',
           27: 'Traffic signals',
           28: 'Pedestrians',
           29: 'Children crossing',
           30: 'Bicycles crossing',
           31: 'Beware of ice/snow',
           32: 'Wild animals crossing',
           33: 'End speed + passing limits',
           34: 'Turn right ahead',
           35: 'Turn left ahead',
           36: 'Ahead only',
           37: 'Go straight or right',
           38: 'Go straight or left',
           39: 'Keep right',
           40: 'Keep left',
           41: 'Roundabout mandatory',
           42: 'End of no passing',
           43: 'End no passing veh > 3.5 tons'}

cap = cv2.VideoCapture(0)
capWidth = int(cap.get(3))
capHeight = int(cap.get(4))

ter = Tk()
ter.geometry(str(capWidth) + "x" + str(capHeight + 60))
ter.resizable(False, False)
ter.config(bg="black")
ter.bind('<Escape>', lambda e: ter.quit())

l1 = Label(ter, compound=CENTER, anchor=CENTER, relief=RAISED, bg="black")
l1.pack()


def classify():
    success, frame = cap.read()
    image = Image.fromarray(frame, 'RGB')
    resize_image = image.resize((30, 30))
    expand_input = np.expand_dims(resize_image, axis=0)
    input_data = np.array(expand_input)
    input_data = input_data / 255
    pred = model.predict(input_data)
    result = pred.argmax()
    print(result)
    sign = classes[result + 1]
    p_val = np.amax(pred)
    print(p_val)
    if round(p_val * 100, 2) > 95:
        cv2.putText(frame, sign, (120, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        print(sign)
        lang = 'en'
        speak = gTTS(text=sign, lang=lang, slow=False)
        if os.path.exists("speak.mp3"):
            os.remove("speak.mp3")
        speak.save("speak.mp3")

        playsound("speak.mp3")

    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

    img1 = Image.fromarray(cv2image)

    imgtk = ImageTk.PhotoImage(image=img1)
    l1.imgtk = imgtk
    l1.configure(image=imgtk)
    l1.after(10, cam)


def cam():
    classify()

    # cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    #
    # img1 = Image.fromarray(cv2image)
    #
    # imgtk = ImageTk.PhotoImage(image=img1)
    # l1.imgtk = imgtk
    # l1.configure(image=imgtk)
    # l1.after(10, cam)


Button(text="CHECK", width=10, height=3, bg="yellow", command=classify, relief="ridge").place(relx=0.45, rely=0.9)

ter.mainloop()
