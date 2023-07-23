#headerSection
import cv2
import os
import numpy as np
import dlib
from imutils import face_utils
import random
#-----------------------------------------------------------------------------
lst=["Be Alert!.A Man Sleep in his Driving.","Back to Drive. Be Alert!","Can't Sleep-Think about your future.","Be Alert!,Don't Sleep at All"]
label2=random.choices(lst)
#webcamSection
cap=cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
#Initializing the face detector and landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#status marking for current state
sleep = 0
drowsy = 0
active = 0
color=(0,0,0)
#functions
def compute(ptA,ptB):
        dist = np.linalg.norm(ptA - ptB)
        return dist

def blinked(a,b,c,d,e,f):
   up = compute(b,d) + compute(c,e)
   down = compute(a,f)
   ratio = up/(2.0*down)

    #Checking if it is blinked
   if(ratio>0.25):
             return 2
   elif(ratio>0.21 and ratio<=0.25):
            return 1
   else:
           return 0

#graphics
#imageoverlappedongraphicsbackground
imgback=cv2.imread("background.png")
#toreterivedataofmodefromfolder
modefolder="Resources/Resources/bg"
listmodefolder=os.listdir(modefolder)
imgmodelist=[]
for path in listmodefolder:
    imgmodelist.append(cv2.imread(os.path.join(modefolder,path)))
print(len(imgmodelist))




while True:
    success,img=cap.read()
    imgback[162:162 + 480,55:55 + 640]=img
    imgback[44:44 + 633,808:808 + 414]=imgmodelist[0]

    if not success:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    gray = cv2.cvtColor(imgback, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    #detected face in faces array
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        face_frame = imgback.copy()
        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        #The numbers are actually the landmarks which will show eye
        left_blink = blinked(landmarks[36],landmarks[37], 
           landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42],landmarks[43], 
          landmarks[44], landmarks[47], landmarks[46], landmarks[45])
        
        #Now judge what to do for the eye blinks
        if(left_blink==0 or right_blink==0):
               sleep+=1
               drowsy=0
               active=0
               if(sleep>6):
                   #414x640imagesize
                      imgback[162:162 + 480,55:55 + 640]=img
                      imgback[44:44 + 635,808:808 + 414]=imgmodelist[1]
                      color = (0,0,255)

        elif(left_blink==1 or right_blink==1):
                sleep=0
                active=0
                drowsy+=1
                if(drowsy>6):
                       
                       imgback[162:162 + 480,55:55 + 640]=img
                       imgback[44:44 + 635,808:808 + 417]=imgmodelist[3]
                       color = (0,0,255)

        else:
               drowsy=0
               sleep=0
               active+=1
               if(active>6):
                      
                      imgback[162:162 + 480,55:55 + 640]=img
                      imgback[44:44 + 633,808:808 + 414]=imgmodelist[2]
                      color = (0,255,0)
          
        #cv2.putText(imgback, status, (910,550), cv2.FONT_HERSHEY_SIMPLEX, 2.2, color,3)
        label2=str(label2)
        cv2.putText(imgback, label2[2:-2], (850,180), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0,105), 2, cv2.LINE_AA)
        
    for n in range(0, 68):
           (x,y) = landmarks[n]
           cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)




    cv2.imshow("DriverAlertSystem",imgback)
    cv2.waitKey(1)
