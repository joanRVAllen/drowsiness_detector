import cv2
import numpy as np
import time
import tensorflow as tf
from keras.preprocessing.image import array_to_img, img_to_array
from tensorflow.keras.models import load_model

class Drowsiness:
    def __init__(self):
        self.model = load_model('../model/model_trial')
        self.face_cascade = cv2.CascadeClassifier('../video_feed/haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier('../video_feed/haarcascade_eye.xml')
        
        # TODO: better model for open/closed detection
            # for asian eyes
            # for squinting
            # for higher angle

    def detect_eyes(self, img):
        gray_picture = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # grayscale
        det_face = self.face_cascade.detectMultiScale(gray_picture, 1.3, 5)
        for (x, y, w, h) in det_face: # draw square on face
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,0),2)
        if det_face == ():
            return img
        x,y,w,h = det_face[0]
        gray_face = gray_picture[y:y+h, x:x+w]
        face = img[y:y+h, x:x+w] # coordinates of face

        # crop the face
        det_eyes = self.eye_cascade.detectMultiScale(gray_face)
        
        # TODO: improve eye detection
            # sometimes, it detects ears, nose and background as face/eyes
            # set the face detection to just the largest square (driver)
            # eyes should be leveled and one eye on the right, one on the left
        half = int(np.size(face, 0) / 2)
        upper = [i for i in det_eyes if (i[1]<half)]
        if len(upper) <= 2:
            upper.append([0,0,1,1])
            upper.append([0,0,1,1])
            for (ex,ey,ew,eh) in upper:
                cv2.rectangle(face,(ex,ey),(ex+ew,ey+eh),(0,255,255),2)
        elif len(upper) > 2:
            up_eyes = [i[3] for i in upper]
            biggest = sorted(up_eyes)[-2:]
            ind = [up_eyes.index(i) for i in biggest]
            upper = [upper[i] for i in ind]
            for (ex,ey,ew,eh) in upper:
                cv2.rectangle(face,(ex, ey),(ex+ew, ey+eh),(0,255,255),2)
        left = upper[0]
        right = upper[1]
        self.left = face[left[1]:left[1]+left[3], left[0]:left[0]+left[2]]
        self.right = face[right[1]:right[1]+right[3], right[0]:right[0]+right[2]]


    def pred(self, array):
        input_ = tf.reshape(array, (-1, 6400))
        prediction = (self.model.predict(input_) > 0.5).astype('int32')[0][0]
        if prediction == 0:
            return 'close'
        elif prediction == 1:
            return 'open'
    
    def drowsy(self, image):
        def format(img):
            image = array_to_img(img)
            image = image.convert('L').resize((80,80))
            return img_to_array(image)
        l = self.pred(format(self.left))
        r = self.pred(format(self.right))
        font = cv2.FONT_HERSHEY_SIMPLEX
        if l == 'open' and r == 'open':
            self.tag = 'open'
            return cv2.putText(image, 
                'OPEN', 
                (500, 50), 
                font, 2, 
                (0, 0, 0), 
                6, 
                cv2.LINE_4)
        else:
            self.tag = 'close'
            return cv2.putText(image, 
                'CLOSE', 
                (500, 50), 
                font, 2, 
                (0, 0, 0), 
                6, 
                cv2.LINE_4)
    
    def video_feed(self):
        vid = cv2.VideoCapture(0)
        # clo = time.time()
        # TODO: account for drowsy/sleepy detection and lag
            # make another counter for sleepy/drowsy and establish a threshold
        counter = 0
        while (True):
            _, frame = vid.read()
            self.detect_eyes(frame)

            self.drowsy(frame)

            cv2.imshow('frame', frame)

            if self.tag == 'close':
                # print('close', time.time() - clo)
                counter += 1
                # clo = time.time()
            
            elif self.tag == 'open':
                # print('open', time.time() - clo)
                counter = 0
                # clo = time.time()
            if counter > 3 and counter < 6:
                print('You are drowsy')
                # TODO: make alarm instead of printing a statement
            
            elif counter > 5:
                print('you are sleepy')
                # TODO: make alarm instead of printing a statement

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        vid.release()
        cv2.destroyAllWindows()

