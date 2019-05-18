import tensorflow as tf
from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose,BatchNormalization
from tensorflow.python.keras.models import load_model
from keras.callbacks import ModelCheckpoint
#from tensorflow.python.keras.backend.tensorflow_backend import set_session
import matplotlib.pyplot as plt
from tensorflow.python.keras import backend




from tensorflow.python.keras.models import load_model
from keras.models import Model
from keras.layers import Input
#from tensorflow.python.keras.models import predict
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
import tensorflow as tf
from tensorflow.python.keras import backend
from sklearn.cluster import KMeans
from scipy.spatial import distance

from itertools import chain

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

sensitivity=1.1
##########################################,name='midPoint'

def make_directory_if_not_exists(path):
    while not os.path.isdir(path):
        try:
            os.makedirs(path)
            break    
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise
        except WindowsError:
            print ("got WindowsError")
            pass  

def FBM(model,cut,test):
    inp = model.input                                           # input placeholder
    outputs = [layer.output for layer in model.layers]          # all layer outputs
    functors = [backend.function([inp, backend.learning_phase()], [out]) for out in outputs]    # evaluation functions

    
    layer_outs = [func([test, 1.]) for func in functors]
    prediction=layer_outs[cut][0]
    return prediction

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'#no gpu

src = "./data" #pokeRGB_black
dst = "./crop_Nathan" # resized
dst2= "./crop_face1_male"


X=[]
Y=[]
IMG_SIZE=50
IMG_SIZE_UP=IMG_SIZE*2

for each in os.listdir(dst):
    
    img = cv2.imread(os.path.join(dst,each))
    img=cv2.resize(img,(100,100),cv2.INTER_CUBIC)
    X.append(img)
    person1=X
print(len(X))
##
##for each in os.listdir(dst2):
##    
##    img = cv2.imread(os.path.join(dst2,each))
##    img=cv2.resize(img,(100,100),cv2.INTER_CUBIC)
##    Y.append(img)
##print(len(Y))
##person2=Y
##person2AndPerson1=list(person2)
##print('length of person2 person2AndPerson1 '+str(len(person2AndPerson1)))
###person2AndPerson1.append(person2)
##person2AndPerson1.extend(person1)
##print('length of person2+person1 person2AndPerson1 '+str(len(person2AndPerson1)))
##




X = np.array(X).reshape(-1, IMG_SIZE_UP, IMG_SIZE_UP, 3)#type changed to numpy for shape
#Y = np.array(Y).reshape(-1, IMG_SIZE_UP, IMG_SIZE_UP, 3)#type changed to numpy for shape
X = X/255.0
#Y = Y/255.0

print('here')
#plt.imshow(X[11])                           
#plt.axis('off')
#plt.show()

current_output_local='output/million_a16/'

model = load_model(current_output_local+'weights-improvement-02-0.69.hdf5')#'my_model.h5'
model.summary()
print('here2')
####
####
#####model = model_from_json(current_output_local+'model_architecture.json')
####model.load_weights(current_output_local+'model_weights.h5')
##
####def loadModel(jsonStr, weightStr):
####    json_file = open(jsonStr, 'r')
####    loaded_nnet = json_file.read()
####    json_file.close()
####
####    serve_model = model_from_json(loaded_nnet)
####    serve_model.load_weights(weightStr)
####
####    serve_model.compile(optimizer=tf.train.AdamOptimizer(),
####                        loss='mse',
####                        metrics=['accuracy'])
####    return serve_model
#

#print(model.get_weights())
#model = loadModel(current_output_local+'model_architecture.json', current_output_local+'model_weights.h5')#, current_output_local+'model_weights.h5'

#model = model.load_weights(current_output_local+'weights-improvement-01-0.83.hdf5')



#model = load_model(current_output_local+'keras_model.hdf5')



##model = load_model(current_output_local+'please.ckpt.index')
##
##saver = tf.train.Saver()
##sess = backend.get_session()
##saver.restore(sess, current_output_local)

#model.load_weights(current_output_local+'please.ckpt')

#model.set_weights(current_output_local+'please.ckpt')#'please.ckpt'


###########################perdiction####################################
#print(Y.shape)
#X=X[0:100,:,:,:]
#data=X[0:100,:,:,:]
#print(data.shape)
#data=np.array(data).reshape(-1, IMG_SIZE_UP, IMG_SIZE_UP, 3)
#print(data.shape)
#print(data)



##############################all of the magic#######################################################################
inp = model.input                                           # input placeholder
outputs = [layer.output for layer in model.layers]          # all layer outputs
functors = [backend.function([inp, backend.learning_phase()], [out]) for out in outputs]    # evaluation functions

test=X
layer_outs = [func([test, 1.]) for func in functors]
######################################################################################################################


#print(layer_outs[0])
prediction=layer_outs[3][0]

data=np.array(prediction)
#data*= 255
#data=data.astype(np.uint8)
print('Shape of data: '+str(data.shape))



kmeans = KMeans(n_clusters=1, random_state=0).fit(data)
print(kmeans.labels_)
print(kmeans.cluster_centers_.shape)
print(kmeans.cluster_centers_)
######kmeans.predict([[0, 0], [12, 3]])
dist=0
for compress in data:
    dist_temp=distance.euclidean(compress,kmeans.cluster_centers_)
    if dist_temp>dist:
        dist=dist_temp








FBM(model,3,test)



#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)

ret, img = cap.read()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
x,y=gray.shape
total_pixels=x+y
sensitivity=.2#range 0-1.

while 1:
    last_gray=gray
    ret, img = cap.read(0)
    cv2.imshow('img',img)
    cv2.waitKey(100)
    if ret==1:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
       
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            
            test=cv2.resize(img[y:y+h, x:x+w],(100,100),cv2.INTER_CUBIC)
            #compress=FBM(model,3,test)
            test = np.array(test).reshape(-1, IMG_SIZE_UP, IMG_SIZE_UP, 3)
            layer_outs = [func([test, 1.]) for func in functors]
            prediction=layer_outs[3][0]

            data=np.array(prediction)
            data/= 255
            #data=data.astype(np.uint8)

            
            dist_temp=distance.euclidean(data,kmeans.cluster_centers_)
            if dist_temp<dist:
                print(str(dist_temp)+' < '+str(dist))
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            else:
                print(str(dist_temp)+' < '+str(dist))
                cv2.rectangle(img,(x,y),(x+w,y+h),(128,128,128),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            #eyes = eye_cascade.detectMultiScale(roi_gray)
            #for (ex,ey,ew,eh) in eyes:
            #    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            cv2.imshow('img',img)
    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

























