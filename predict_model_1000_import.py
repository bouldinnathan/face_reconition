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



#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'#no gpu

src = "./data" #pokeRGB_black
dst = "./crop_face1_female" # resized
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

for each in os.listdir(dst2):
    
    img = cv2.imread(os.path.join(dst2,each))
    img=cv2.resize(img,(100,100),cv2.INTER_CUBIC)
    Y.append(img)
print(len(Y))
person2=Y
person2AndPerson1=list(person2)
print('length of person2 person2AndPerson1 '+str(len(person2AndPerson1)))
#person2AndPerson1.append(person2)
person2AndPerson1.extend(person1)
print('length of person2+person1 person2AndPerson1 '+str(len(person2AndPerson1)))

#interleving to face to make faults matches
interlevedPerson2AndPerson1=list(chain.from_iterable(zip(person1[:len(person2)], person2[:len(person1)])))
print('Length of interleved '+str(len(interlevedPerson2AndPerson1)))

notMatched=[]
for i in range(len(interlevedPerson2AndPerson1)):
    notMatched.append([0,1])
notMatched = np.array(notMatched).reshape(-1,2)

print(type(person2AndPerson1))
person2AndPerson1.extend(interlevedPerson2AndPerson1)
X=person2AndPerson1
print('length of person2+person1+interleaved person2AndPerson1 '+str(len(person2AndPerson1)))



answers=np.array([[1,0],
         [1,0],
         [1,0],
         [1,0],
         [1,0],
         [1,0],
         [1,0],
         [1,0],
         [1,0],
         [1,0],
         [1,0],
         [1,0],
         [1,0],
         [1,0],
         [1,0],
         [1,0],
         [1,0],
         [1,0],
         [1,0],
         [1,0],
         [1,0],
         [1,0],
         [1,0],
         [1,0],
         [1,0],
         [1,0],
         [1,0],
         [1,0],
         [1,0],
         [1,0],
         [1,0],
         [1,0],
         [1,0],
         [1,0],
         [1,0],
         [1,0],
         [0,1],
         [1,0],
         [1,0],
         [1,0],
         [1,0],
         [1,0],
         [1,0],
         [1,0],
         [1,0],
         [1,0],
         [1,0],
         [1,0],
         [1,0],
         [1,0],
         [1,0],
         [1,0],
         [1,0],
         [1,0],
         [1,0],
         [1,0],
         [1,0],
         [1,0],
         [1,0],
         [1,0],
         [1,0],
         [1,0],
         [1,0],
         [1,0]])

answers=np.concatenate((answers, notMatched), axis=0)


X = np.array(X).reshape(-1, IMG_SIZE_UP, IMG_SIZE_UP, 3)#type changed to numpy for shape
#Y = np.array(Y).reshape(-1, IMG_SIZE_UP, IMG_SIZE_UP, 3)#type changed to numpy for shape
X = X/255.0
#Y = Y/255.0
answers = np.array(answers).reshape(-1,2)#type changed to numpy for shape

#plt.imshow(X[11])                           
#plt.axis('off')
#plt.show()

current_output_local='output/million_a16/'

model = load_model(current_output_local+'weights-improvement-02-0.69.hdf5')#'my_model.h5'
model.summary()
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
data*= 255
data=data.astype(np.uint8)
print('Shape of data: '+str(data.shape))


pca = PCA(120)  # project from 64 to 2 dimensions
projected = pca.fit_transform(data)
print(projected.shape)

plt.figure()
plt.imshow(data)
plt.show()

shape=np.zeros((121,3))
print(shape.shape)
colors = ("red", "green", "blue")

##plt.scatter(projected[:, 0], projected[:, 1],
##            c=shape.shape, edgecolor='none', alpha=0.5,
##            cmap=plt.cm.get_cmap('spectral', 10))#cmap=plt.cm.Spectral(np.linspace(0, 1, 10))   cmap=plt.cm.get_cmap('spectral', 10)
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar();



#prediction=model.predict(data, batch_size=None, verbose=1)


prediction=layer_outs

#print(prediction[1])

#prediction*= 255

#print(prediction[3][0])

data=prediction[2]
#data=prediction[5][0][0]
data = np.array(data)
data*= 255
data=data.astype(np.uint8)
data=data[0]


#prediction=int(prediction)
#prediction*= 255
#prediction=prediction.astype(np.uint8)
print(data.shape)
mid_point_data=list()
##for i in range(len(data)-1):
##    if i<0:
##        1+1
##        print(i)
##    else:
##        #mid_point_data=[mid_point_data,np.concatenate((data[i],data[i+1]),axis=1)]
##        mid_point_data.append(np.concatenate((data[i],data[i+1]),axis=-1))
##        #mid_point_data=np.concatenate(mid_point_data,np.concatenate(data[0][i],data[0][i+1],axis=1),axiz=0)
##        print(i)
mid_point_data=np.asarray(mid_point_data)
print('shape of data: '+str(mid_point_data.shape))

mid_point_data = np.array(mid_point_data).reshape(-1, IMG_SIZE_UP, IMG_SIZE_UP, 2)
model2 = Sequential()
#print(X.shape)    
model2.add(Conv2D(2, (2, 2), input_shape=mid_point_data.shape[1:],padding="SAME",name='inputPoint'))#input_shape=X.shape[1:]))
model2.add(Conv2D(128, (2, 2),padding="SAME"))
model2.add(Flatten())
model2.add(Dense(2, activation='relu'))
model2.compile(loss='mse',
                optimizer='adam',
                metrics=['accuracy'])
print(model2.summary())

model2.fit(mid_point_data, answers, batch_size=102, epochs=1, validation_split=0.3,verbose=1,shuffle=True)
##############print(data1)
##############
##############x,y,z=data1.shape
##############nzCount = np.count_nonzero(data1)
##############print(str(data1.shape)+" has number of non-zeros: "+str(nzCount)+' of '+str(x*y*z)+' and zeros: '+str(x*y*z-nzCount))
##############original=x*y*z-nzCount
##############
##############subtract_data=np.subtract(data2,data1*sensitivity)
##############x,y,z=subtract_data.shape
##############nzCount = np.count_nonzero(subtract_data>0)
##############print(str(subtract_data.shape)+" has number of non-zeros: "+str(nzCount)+' of '+str(x*y*z)+' and zeros: '+str(x*y*z-nzCount))
##############original_max=x*y*z-nzCount
##############
##############subtract_data=np.add(data2,data1*-(sensitivity-2))
##############x,y,z=subtract_data.shape
##############nzCount = np.count_nonzero(subtract_data>0)
##############print(str(subtract_data.shape)+" has number of non-zeros: "+str(nzCount)+' of '+str(x*y*z)+' and zeros: '+str(x*y*z-nzCount))
##############original_min=x*y*z-nzCount
##############
##############match=0
##############
##############if original<original_max and original>original_min:
##############    match=match+1
#############################################################






#for i in range(len(prediction[1])):
#    prediction[i]=cv2.cvtColor(prediction[i],cv2.COLOR_BGR2RGB)
#print(Prediction[11])

#plt.imshow(Prediction[11])                           
#plt.axis('off')
#plt.show()


#show the two pictures
##############vis = np.concatenate((Y,data, prediction), axis=1)
#cv2.imshow("Data vs Prediction",vis);
#cv2.waitKey(0);
############filename=0
############print(vis.shape)
############for pairs in vis:
############    filename=filename+1
############    cv2.imwrite(current_output_local+str(filename)+".jpg", pairs)
    #plt.imshow(pairs)                           
    #plt.axis('off')
    #plt.show()
cv2.imwrite(str(1)+".jpg", np.concatenate((prediction[5][0][0]*255,prediction[5][0][1]*255,prediction[5][0][2]*255,prediction[5][0][3]*255), axis=1))
