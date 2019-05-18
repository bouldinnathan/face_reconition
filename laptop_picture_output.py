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

import pickle
import os
import cv2
import numpy as np
global make_image_count
make_image_count=0

add_train=False
count=15

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

def make_image(X):
    global make_image_count
    make_image_count=make_image_count+1
    data=X
    prediction=model.predict(data, batch_size=None, verbose=1)
    print(prediction[1])
    prediction*= 255.
    prediction=prediction.astype(np.uint8)

    for i in range(len(prediction[1])):
        prediction[i]=cv2.cvtColor(prediction[i],cv2.COLOR_BGR2RGB)
    #print(Prediction[11])

    #plt.imshow(Prediction[11])                           
    #plt.axis('off')
    #plt.show()
    print(prediction[1])
    print(prediction.shape)

    #show the two pictures
    vis = np.concatenate((X, prediction), axis=1)
    cv2.imwrite(str(make_image_count)+'.jpg', vis)
    


#from keras import backend as K
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.9
#config.gpu_options.allow_growth=True

#K.clear_session()
#sess = tf.Session(config=config)
#K.set_session(sess)
###################################
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.5
#config.gpu_options.allow_growth=True
#sess = tf.Session(config=config)

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'#no gpu
###################################


#class MemoryCallback(Callback):
#    def on_epoch_end(self, epoch, log={}):
#        print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


src = "./data" #pokeRGB_black
dst = "./crop_faces" # resized
dst2= "./crop_faces_dudes"
#dst3= "./crop_not_face"

X=[]
Y=[]
not_face=[]
IMG_SIZE=50
IMG_SIZE_UP=IMG_SIZE*2

for each in os.listdir(dst):
    
    img = cv2.imread(os.path.join(dst,each))
    img=cv2.resize(img,(100,100),cv2.INTER_CUBIC)
    
    X.append(img)

for each in os.listdir(dst2):
    
    img = cv2.imread(os.path.join(dst2,each))
    img=cv2.resize(img,(100,100),cv2.INTER_CUBIC)
    
    X.append(img)

##for each in os.listdir(dst3):
##    
##    img = cv2.imread(os.path.join(dst,each))
##    img=cv2.resize(img,(100,100),cv2.INTER_CUBIC)
##    
##    not_face.append(img)

print(X[0].shape)
X = np.array(X).reshape(-1, IMG_SIZE_UP, IMG_SIZE_UP, 3)#type changed to numpy for shape
#Y = np.array(Y).reshape(-1, IMG_SIZE_UP, IMG_SIZE_UP, 3)#type changed to numpy for shape
#not_face = np.array(Y).reshape(-1, IMG_SIZE_UP, IMG_SIZE_UP, 3)#type changed to numpy for shape
X = X/255.0
#Y = Y/255.0
#not_face = not_face/255.0

#pickle_in = open("X.pickle","rb")
#X = pickle.load(pickle_in)

#pickle_in = open("y.pickle","rb")
#y = pickle.load(pickle_in)

#X = X/255.0

if add_train:
    current_output_local='output/a'+str(count)+'/'
    model = load_model(current_output_local+'/my_model.h5')
    print(model.summary())
else:
    model = Sequential()

    #model.add(Conv2D(100, (3, 3), input_shape=X.shape[1:]))
    #model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Conv2D(100, (3, 3)))
    #model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))

    #model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

    #model.add(Dense(64))

    #model.add(Dense(1))
    #model.add(Activation('sigmoid'))

    print(X.shape)    

    model.add(Conv2D(128, (2, 2), input_shape=X.shape[1:],padding="SAME"))#input_shape=X.shape[1:]))
    model.add(Conv2D(32, (2, 2),padding="SAME"))
    model.add(Conv2D(128, (2, 2),padding="SAME"))
    model.add(Conv2D(3, (2, 2),padding="SAME"))

##    model.add(Conv2D(32, (2, 2), input_shape=X.shape[1:],padding="SAME"))#input_shape=X.shape[1:]))
##    #model.add(BatchNormalization())
##    print('pass')
##    model.add(Conv2D(64, (2, 2),padding="SAME"))
##    #model.add(BatchNormalization())
##    print('pass')
##    model.add(Conv2D(128, (2, 2),padding="SAME"))
##    #model.add(BatchNormalization())
##    print('pass')
##    model.add(Conv2D(256, (2, 2),padding="SAME"))
##    #model.add(BatchNormalization())
##    print('pass')
##    model.add(Conv2D(512, (2, 2),padding="SAME"))
##    #model.add(BatchNormalization())
##    model.add(Conv2D(3, (2, 2),padding="SAME"))
##    #model.add(Dense((3)))

    model.add(Activation('tanh'))
    print('finished model')

    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())




temp=X[0:100,:,:,:]
while 1:
    count=count+1

    current_output_local='output/a'+str(count)+'/'
    make_directory_if_not_exists(current_output_local)
    
    filepath=current_output_local+"weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    checkpoint_two=ModelCheckpoint(current_output_local+'please.ckpt', save_weights_only=True, verbose=1)
    callbacks_list = [checkpoint,checkpoint_two]

    cv2.imwrite(current_output_local+'id10tCheck'+".jpg", X[2])
    
    model.fit(X, X, batch_size=50, epochs=1, validation_split=0.3,callbacks=callbacks_list,verbose=1,shuffle=True)

    model.save(current_output_local+'/my_checkpoint_test', overwrite=True)  # creates a HDF5 file 'my_model.h5'
    #del model  # deletes the existing model
    #model = load_model('output/my_model.h5')

    ######## save method two https://jovianlin.io/saving-loading-keras-models/
    # Save the weights
    model.save_weights(current_output_local+'model_weights.h5', overwrite=True)

    # Save the model architecture
    with open(current_output_local+'model_architecture.json', 'w') as f:
        f.write(model.to_json())
    #######
    ############save method three#################
    saver = tf.train.Saver()
    sess = backend.get_session()
    saver.save(sess, current_output_local)

    model.save(current_output_local+'my_model.h5')
    #print(model.get_weights())


    ############    
###does not like to reopen old
##    del model
##    del saver
##
##    model = load_model('keras_model.hdf5')
##
##    saver = tf.train.Saver()
##    sess = backend.get_session()
##    saver.restore(sess, current_output_local)





    ###########################perdiction####################################
    print(X.shape)
    temp=temp
    data=X[0:100,:,:,:]
    data=np.array(data).reshape(-1, IMG_SIZE_UP, IMG_SIZE_UP, 3)
    #print(data)
    prediction=model.predict(data, batch_size=None, verbose=1)
    print(prediction[1])
    
    prediction*= 255.
    prediction=prediction.astype(np.uint8)

    data*= 255
    data=data.astype(np.uint8)

    temp*= 255
    temp=temp.astype(np.uint8)
    #for i in range(len(prediction[1])):
    #    prediction[i]=cv2.cvtColor(prediction[i],cv2.COLOR_BGR2RGB)
    #print(Prediction[11])

    #plt.imshow(Prediction[11])                           
    #plt.axis('off')
    #plt.show()
    print(prediction[1])
    print(prediction.shape)

    #show the two pictures
    vis = np.concatenate((temp,data, prediction), axis=1)
    del temp,data,prediction
    #cv2.imshow("Data vs Prediction",Y[3]);
    #cv2.waitKey(0);
    filename=0
    print(vis.shape)
    for pairs in vis:
        filename=filename+1
        cv2.imwrite(current_output_local+str(filename)+".jpg", pairs)
        #plt.imshow(pairs)                           
        #plt.axis('off')
        #plt.show()



