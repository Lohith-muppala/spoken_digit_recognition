
import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
import cv2
from keras.models import Sequential
import tensorflow as tf
from keras import optimizers
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from tensorboardcolab import TensorBoardColab, TensorBoardColabCallback
import sounddevice as sd
from scipy.io.wavfile import write
import scipy.io.wavfile as wav
import scipy.ndimage

current_directory = os.path.dirname(os.path.abspath(__file__))
images_directory = os.path.join(current_directory,"spectrograms//")
model_save_dir = os.path.join(current_directory,'trained_model.h5')


def create_dict(path):
    imgs_dict = []
    labels = []
    for file in os.listdir(path):
        label = file.split('_')[0]
        img = cv2.imread(path+file)
        imgs_dict.append(img)
        labels.append(label)
    return imgs_dict,labels
   

def main():
    data,labels = create_dict(images_directory)
    print(len(data))
    data = np.array(data)
    print(data.shape)
    print('Do you want to train the model or use a pre-existing model?')
    ans = input('Please enter your choice (yes or no): ')
    if ((ans == 'Yes') or (ans == 'yes')):
        print('Training the model again...')
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
        data_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=data_shape))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))
        adam = optimizers.Adam(lr = 0.001)
        model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
        model.summary()
        history = model.fit(X_train, y_train, batch_size=32, epochs=100, verbose=1, validation_data=(X_test, y_test),callbacks=[ tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)])
        results = model.evaluate(X_test, y_test)
        print("The accuracy of the model is {:.2f}%".format(results[1]*100))
        
        tf.saved_model.save(model,model_save_dir)
        print('Model saved....')
    
    print('Opening the pre-trained saved model..')
    model = tf.keras.models.load_model(model_save_dir)
    model.summary()

    spectrogram_dimensions=(64, 64)
    noverlap=16
    cmap='gray_r'
    fs = 8000  # Sample rate
    seconds = 1  # Duration of recording
    test_path = os.path.join(current_directory,'test.png')
    input("Press enter to record the number: ")
    
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    write(os.path.join(current_directory,'test.wav'), fs, myrecording)  # Save as WAV file 
    print('Done recording')
    sample_rate, samples = wav.read(os.path.join(current_directory,'aud_test.wav'))
    
    fig = plt.figure()
    fig.set_size_inches((spectrogram_dimensions[0]/fig.get_dpi(), spectrogram_dimensions[1]/fig.get_dpi()))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.specgram(samples, cmap=cmap, Fs=2, noverlap=noverlap)
    fig.savefig(os.path.join(current_directory,'test.png'), bbox_inches="tight", pad_inches=0)
    img_test = cv2.imread(os.path.join(current_directory,'test.png'))
    img_test = np.expand_dims(img_test, axis=0)
    print('The predicted number is: ',np.argmax(model.predict(img_test)))

if __name__ == '__main__':
    main()