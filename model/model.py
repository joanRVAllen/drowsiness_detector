from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

class DrowsyModel:
    def __init__(self):
        self
    
    def convert_img(self, root, img_name):
        img = load_img(root + img_name, color_mode='grayscale', target_size=(80,80))
        img_arr = np.array(img_to_array(img), dtype=object)
        return img_arr

    def to_df(self, root, label):
        '''
        root: path to the images folder
        label: (int) 0 or 1
        '''
        _, _, files = next(os.walk(root))
        list_ = []
        for i in files:
            list_.append(self.convert_img(root, i))
        df = pd.DataFrame(data=list_, columns=['x'])
        df['y'] = [label] * len(df.x)
        return df
    
    def training(self):
        close_ = '../model/train/Closed_Eyes/'
        open_ = '../model/train/Open_Eyes/'
        closed = self.to_df(close_, 0)
        opened = self.to_df(open_, 1)

        eyes = [closed, opened]
        result = pd.concat(eyes, ignore_index=True)
        result = shuffle(result, random_state=32).reset_index(drop=True)
        print((result.shape))

        # split data
        X = result['x']
        y = result['y']

        # train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32, shuffle=True)

        # check the length of dataframes
        assert len(X_train) == int(len(X) * 0.8), 'X_train error'
        assert len(y_train) == int(len(X) * 0.8), 'y_train error'
        assert len(X_test) == len(X) - len(X_train), 'X_test error'
        assert len(y_test) == len(y) - len(y_train), 'y_test error'

        # normalize
        max_value = 255
        X_train = X_train / max_value
        X_test = X_test / max_value

        # convert to tf format
        X_train = X_train.apply(lambda x: tf.convert_to_tensor(x, dtype=tf.float32))
        X_test = X_test.apply(lambda x: tf.convert_to_tensor(x, dtype=tf.float32))

        # flatten
        X_train = X_train.apply(lambda x: tf.reshape(x, [-1]))
        X_test = X_test.apply(lambda x: tf.reshape(x, [-1]))

        # stack to tensorflow data
        X_vtr = [i for i in X_train]
        X_vte = [i for i in X_test]
        X_train = tf.stack(X_vtr)
        X_test = tf.stack(X_vte)

        return X_train, X_test, y_train, y_test

    def model(self):
        X_train, X_test, y_train, y_test = self.training()
        self.model = Sequential()
        self.model.add(
            Dense(50,
            activation='relu',
            input_dim=6400)
        )
        self.model.add(
            Dense(30,
            activation='relu')
        )
        self.model.add(
            Dense(1,
            activation='sigmoid')
        )
        # compile
        self.model.compile(optimizer='adam', 
        loss='binary_crossentropy',
        metrics=['accuracy'])
        # fit self.model
        self.results = self.model.fit(X_train, y_train, 
        epochs=5, validation_data=(X_test, y_test))
        # evaluate
        self.test_score = self.model.evaluate(X_test, y_test)

    def save_model(self, model_name='model_trial'):
        self.model.save(model_name) # save model