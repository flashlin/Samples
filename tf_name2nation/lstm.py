import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input
from tensorflow.keras import Model, Sequential

def build_lstm():
   input = Input((40,1,1))
   x = Conv2D(32, 3, activation='relu', padding='same')(input)
   x = Flatten()(x)
   x = Dense(128, activation='relu')(x)
   output = Dense(10, activation='softmax')(x)
   model = Model(input,output)
   model.summary()


if __name__ == '__main__':
   build_lstm()
   
