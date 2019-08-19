from keras import backend as K
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, MaxPool2D, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import regularizers
import os

def build_generator():

    # ---------------------
    #  U-Net
    # ---------------------
    input_size = (256, 256, 1)

    """ first encoder for photopeak image """
    input_ph = Input(input_size)
    conv1_ph = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input_ph)
    conv1_ph = BatchNormalization()(conv1_ph)
    conv1_ph = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1_ph)
    conv1_ph = BatchNormalization()(conv1_ph)
    pool1_ph = MaxPool2D(pool_size=(2, 2))(conv1_ph)

    conv2_ph = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1_ph)
    conv2_ph = BatchNormalization()(conv2_ph)
    conv2_ph = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2_ph)
    conv2_ph = BatchNormalization()(conv2_ph)
    pool2_ph = MaxPool2D(pool_size=(2, 2))(conv2_ph)

    conv3_ph = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2_ph)
    conv3_ph = BatchNormalization()(conv3_ph)
    conv3_ph = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3_ph)
    conv3_ph = BatchNormalization()(conv3_ph)
    pool3_ph = MaxPool2D(pool_size=(2, 2))(conv3_ph)

    conv4_ph = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3_ph)
    conv4_ph = BatchNormalization()(conv4_ph)
    conv4_ph = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4_ph)
    conv4_ph = BatchNormalization()(conv4_ph)
    drop4_ph = Dropout(0.5)(conv4_ph)
    pool4_ph = MaxPool2D(pool_size=(2, 2))(drop4_ph) 

    """ second encoder for scatter image """
    input_sc = Input(input_size)
    conv1_sc = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input_sc)
    conv1_sc = BatchNormalization()(conv1_sc)
    conv1_sc = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1_sc)
    conv1_sc = BatchNormalization()(conv1_sc)
    pool1_sc = MaxPool2D(pool_size=(2, 2))(conv1_sc)  # 192x192

    conv2_sc = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1_sc)
    conv2_sc = BatchNormalization()(conv2_sc)
    conv2_sc = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2_sc)
    conv2_sc = BatchNormalization()(conv2_sc)
    pool2_sc = MaxPool2D(pool_size=(2, 2))(conv2_sc)  # 96x96

    conv3_sc = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2_sc)
    conv3_sc = BatchNormalization()(conv3_sc)
    conv3_sc = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3_sc)
    conv3_sc = BatchNormalization()(conv3_sc)
    pool3_sc = MaxPool2D(pool_size=(2, 2))(conv3_sc)  # 48x48

    conv4_sc = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3_sc)
    conv4_sc = BatchNormalization()(conv4_sc)
    conv4_sc = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4_sc)
    conv4_sc = BatchNormalization()(conv4_sc)
    drop4_sc = Dropout(0.5)(conv4_sc)
    pool4_sc = MaxPool2D(pool_size=(2, 2))(drop4_sc)  # 24x24

    conv5_sc = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4_sc)
    conv5_sc = BatchNormalization()(conv5_sc)
    conv5_sc = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5_sc)
    conv5_sc = BatchNormalization()(conv5_sc)
    conv5_sc = Dropout(0.5)(conv5_sc) 

    conv5_ph = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4_ph)
    conv5_ph = BatchNormalization()(conv5_ph)
    conv5_ph = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5_ph)
    conv5_ph = BatchNormalization()(conv5_ph)
    conv5_ph = Dropout(0.5)(conv5_ph)

    merge5_cm = concatenate([conv5_ph, conv5_sc], axis=3)  # 12x12

    up7_cm = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(merge5_cm))  # 24x24
    up7_cm = BatchNormalization()(up7_cm)
    merge7_cm = concatenate([drop4_sc, drop4_ph, up7_cm], axis=3)  # cm: cross modality
    conv7_cm = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7_cm)
    conv7_cm = BatchNormalization()(conv7_cm)
    conv7_cm = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7_cm)
    conv7_cm = BatchNormalization()(conv7_cm)

    up8_cm = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7_cm))
    up8_cm = BatchNormalization()(up8_cm)
    merge8_cm = concatenate([conv3_sc, conv3_ph, up8_cm], axis=3)  # cm: cross modality
    conv8_cm = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8_cm)
    conv8_cm = BatchNormalization()(conv8_cm)
    conv8_cm = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8_cm)
    conv8_cm = BatchNormalization()(conv8_cm)

    up9_cm = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8_cm))
    up9_cm = BatchNormalization()(up9_cm)
    merge9_cm = concatenate([conv2_sc, conv2_ph, up9_cm], axis=3)  # cm: cross modality
    conv9_cm = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9_cm)
    conv9_cm = BatchNormalization()(conv9_cm)
    conv9_cm = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9_cm)
    conv9_cm = BatchNormalization()(conv9_cm)

    up10_cm = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv9_cm))
    up10_cm = BatchNormalization()(up10_cm)
    merge10_cm = concatenate([conv1_sc, conv1_ph, up10_cm], axis=3)  # cm: cross modality
    conv10_cm = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge10_cm)
    conv10_cm = BatchNormalization()(conv10_cm)
    conv10_cm = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv10_cm)
    conv10_cm = BatchNormalization()(conv10_cm)

    conv11_cm = Conv2D(filters=4, kernel_size=3, activation='relu', padding='same')(conv10_cm)
    conv11_cm = Conv2D(filters=2, kernel_size=3, activation='relu', padding='same')(conv11_cm)
    conv11_cm = Conv2D(filters=1, kernel_size=3, activation='relu', padding='same')(conv11_cm)

    model = Model(inputs=[input_ph, input_sc], outputs=conv11_cm)
    model.summary()


    '''
    input_size = (256, 256, 1)
    input_photo = Input(input_size)
    input_scatter = Input(input_size)
    input_img = concatenate([input_photo, input_scatter], axis=3)

    conv1 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input_img)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPool2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPool2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPool2D(pool_size=(2, 2))(drop3)

    conv4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    drop4 = Dropout(0.5)(conv4)

    up8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop4))
    up8 = BatchNormalization()(up8)
    merge8 = concatenate([drop3, up8], axis=3)
    conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    up9 = BatchNormalization()(up9)
    merge9 = concatenate([conv2, up9], axis=3)
    conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = BatchNormalization()(conv9)

    up10 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv9))
    up10 = BatchNormalization()(up10)
    merge10 = concatenate([conv1, up10], axis=3)
    conv10 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge10)
    conv10 = BatchNormalization()(conv10)
    conv10 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv10)
    conv10 = BatchNormalization()(conv10)

    conv11 = Conv2D(filters=4, kernel_size=3, activation='relu', padding='same')(conv10)
    conv11 = BatchNormalization()(conv11)
    conv12 = Conv2D(filters=2, kernel_size=3, activation='relu', padding='same')(conv11)
    conv12 = BatchNormalization()(conv12)
    conv13 = Conv2D(filters=1, kernel_size=1, activation='relu', padding='same')(conv12)

    model = Model(inputs=[input_photo, input_scatter], outputs=conv13)
    model.summary()
    '''
    return model
