from keras import backend as K
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Average, multiply, concatenate, Lambda, Multiply
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import to_categorical
from scipy.misc import imsave, imread
from image_reading import load_image_from_folder
import tensorflow as tf
from build_generator_unet import build_generator
#from keras.utils.vis_utils import plot_model
import sys,os
import matplotlib.pyplot as plt

import numpy as np

class SPECTGan():
    def __init__(self):
        # Input shape
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = build_generator()

        # The generator takes noise as input and generates imgs
        #seg       = Input(shape=(192,192,1))
        img_photo  = Input(shape=(256, 256, 1))
        img_scatter = Input(shape=(256, 256, 1))
        img_syn = self.generator([img_photo, img_scatter])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False
        #self.discriminator.compile(loss='binary_crossentropy',
         #                          optimizer=optimizer,
          #                         metrics=['accuracy'])
        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img_syn)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model([img_photo, img_scatter], valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

        self.output_path = '/netscratch/jchen/SPECT_CT_Syn/outputs/'
        self.output_image_path = self.output_path + 'images'
        #self.input_path = '/netscratch/jchen/SPECTimg_sep_slice/'
        self.input_path = '/netscratch/jchen/spect_recon_data/imgs_sc/'
        #plot_model(self.combined, to_file=self.output_path+'SPECT_syn.png', show_shapes=True, show_layer_names=True)
        #sys.exit(0)


    def build_discriminator(self):

        input_spect = Input((256, 256, 1))

        conv0 = Conv2D(32, kernel_size=3, strides=2, padding="same")(input_spect)
        conv0 = LeakyReLU(alpha=0.2)(conv0)
        conv0 = Dropout(0.25)(conv0)

        conv1 = Conv2D(64, kernel_size=3, strides=2, padding="same")(conv0)
        conv1 = ZeroPadding2D(padding=((0,1),(0,1)))(conv1)
        conv1 = BatchNormalization(momentum=0.8)(conv1)
        conv1 = LeakyReLU(alpha=0.2)(conv1)
        conv1 = Dropout(0.25)(conv1)

        conv2 = Conv2D(128, kernel_size=3, strides=2, padding="same")(conv1)
        conv2 = BatchNormalization(momentum=0.8)(conv2)
        conv2 = LeakyReLU(alpha=0.2)(conv2)
        conv2 = Dropout(0.25)(conv2)

        conv3 = Conv2D(256, kernel_size=3, strides=2, padding="same")(conv2)
        conv3 = BatchNormalization(momentum=0.8)(conv3)
        conv3 = LeakyReLU(alpha=0.2)(conv3)
        conv3 = Dropout(0.25)(conv3)
        conv3 = Flatten()(conv3)

        fc4 = Dense(1, activation='sigmoid')(conv3)

        model = Model(inputs=[input_spect], outputs=fc4)
        model.summary()

        return model

    def train(self, epochs, batch_size=35, save_interval=50):

        """Load data set"""
        train_portion = 0.9
        valid_portion = 0.1

        image_array, label_array = load_image_from_folder(self.input_path, (256, 256), HE=False, Truc=False, Aug=False)
        print("image_array, label_array generation done")

        image_train = image_array[0:int(train_portion * len(image_array)), :, :]
        label_train = label_array[0:int(train_portion * len(image_array)), :, :]
        image_valid = image_array[int(train_portion * len(image_array)):len(image_array), :, :]
        label_valid = label_array[int(train_portion * len(image_array)):len(image_array), :, :]

        # Correct data format
        image_train = np.expand_dims(image_train, axis=3)
        label_train = np.expand_dims(label_train, axis=3)
        image_valid = np.expand_dims(image_valid, axis=3)
        label_valid = np.expand_dims(label_valid, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        with open(self.output_path + 'stdout.txt', 'w') as f:
            pass

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a batch of images
            idx  = np.random.randint(0, image_train.shape[0], batch_size)
            imgs = image_train[idx]

            # photopeak window images
            imgPhoto = imgs[:, :, 0:256, :]
            imgPhoto = imgPhoto.reshape(len(imgPhoto), 256, 256, 1)

            # scatter window images
            imgScatter = imgs[:, :, 256:256 * 2, :]
            imgScatter = imgScatter.reshape(len(imgScatter),256,256, 1)

            # true CT images
            labels = label_train[idx]
            labels = labels.reshape(len(imgPhoto),256,256, 1)

            # Sample noise and generate a batch of new images
            gen_syn = self.generator.predict([imgPhoto, imgScatter])

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch([labels], valid)
            d_loss_fake = self.discriminator.train_on_batch([gen_syn] , fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch([imgPhoto, imgScatter], valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            with open(self.output_path + 'stdout.txt', 'a') as f:
                print('Epoch: ' + str(epoch) + '\ntrain d-loss: ' + str(d_loss) + '\ntrain g-loss: ' + str(g_loss), file=f)

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.validation_step(epoch, image_valid, label_valid, batch_size)

    def validation_step(self, epoch, image_valid, label_valid, batch_size):
        save_imgs = 2
        MSE_epoch = []
        for n in range(10):
            idx   = np.random.randint(0, image_valid.shape[0], batch_size)
            imgs  = image_valid[idx]
            imgPhoto = imgs[:, :, 0:256, :]
            imgPhoto = imgPhoto.reshape(len(imgPhoto), 256, 256, 1)
            imgScatter = imgs[:, :, 256:256 * 2, :]
            imgScatter = imgScatter.reshape(len(imgScatter), 256, 256, 1)
            labels   = label_valid[idx]
            labels = labels.reshape(len(imgScatter), 256, 256, 1)
            gen_syn = self.generator.predict([imgPhoto, imgScatter])

            if n < 1:
                for i in range(save_imgs):
                    plt.figure(num=None, figsize=(15, 6), dpi=200, facecolor='w', edgecolor='k')

                    plt.subplot(1, 4, 1); plt.axis('off'); plt.imshow(imgPhoto[i, :, :, 0], cmap='gray'); plt.title('photopeak image')

                    plt.subplot(1, 4, 2); plt.axis('off'); plt.imshow(imgScatter[i, :, :, 0], cmap='gray'); plt.title('scatter window image')

                    plt.subplot(1, 4, 3); plt.axis('off'); plt.imshow(labels[i, :, :, 0], cmap='gray'); plt.title('True CT')

                    plt.subplot(1, 4, 4); plt.axis('off'); plt.imshow(gen_syn[i, :, :, 0], cmap='gray'); plt.title('Syn CT')

                    output_name = 'syn.' + str(epoch) + '.'+str(i)+'.png'
                    plt.savefig(self.output_image_path+output_name)
                    plt.close()


            MSE_epoch.append(self.mean_square_error(gen_syn, labels))

        with open(self.output_path + 'stdout.txt', 'a') as f:
            print('val MSE score: ' + str(np.mean(MSE_epoch, 0)), file=f)

    def mean_square_error(self, y_pred, y_true):
        diff = y_pred - y_true
        mse = np.mean(np.sum(np.power(diff, 2), axis=1))
        return mse


if __name__ == '__main__':
    sys.path.append('/netscratch/jchen/SPECTVae_syn/')
    sys.path.append('/data/jchen/anaconda3/lib/python3.7/site-packages/')
    sys.path.append('/data/jchen/anaconda3/lib/python3.7/site-packages/graphviz')

    #os.environ["PATH"] += os.pathsep + '/data/jchen/anaconda3/lib/python3.7/site-packages'
    """Setup GPU """
    if K.backend() == 'tensorflow':
        # Use only gpu #X (with tf.device(/gpu:X) does not work)
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        # Automatically choose an existing and supported device if the specified one does not exist
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        # To constrain the use of gpu memory, otherwise all memory is used
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        K.set_session(sess)
        print('GPU Setup done')
    dcgan = SPECTGan()
    dcgan.train(epochs=4000, batch_size=50, save_interval=50)