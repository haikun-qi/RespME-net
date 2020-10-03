import numpy as np
import keras
import scipy.io as sio
import random
import os


# data generator for training
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=16, dim=(64,64,64), n_channels = 2, shuffle=True):
        'Initialization'
        self.dim = dim
        self.n_channels = n_channels
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X= self.__data_generation(list_IDs_temp)

        return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size,)+ self.dim +(self.n_channels,),dtype='float32')

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = load_data(ID, self.dim)
        return X

def load_data(ID, dims):
    crop_size = np.array([192,192,80])

    filename = ID['name']
    xx = ID['xx']
    yy = ID['yy']
    zz = ID['zz']
    bin_num = ID['bin_num']
    # print(filename,bin_num,xx)
    ### load data & normalization
    mat_contents = sio.loadmat(filename)
    x_data = mat_contents['msense']
    x_data = np.array(x_data)

    im1 = x_data[:, :, :, 0]
    im2 = x_data[:, :, :, bin_num]

    im1 = (im1 - np.min(im1)) / (np.max(im1) - np.min(im1))
    im2 = (im2 - np.min(im2)) / (np.max(im2) - np.min(im2))
    x_data=[im1, im2]
    x_data = np.array(x_data)

    ### crop to the central part
    x_data = np.transpose(x_data, [1, 2, 3, 0])
    x_shape = np.shape(x_data)[0:-1]
    x_shape = np.array(x_shape)
    limit = x_shape - crop_size + 1
    stx = int((limit[0].astype('float32')) / 2.0)
    sty = int((limit[1].astype('float32')) / 2.0)
    stz = int((limit[2].astype('float32')) / 2.0)
    x_data = x_data[stx:stx + crop_size[0], sty:sty + crop_size[1], stz:stz + crop_size[2], :]

    ### get the patch
    return x_data[xx:xx+dims[0], yy:yy+dims[1], zz:zz+dims[2], :]


# load 3D test volume
class Test_Volume():
    def __init__(self, config, Iref, Imov):

        # define constants
        self.Iref = Iref
        self.Imov = Imov

        self._volume_list = []

        self.patch_size = config['patch_size']

        for f in os.listdir(config['test_dir']):
            if os.path.isfile(os.path.join(config['test_dir'], f)) \
                    and os.path.splitext(f)[1] == '.mat':

                self._volume_list += [os.path.join(config['test_dir'], f)]

        self._current_volume = 0
        self._num_volumes = len(self._volume_list)


    ### pre-load test data
    def load(self, vol_ind = None):
        if vol_ind is None:
            vol_ind = self._current_volume

        f_f = self._volume_list[vol_ind]
        img = self.load_volume(f_f)  ## load the whole image
        self._current_volume = (self._current_volume + 1) % len(self._volume_list)
        self.img = img

    def get_eval_feed_dict(self, st_ind, ref_bin, mov_bin):
        # extract ref and mov image patches from loaded volumes
        if self.patch_size:
            img = self.img[st_ind[0]:st_ind[0]+self.patch_size[0],
                         st_ind[1]:st_ind[1]+self.patch_size[1],
                         st_ind[2]:st_ind[2]+self.patch_size[2],:]
        else:
            img = self.img
        im1=[]
        im2=[]
        for bin in mov_bin:
            im1.append(img[...,ref_bin])
            im2.append(img[...,bin])

        im1 = np.asarray(im1)[..., np.newaxis]
        im2 = np.asarray(im2)[..., np.newaxis]

        return {self.Iref: im1.astype(np.float32),
                self.Imov: im2.astype(np.float32)}

    def load_volume(self, f_t):
        try:
            img_bins = sio.loadmat(f_t)['Itarget']
        except:
            img_bins = sio.loadmat(f_t)['msense']

        img_bins = np.array(img_bins)
        for n in range(img_bins.shape[-1]):
            im = img_bins[...,n]
            im = (im - np.min(im)) / (np.max(im) - np.min(im))
            img_bins[...,n]=im

        return img_bins
