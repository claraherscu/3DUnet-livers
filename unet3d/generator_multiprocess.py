import keras
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from unet3d.data import write_data_to_file, open_data_file
from multiprocessing import Pool


class ClassDataGenerator(keras.utils.Sequence):
    """
    Classifer Data Generator. inherits keras.utils.Sequence, that provides multi-process iterator over the dataset.
    """

    def __init__(self, file_name, batch_size=1024, data_split=100, start=0, end=None, root_name_x='data',
                 root_name_y='truth', imgen_params=None, is_train=True, crop_size=(256, 256)):
        """
        initialization
        :param file_name: hd5 file name to load data from
        :param batch_size: Size of the batches that the training generator will provide
        :param data_split: How the training and validation data will be split. 0 means all the data will be used for
                validation and none of it will be used for training. 1 means that all the data will be used for training
                and none will be used for validation. Default is 0.8 or 80%.
        :param start: index to start reading from file
        :param end: index to end reading from file
        :param root_name_x: the name of the entry in the hdf5 where the X data is held
        :param root_name_y: the name of the entry in the hdf5 where the y data is held
        :param imgen_params: parameters for the keras ImageDataGenerator
        :param is_train:
        :param crop_size:
        """

        # TODO have to apply same augmentation on mask and image
        # TODO have to return patches and not whole images

        self.imgen = ImageDataGenerator(**imgen_params)
        # self.crop_size = np.array(crop_size)

        self.f = open_data_file(file_name, 'r')

        print('loading all the x data to memory...')
        self.x_all = getattr(self.f.root, root_name_x)[start:end]
        print('loaded')
        #self.x_all = getattr(self.f.root, root_name_x)

        print('loading all the y data to memory...')
        self.y_all = getattr(self.f.root, root_name_y)[start:end]
        print('loaded')

        self.x_shape = self.x_all[0].shape

        self.total_len = len(self.y_all)
        print(self.y_all.shape)
        self.batch_size = batch_size
        self.len_segment = int(self.total_len / data_split)
        # self.is_train = is_train

        self.shuffled_index = np.arange(self.total_len)
        if is_train:
            np.random.shuffle(self.shuffled_index)

        NUM_PROCESSORS = 10
        self.pool = Pool(NUM_PROCESSORS)

        # self.x_cur = self.x_all[self.shuffled_index[:self.len_segment]]
        # self.y_cur = self.y_all[self.shuffled_index[:self.len_segment]]
        # self.idx = 0
        # self.cur_seg_idx = 0

    def __len__(self):
        "denotes number of batches per epoch"
        return int(np.floor(self.total_len / self.batch_size))

    def terminate_pool(self):
        self.pool.terminate()

    def __getitem__(self, index):
        "generate one batch of data"
        # generate indices of the batch
        indices = self.shuffled_index[index*self.batch_size:(index+1)*self.batch_size]

        # generate data from indices
        batch_images = np.zeros((self.batch_size, ) + self.x_shape)
        # if len(batch_images.shape) < 4:
        #     batch_images = batch_images[..., np.newaxis]

        for i, ind in enumerate(indices):
            img = self.x_all[ind]
            # if len(img.shape) < 4:
            #     img = img[..., np.newaxis]
            batch_images[i] = img

        if self.is_train:
            ret = self.pool.map(self.imgen.random_transform, batch_images)
            batch_images = np.array(ret)

        middle = np.array(batch_images[0].shape[:2]) / 2
        st = (middle - self.crop_size / 2).round()
        batch_cropped = batch_images[:, int(st[0]):int(st[0])+self.crop_size[0], int(st[1]):int(st[1])+self.crop_size[1], ...]
        for i,b in enumerate(batch_cropped):
            if np.abs(b).sum() < 1e-3 or not np.isfinite(b).all():
                print('noooooooo', index, self.is_train, i, np.abs(b).sum(), indices)
        grades = self.y_all[indices]
        if not np.isfinite(grades).all():
            print('HOLLLLYYY SHIITTTTT', index, self.is_train)
        batch_y = keras.utils.to_categorical(grades, len(self.classes))

        return batch_cropped, batch_y

    def on_epoch_end(self):
        "re-shuffles indices after each epoch"
        if self.is_train:
            np.random.shuffle(self.shuffled_index)
