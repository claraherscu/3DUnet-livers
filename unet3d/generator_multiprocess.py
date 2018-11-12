import keras
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from unet3d.data import open_data_file
from multiprocessing import Pool
from functools import partial
from unet3d.generator import data_generator
from unet3d.generator import create_patch_index_list
import copy


class ClassDataGenerator(keras.utils.Sequence):
    """
    Classifer Data Generator. inherits keras.utils.Sequence, that provides multi-process iterator over the dataset.
    """

    def __init__(self, file_name, indices, batch_size=1024, x_shape=None, root_name_x='data',
                 root_name_y='truth', root_name_norm='normalization', imgen_params=None, seed=1,
                 is_train=True, n_processors=4):
        """
        initialization
        :param file_name: hd5 file name to load data from
        :param indices: indices to read from file
        :param batch_size: Size of the batches that the training generator will provide
        :param root_name_x: the name of the entry in the hdf5 where the X data is held
        :param root_name_y: the name of the entry in the hdf5 where the y data is held
        :type root_name_norm: the name of the entry in the hdf5 where the normalization data is held
        :param imgen_params: parameters for the keras ImageDataGenerator
        :param seed: seed for random augmentations. will use same seed for data and masks to get the same augemntations
        :param is_train:
        :type n_processors: Number of processors to use in parallel for augmentations
        """
        self.index = indices

        self.imgen = ImageDataGenerator(**imgen_params)  # TODO: doesn't support 3D?
        self.maskgen = ImageDataGenerator(**imgen_params)
        self.seed = seed

        self.f = open_data_file(file_name, 'r')
        self.root_name_x = root_name_x
        self.root_name_y = root_name_y
        self.root_name_norm = root_name_norm

        self.x_table = self.f.root[self.root_name_x]
        self.y_table = self.f.root[self.root_name_y]
        self.norm_table = self.f.root[self.root_name_norm]

        self.x_shape = x_shape  # on images it is (512, 512, 60), on patches (8, 8, 8)

        self.total_len = len(self.index)
        self.batch_size = batch_size
        self.is_train = is_train
        self.steps_per_epoch = np.floor(self.total_len / self.batch_size).astype(np.int)

        if is_train:
            np.random.shuffle(self.index)

        self.n_processors = n_processors

    def __len__(self):
        "denotes number of batches per epoch"
        return int(np.floor(self.total_len / self.batch_size))

    @staticmethod
    def normalize(data):
        """
        normalize the data using given normalization factors (mean, std)
        :param data: tuple (data, normalization factors)
        :return: normalized data
        """
        data, norm_factors = data
        data = data.astype(np.float32)
        data -= norm_factors[0]
        data /= norm_factors[1]
        return data

    def __getitem__(self, index):
        "generate one batch of data"
        # generate indices of the batch
        indices = self.index[index*self.batch_size:(index+1)*self.batch_size]

        # generate data from indices
        batch_images = self.x_table[indices, :]

        # normalize the data
        norm_factors = self.norm_table[indices, :]
        # TODO find a more efficient way to create this array
        data_to_normalize = [(batch_images[i], norm_factors[i]) for i in range(batch_images.shape[0])]
        with Pool(self.n_processors) as pool:
            batch_images = pool.map(self.normalize, data_to_normalize)

        batch_images = np.asarray(batch_images)

        # TODO: return augmentation - has error affine matrix has wrong number of rows
        # # augmentation
        # if self.is_train:
        #     rand_transform = partial(self.imgen.random_transform, seed=self.seed)
        #     ret = self.pool.map(rand_transform, batch_images)
        #     batch_images = np.array(ret)

        # generate data masks from indices
        batch_y = self.y_table[indices, :]

        # TODO: return augmentation
        # # same augmentation for y
        # if self.is_train:
        #     rand_transform_y = partial(self.maskgen.random_transform, seed=self.seed)
        #     ret_y = self.pool.map(rand_transform_y, batch_images)
        #     batch_y = np.array(ret_y)

        return batch_images, batch_y

    def on_epoch_end(self):
        "re-shuffles indices after each epoch"
        if self.is_train:
            np.random.shuffle(self.index)
