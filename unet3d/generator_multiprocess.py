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

    def __init__(self, file_name, batch_size=1024, data_split=0.8, start=0, end=None, root_name_x='data',
                 root_name_y='truth', imgen_params=None, seed=1, is_train=True, n_processors=4):
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
        :param seed: seed for random augmentations. will use same seed for data and masks to get the same augemntations
        :param is_train:
        :param patch_shape: (int, int, int) 3d shape of patches to be generated from data
        :param crop_size:
        :param patch_overlap: Number of pixels/voxels that will be overlapped in the validation data. (requires
                patch_shape to not be None)
        :param patch_start_offset: Tuple of length 3 containing integer values. Data will randomly be
                offset by a number of pixels between (0, 0, 0) and the given tuple. (default is None)
        :type n_processors: Number of processors to use in parallel for augmentations
        """

        self.imgen = ImageDataGenerator(**imgen_params)  # TODO: doesn't support 3D?
        self.maskgen = ImageDataGenerator(**imgen_params)
        self.seed = seed

        self.f = open_data_file(file_name, 'r')

        print('loading all the x data to memory...')
        self.x_all = getattr(self.f.root, root_name_x)[start:end]
        # self.x_all = np.squeeze(self.x_all)  # TODO: do we need this?
        print('loaded')

        print('loading all the y data to memory...')
        self.y_all = getattr(self.f.root, root_name_y)[start:end]
        # self.y_all = np.squeeze(self.y_all)  # TODO: do we need this?
        print('loaded')

        self.x_shape = self.x_all[0].shape  # on images it is (512, 512, 60), on patches (8, 8, 8)

        self.total_len = len(self.y_all)
        self.batch_size = batch_size
        self.is_train = is_train

        self.shuffled_index = np.arange(self.total_len)
        if is_train:
            np.random.shuffle(self.shuffled_index)

        self.pool = Pool(n_processors)

    def __len__(self):
        "denotes number of batches per epoch"
        return int(np.floor(self.total_len / self.batch_size))

    def terminate_pool(self):
        self.pool.terminate()

    def __getitem__(self, index):
        "generate one batch of data"
        # TODO this way we have batch_size images in a batch inside of batch_size patches -- too much
        # generate indices of the batch
        indices = self.shuffled_index[index*self.batch_size:(index+1)*self.batch_size]
        # if self.patch_shape:
        #     patch_indices = create_patch_index_list(indices,
        #                                             self.x_shape,
        #                                             self.patch_shape,
        #                                             self.patch_overlap,
        #                                             self.patch_start_offset)
        #
        # else:
        #     patch_indices = copy.copy(indices)

        # generate data from indices
        batch_images = np.zeros((self.batch_size, ) + self.x_shape)

        for i, ind in enumerate(indices):
            img = self.x_all[ind]
            batch_images[i] = img

        # TODO extract all patches of X and y (use pool?)

        # patch_images = np.zeros((self.batch_size, ) + self.patch_shape)
        # while len(patch_indices) > 0:
        #     index = patch_indices.pop()
        #     if self.patch_shape:

        # TODO: return augmentation - has error affine matrix has wrong number of rows
        # # augmentation
        # if self.is_train:
        #     rand_transform = partial(self.imgen.random_transform, seed=self.seed)
        #     ret = self.pool.map(rand_transform, batch_images)
        #     batch_images = np.array(ret)

        # generate data masks from indices
        batch_y = np.zeros((self.batch_size, ) + self.x_shape)
        for i, ind in enumerate(indices):
            img = self.y_all[ind]
            batch_y[i] = img

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
            np.random.shuffle(self.shuffled_index)
