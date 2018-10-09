import os

import numpy as np
import tables

from .normalize import normalize_data_storage, reslice_image_set
from unet3d.generator import create_patch_index_list, get_data_from_file


def create_data_file(out_file, n_channels, n_samples, image_shape, storage_names=('data', 'truth', 'affine'),
                     affine_shape=(0,4,4)):
    hdf5_file = tables.open_file(out_file, mode='w')
    filters = tables.Filters(complevel=5)  #, complib='blosc')  # suggested remove in https://github.com/ellisdg/3DUnetCNN/issues/58
    data_shape = tuple([0, n_channels] + list(image_shape))
    truth_shape = tuple([0, 1] + list(image_shape))
    data_storage = hdf5_file.create_earray(hdf5_file.root, storage_names[0], tables.Float32Atom(), shape=data_shape,
                                           filters=filters, expectedrows=n_samples)
    truth_storage = hdf5_file.create_earray(hdf5_file.root, storage_names[1], tables.UInt8Atom(), shape=truth_shape,
                                            filters=filters, expectedrows=n_samples)
    affine_storage = hdf5_file.create_earray(hdf5_file.root, storage_names[2], tables.Float32Atom(), shape=affine_shape,
                                             filters=filters, expectedrows=n_samples)
    return hdf5_file, data_storage, truth_storage, affine_storage


def write_image_data_to_file(image_files, data_storage, truth_storage, image_shape, n_channels, affine_storage,
                             truth_dtype=np.uint8, crop=True):
    for set_of_files in image_files:  # iterate through tuples of (modality1, modality2, ..., GT)
        images = reslice_image_set(set_of_files, image_shape, label_indices=len(set_of_files) - 1, crop=crop)
        subject_data = [image.get_data() for image in images]
        add_data_to_storage(data_storage, truth_storage, affine_storage, subject_data, images[0].affine, n_channels,
                            truth_dtype)
    return data_storage, truth_storage


def add_data_to_storage(data_storage, truth_storage, affine_storage, subject_data, affine, n_channels, truth_dtype):
    data_storage.append(np.asarray(subject_data[:n_channels])[np.newaxis])
    truth_storage.append(np.asarray(subject_data[n_channels], dtype=truth_dtype)[np.newaxis][np.newaxis])
    affine_storage.append(np.asarray(affine)[np.newaxis])


def add_patch_data_to_storage(patch_data_storage, patch_truth_storage, patch_index_storage, img_patch_data,
                              truth_patch_data, index_patch_data, truth_dtype=np.uint8):
    patch_data_storage.append(np.asarray(img_patch_data)[np.newaxis])
    patch_truth_storage.append(np.asarray(truth_patch_data, dtype=truth_dtype)[np.newaxis][np.newaxis])
    patch_index_storage.append(np.asarray(index_patch_data)[np.newaxis])  # TODO: understand syntax


def write_data_to_file(training_data_files, out_file, image_shape, truth_dtype=np.uint8, subject_ids=None,
                       normalize=True, crop=True):
    """
    Takes in a set of training images and writes those images to an hdf5 file.
    :param training_data_files: List of tuples containing the training data files. The modalities should be listed in
    the same order in each tuple. The last item in each tuple must be the labeled image. 
    Example: [('sub1-T1.nii.gz', 'sub1-T2.nii.gz', 'sub1-truth.nii.gz'), 
              ('sub2-T1.nii.gz', 'sub2-T2.nii.gz', 'sub2-truth.nii.gz')]
    :param out_file: Where the hdf5 file will be written to.
    :param image_shape: Shape of the images that will be saved to the hdf5 file.
    :param truth_dtype: Default is 8-bit unsigned integer. 
    :return: Location of the hdf5 file with the image data written to it. 
    """
    n_samples = len(training_data_files)
    n_channels = len(training_data_files[0]) - 1  # n_channels is actually the number of modalities we have

    try:
        hdf5_file, data_storage, truth_storage, affine_storage = create_data_file(out_file,
                                                                                  n_channels=n_channels,
                                                                                  n_samples=n_samples,
                                                                                  image_shape=image_shape)
    except Exception as e:
        # If something goes wrong, delete the incomplete data file
        os.remove(out_file)
        raise e

    write_image_data_to_file(training_data_files, data_storage, truth_storage, image_shape,
                             truth_dtype=truth_dtype, n_channels=n_channels, affine_storage=affine_storage, crop=crop)
    if subject_ids:
        hdf5_file.create_array(hdf5_file.root, 'subject_ids', obj=subject_ids)
    if normalize:
        normalize_data_storage(data_storage)
    hdf5_file.close()
    return out_file


def write_patches_data_to_file(patches_data_file, patch_shape, n_samples, n_channels, data_file, patch_overlap=0):
    """
    write all the patches (data, GT and indices) to a file
    :param patches_data_file: path filename where the patch data should be stored
    :param data_file: filename where the image data is stored
    :param patch_shape: 3-tuple shape of every patch
    :param patch_overlap: Number of pixels/voxels that will be overlapped in the data.
    (requires patch_shape to not be None)
    :return:
    """
    # creating hd5 file for the patches
    try:
        hdf5_file, data_storage, truth_storage, index_storage = create_data_file(patches_data_file,
                                                                                 n_channels=n_channels,
                                                                                 n_samples=n_samples,
                                                                                 image_shape=patch_shape,
                                                                                 storage_names=('data',
                                                                                                'truth',
                                                                                                'index'),
                                                                                 affine_shape=(0, 3))
    except Exception as e:
        # If something goes wrong, delete the incomplete data file
        os.remove(patches_data_file)
        raise e

    x_shape = data_file.root.data.shape
    # collecting indices of all the patches of all the images
    index_list = create_patch_index_list(index_list=range(x_shape[0]),
                                         image_shape=x_shape[-3:],
                                         patch_shape=patch_shape,
                                         patch_overlap=patch_overlap)

    # iterating over all patch indices and adding them to the patches_data_file
    while len(index_list) > 0:
        index = index_list.pop()
        data, truth = get_data_from_file(data_file, index, patch_shape=patch_shape)  # extracting data for this index
        index, patch_index = index  # extracting the patch index

        # writing the data to h5
        add_patch_data_to_storage(patch_data_storage=data_storage,
                                  patch_truth_storage=truth_storage,
                                  patch_index_storage=index_storage,
                                  img_patch_data=data,
                                  truth_patch_data=truth,
                                  index_patch_data=patch_index)

    hdf5_file.close()
    return patches_data_file


def open_data_file(filename, readwrite="r"):
    return tables.open_file(filename, readwrite)
