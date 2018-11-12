import os
import time
import numpy as np
import tables

from .normalize import normalize_data_storage, reslice_image_set
from unet3d.generator import create_patch_index_list, get_data_from_file


def create_data_file(out_file, n_channels, n_samples, image_shape, storage_names=('data', 'truth', 'affine'),
                     affine_shape=(0, 4, 4), normalize=True, affine_dtype=tables.Float32Atom()):
    hdf5_file = tables.open_file(out_file, mode='w')
    filters = tables.Filters(complevel=5)  #, complib='blosc')  # suggested remove in https://github.com/ellisdg/3DUnetCNN/issues/58
    data_shape = tuple([0, n_channels] + list(image_shape))
    truth_shape = tuple([0, 1] + list(image_shape))

    if not normalize:
        data_storage = hdf5_file.create_earray(hdf5_file.root, storage_names[0], tables.Int8Atom(), shape=data_shape,
                                               filters=filters, expectedrows=n_samples)
    else:
        data_storage = hdf5_file.create_earray(hdf5_file.root, storage_names[0], tables.Float32Atom(), shape=data_shape,
                                               filters=filters, expectedrows=n_samples)
    truth_storage = hdf5_file.create_earray(hdf5_file.root, storage_names[1], tables.UInt8Atom(), shape=truth_shape,
                                            filters=filters, expectedrows=n_samples)
    affine_storage = hdf5_file.create_earray(hdf5_file.root, storage_names[2], affine_dtype, shape=affine_shape,
                                             filters=filters, expectedrows=n_samples)
    if len(storage_names) == 4:
        normalization_storage = hdf5_file.create_earray(hdf5_file.root, storage_names[3], tables.Float32Atom(),
                                                        shape=(0, 2), filters=filters, expectedrows=n_samples)
        # will hold mean and std of this case for later normalization
        return hdf5_file, data_storage, truth_storage, affine_storage, normalization_storage
    return hdf5_file, data_storage, truth_storage, affine_storage


def write_image_data_to_file(image_files, data_storage, truth_storage, image_shape, n_channels, affine_storage,
                             truth_dtype=np.uint8, crop=True, normalization_storage=None):
    for set_of_files in image_files:  # iterate through tuples of (modality1, modality2, ..., GT)
        images = reslice_image_set(set_of_files, image_shape, label_indices=len(set_of_files) - 1, crop=crop)
        subject_data = [image.get_data() for image in images]
        add_data_to_storage(data_storage, truth_storage, affine_storage, subject_data, images[0].affine, n_channels,
                            truth_dtype, normalization_storage=normalization_storage)
    return data_storage, truth_storage


def add_data_to_storage(data_storage, truth_storage, affine_storage, subject_data, affine, n_channels, truth_dtype,
                        normalization_storage=None):
    data_storage.append(np.asarray(subject_data[:n_channels])[np.newaxis])
    truth_storage.append(np.asarray(subject_data[n_channels], dtype=truth_dtype)[np.newaxis][np.newaxis])
    affine_storage.append(np.asarray(affine)[np.newaxis])
    if normalization_storage is not None:
        mean = np.mean(subject_data[:n_channels])#, axis=(1, 2, 3))
        std = np.std(subject_data[:n_channels])#, axis=(1, 2, 3))
        normalization_storage.append(np.asarray([mean, std])[np.newaxis])


def add_patch_data_to_storage(patch_data_storage, patch_truth_storage, patch_index_storage, img_patch_data,
                              truth_patch_data, index_patch_data, patch_normalization_storage, normalization_patch_data,
                              truth_dtype=np.uint8):
    """save unnormalized patch data as an hdf5 file"""
    patch_data_storage.append(np.asarray(img_patch_data, dtype=np.uint8)[np.newaxis])
    patch_truth_storage.append(np.asarray(truth_patch_data, dtype=truth_dtype)[np.newaxis][np.newaxis])
    patch_index_storage.append(np.asarray(index_patch_data)[np.newaxis])  # TODO: understand syntax
    patch_normalization_storage.append(np.asarray(normalization_patch_data)[np.newaxis])


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
        if not normalize:
            hdf5_file, data_storage, truth_storage, affine_storage, normalization_storage = \
                create_data_file(out_file,
                                 n_channels=n_channels,
                                 n_samples=n_samples,
                                 image_shape=image_shape,
                                 normalize=normalize,
                                 storage_names=('data', 'truth', 'index', 'normalization'))
        else:
            hdf5_file, data_storage, truth_storage, affine_storage = create_data_file(out_file,
                                                                                      n_channels=n_channels,
                                                                                      n_samples=n_samples,
                                                                                      image_shape=image_shape,
                                                                                      normalize=normalize)
            normalization_storage = None

    except Exception as e:
        # If something goes wrong, delete the incomplete data file
        os.remove(out_file)
        raise e

    write_image_data_to_file(training_data_files, data_storage, truth_storage, image_shape,
                             truth_dtype=truth_dtype, n_channels=n_channels, affine_storage=affine_storage, crop=crop,
                             normalization_storage=normalization_storage)
    if subject_ids:
        hdf5_file.create_array(hdf5_file.root, 'subject_ids', obj=subject_ids)
    if normalize:
        normalize_data_storage(data_storage)
    hdf5_file.close()
    return out_file


def write_patches_data_to_file(patches_data_file, patch_shape, data_file, patch_overlap=0,
                               indices=None):
    """
    write all the patches (data, GT, indices, and normalization factors (mean, std)) to a file,
    using multiple processes executing 'write_patches_data_to_file' function on a subset of the indices.
    :param patches_data_file: path filename where the patch data should be stored
    :param data_file: filename where the image data is stored
    :param patch_shape: 3-tuple shape of every patch
    :param patch_overlap: Number of pixels/voxels that will be overlapped in the data.
    (requires patch_shape to not be None)
    :param indices: indices of the subset of files that should be stored in this file (if None, take all files)
    :return: patches_data_file
    """
    print("Writing patches to destination:", patches_data_file)
    print("Working on cases:", indices)

    x_shape = data_file.root.data.shape
    n_channels = x_shape[1]  # n_channels is actually the number of modalities we have

    # collecting indices of all the patches of all the needed images
    if indices is None:
        indices = range(x_shape[0])

    # TODO: maybe check for empty patches?
    index_list = create_patch_index_list(index_list=indices,
                                         image_shape=x_shape[-3:],
                                         patch_shape=patch_shape,
                                         patch_overlap=patch_overlap)

    # creating hd5 file for the patches
    try:
        hdf5_file, data_storage, truth_storage, index_storage, normalization_storage = \
            create_data_file(patches_data_file,
                             n_channels=n_channels,
                             n_samples=len(index_list),
                             image_shape=patch_shape,
                             storage_names=('data', 'truth', 'index', 'normalization'),
                             affine_shape=(0, 4),
                             affine_dtype=tables.UInt8Atom(),
                             normalize=False)
    except Exception as e:
        # If something goes wrong, delete the incomplete data file
        os.remove(patches_data_file)
        raise e

    # iterating over all patch indices and adding them to the patches_data_file
    i = 1
    total = len(index_list)
    t = time.time()
    while len(index_list) > 0:
        if i % 100 == 0:
            print('done', str(i*100 / total)+'%\tat index', i, 'out of', total, '\ttook', time.time()-t, 'seconds')
        i += 1
        index = index_list.pop()

        # extracting data for this index
        data, truth, normalization = get_data_from_file(data_file, index, patch_shape=patch_shape,
                                                        read_normalization_factors=True)
        index, patch_index = index  # extracting the patch index
        patch_index = np.append(arr=np.array(patch_index), axis=0, values=np.array([index]))

        # writing the data to h5
        add_patch_data_to_storage(patch_data_storage=data_storage,
                                  patch_truth_storage=truth_storage,
                                  patch_index_storage=index_storage,
                                  patch_normalization_storage=normalization_storage,
                                  img_patch_data=data,
                                  truth_patch_data=truth,
                                  index_patch_data=patch_index,
                                  normalization_patch_data=normalization)

    hdf5_file.close()
    return patches_data_file


def open_data_file(filename, readwrite="r"):
    return tables.open_file(filename, readwrite)


def concatenate_data_files(out_filname, input_filenames):
    """
    concatenate given input filenames into one big hdf5 file.
    :param out_filname: name of output file
    :param input_filenames: list of names of input files
    :return:
    """
    # getting all necessary information from input files
    input_file = tables.open_file(input_filenames[0], "r")
    n_channels = input_file.root.data.shape[1]  # number of channels in the data
    patch_shape = input_file.root.data.shape[-3:]  # shape of every patch in the data
    input_file.close()

    # total number of entries in the files
    n_entries = 0
    for filename in input_filenames:
        with tables.open_file(filename, "r") as input_file:
            n_entries += input_file.root.data.shape[0]
    print('n_entries is:', n_entries)

    # creating hd5 file for the patches
    try:
        hdf5_file, data_storage, truth_storage, index_storage, normalization_storage = \
            create_data_file(out_filname,
                             n_channels=n_channels,
                             n_samples=n_entries,
                             image_shape=patch_shape,
                             storage_names=('data', 'truth', 'index', 'normalization'),
                             affine_shape=(0, 4),
                             affine_dtype=tables.UInt8Atom(),
                             normalize=False)
    except Exception as e:
        # If something goes wrong, delete the incomplete data file
        os.remove(out_filname)
        raise e
    print('succesfully created file', out_filname)

    # writing data to file
    t = time.time()
    for filename in input_filenames:
        print('appending data from file', filename)
        with tables.open_file(filename, "r") as input_file:
            data = input_file.root.data[:]
            data_storage.append(np.asarray(data, dtype=np.uint8))  # TODO: maybe with np.newaxis?
            print('appended data')
            truth = input_file.root.truth[:]
            truth_storage.append(truth)
            print('appended truth')
            index = input_file.root.index[:]
            index_storage.append(index)
            print('appended index')
            norm = input_file.root.normalization[:]
            normalization_storage.append(norm)
            print('appended normalization\nDone file')
            print('took:', time.time() - t)

    hdf5_file.close()
