import os
import glob
import numpy as np
from argparse import ArgumentParser

from unet3d.data import write_data_to_file, open_data_file, write_patches_data_to_file
from unet3d.generator import get_training_and_validation_generators, get_validation_split, get_patches_validation_split
from unet3d.model import unet_model_3d
from unet3d.training import load_old_model, train_model, get_callbacks
from unet3d.generator_multiprocess import ClassDataGenerator

# ## Debug mode
# import keras.backend as K
# from tensorflow.python import debug as tf_debug
# sess = K.get_session()
# sess = tf_debug.LocalCLIDebugWrapperSession(sess)
# K.set_session(sess)
# ##

# argument parser for running patch creation from command-line
parser = ArgumentParser()
parser.add_argument('-f', '--filename', dest="filename", help="write patches to file, should be h5")
parser.add_argument('-i', '--index', dest="index", nargs="+", type=int, help="take files at these indices to write")


config = dict()
config["pool_size"] = (2, 2, 2)  # pool size for the max pooling operations
config["image_shape"] = (512, 512, 60)  # This determines what shape the images will be cropped/resampled to.
config["patch_shape"] = (8, 8, 8)  # switch to None to train on the whole image
config["labels"] = [1]  # (1, 2, 4)  # the label numbers on the input image TODO -- why was this (1,2,4)
config["n_labels"] = len(config["labels"])
config["all_modalities"] = ["CT"]
config["training_modalities"] = config["all_modalities"]  # change this if you want to only use some of the modalities
config["nb_channels"] = len(config["training_modalities"])
if "patch_shape" in config and config["patch_shape"] is not None:
    config["input_shape"] = tuple([config["nb_channels"]] + list(config["patch_shape"]))
else:
    config["input_shape"] = tuple([config["nb_channels"]] + list(config["image_shape"]))
config["truth_channel"] = config["nb_channels"]
config["deconvolution"] = True  # if False, will use upsampling instead of deconvolution
config["imgen_args"] = dict(horizontal_flip=True,
                            zoom_range=0.1,
                            rotation_range=3)  # arguments for ImageDataGenerator setting data augmentation
config["imgen_seed"] = np.random.randint(1e+5)  # random seed for the image augmentation, the same will be used for image&mask

config["batch_size"] = 64  # originally 12, trying bigger size
config["validation_batch_size"] = 12  # originally 12, reducing to reduce memory use
config["n_epochs"] = 100  # cutoff the training after this many epochs
config["steps_per_epoch"] = 100 #5000 # limiting epoch length to better see convergence
config["patience"] = 10  # learning rate will be reduced after this many epochs if the validation loss is not improving
config["early_stop"] = 50  # training will be stopped after this many epochs without the validation loss improving
config["initial_learning_rate"] = 0.00001
config["learning_rate_drop"] = 0.5  # factor by which the learning rate will be reduced
config["validation_split"] = 0.8  # portion of the data that will be used for training
config["flip"] = False  # augments the data by randomly flipping an axis during
config["permute"] = False  # data shape must be a cube. Augments the data by permuting in various directions
config["distort"] = None  # switch to None if you want no distortion
config["augment"] = config["flip"] or config["distort"]
config["validation_patch_overlap"] = 0  # if > 0, during training, validation patches will be overlapping
config["training_patch_start_offset"] = (16, 16, 16)  # randomly offset the first patch index by up to this offset
config["skip_blank"] = True  # if True, then patches without any target will be skipped

config["write_patches"] = True
config["data_file"] = os.path.abspath("brats/liver_data_unnormalized.h5")
config["patch_data_file"] = os.path.abspath("brats/data_liver_segmentation_patches/liver_patches_int_data_000_130.h5")  # holds the patched data for training
config["model_file"] = os.path.abspath("brats/liver_segmentation_model_patches.h5")
config["training_file"] = os.path.abspath("brats/training_ids_cases.pkl")
config["validation_file"] = os.path.abspath("brats/validation_ids_cases.pkl")
config["training_file_patches"] = os.path.abspath("brats/training_ids_patches.pkl")
config["validation_file_patches"] = os.path.abspath("brats/validation_ids_patches.pkl")
config["patch_index_file"] = os.path.abspath("brats/data_liver_segmentation_patches/case_patch_index.txt")  # file that holds case index:patch index mapping
config["overwrite"] = False  # If True, will previous files. If False, will use previously written files.


def fetch_training_data_files():
    training_data_files = list()
    for subject_dir in glob.glob(os.path.join(os.path.dirname(__file__), "data_liver_segmentation", "preprocessed", "volume-*")):  # after fixing permissions, replace "volume-*" with "*"
        subject_files = list()
        for modality in config["training_modalities"] + ["Liver"]:
            subject_files.append(os.path.join(subject_dir, modality + ".nii.gz"))
        training_data_files.append(tuple(subject_files))
    return training_data_files


def main(overwrite=False, args=None):
    # convert input images into an hdf5 file
    if overwrite or not os.path.exists(config["data_file"]):
        training_files = fetch_training_data_files()

        write_data_to_file(training_files, config["data_file"], image_shape=config["image_shape"], normalize=False)
    data_file_opened = open_data_file(config["data_file"])

    # creating patches file
    if config["write_patches"]:
        if overwrite or not os.path.exists(config["patch_data_file"]):
            if args is not None:
                file = os.path.abspath(args.filename)
                index = args.index
            else:
                file = config["patch_data_file"]
                index = None
            patches_data_file = write_patches_data_to_file(patches_data_file=file,
                                                           patch_shape=config["patch_shape"],
                                                           data_file=data_file_opened,
                                                           indices=index)

    if not overwrite and os.path.exists(config["model_file"]):
        model = load_old_model(config["model_file"])
    else:
        # instantiate new model
        model = unet_model_3d(input_shape=config["input_shape"],
                              pool_size=config["pool_size"],
                              n_labels=config["n_labels"],
                              initial_learning_rate=config["initial_learning_rate"],
                              deconvolution=config["deconvolution"])

    # split to train and validation
    training_list, validation_list = get_validation_split(data_file_opened,
                                                          data_split=config["validation_split"],
                                                          overwrite=config["overwrite"],
                                                          training_file=config["training_file"],
                                                          validation_file=config["validation_file"])

    # translate the training_list and validation_list to list of the patch indices
    training_list_patches, validation_list_patches = get_patches_validation_split(
        config["patch_index_file"],
        training_list,
        validation_list,
        training_file_patches=config["training_file_patches"],
        validation_file_patches=config["validation_file_patches"],
        overwrite=config["overwrite"])

    # create generator
    training_gen = ClassDataGenerator(config["patch_data_file"],
                                      indices=training_list_patches,
                                      imgen_params=config["imgen_args"],
                                      batch_size=config["batch_size"],
                                      x_shape=config["patch_shape"],
                                      seed=config["imgen_seed"])
    validation_gen = ClassDataGenerator(os.path.abspath('brats/data_liver_segmentation_patches/liver_patches_int_data_000_130_copy.h5'),
                                        indices=validation_list_patches,
                                        imgen_params=config["imgen_args"],
                                        batch_size=config["validation_batch_size"],
                                        x_shape=config["patch_shape"],
                                        seed=config["imgen_seed"])

    # run training
    model.fit_generator(generator=training_gen,
                        steps_per_epoch=config["steps_per_epoch"],
                        epochs=config["n_epochs"],
                        validation_data=validation_gen,
                        validation_steps=config["steps_per_epoch"],
                        callbacks=get_callbacks(config["model_file"],
                                                initial_learning_rate=config["initial_learning_rate"],
                                                learning_rate_drop=config["learning_rate_drop"],
                                                learning_rate_epochs=None,
                                                learning_rate_patience=config["patience"],
                                                early_stopping_patience=config["early_stop"]),
                        use_multiprocessing=True)

    training_gen.f.close()
    validation_gen.f.close()
    data_file_opened.close()

if __name__ == "__main__":
    main(overwrite=config["overwrite"])
