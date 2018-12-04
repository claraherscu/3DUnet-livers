import os

from brats.train import config
from unet3d.prediction import run_validation_cases


def main():
    prediction_dir = os.path.abspath("prediction")

    # original prediction code
    run_validation_cases(validation_keys_file=config["validation_file"],
                         model_file=config["model_file"],
                         training_modalities=config["training_modalities"],
                         labels=config["labels"],
                         hdf5_file=os.path.abspath("brats/liver_data.h5"),
                         output_label_map=True,
                         output_dir=prediction_dir)

    # # prediction for patched cases
    # run_validation_cases(validation_keys_file=config["validation_file_patches"],
    #                      model_file=config["model_file"],
    #                      training_modalities=config["training_modalities"],
    #                      labels=config["labels"],
    #                      hdf5_file=config["patch_data_file"],
    #                      output_label_map=True,
    #                      output_dir=prediction_dir)


if __name__ == "__main__":
    main()
