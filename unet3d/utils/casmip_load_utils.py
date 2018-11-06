import nibabel as nib
import dicom2nifti as d2n
import tempfile
import os


def get_ornt_trans(affine, order='SLA'):
    """
    Returns the needed orientation tansform from affine to desired order
    see nipy.org/nibabel/coordinate_systems
    :param affine: the current affine transform
    :param order: desired orientation. defaults to SLA, as we use with tf.
    :return: the orientation transform
    """
    orig_ornt = nib.orientations.io_orientation(affine)
    new_ornt = nib.orientations.axcodes2ornt(order)
    ornt_trans = nib.orientations.ornt_transform(orig_ornt, new_ornt)

    return ornt_trans


def reorder_voxels(orig_voxels, affine, order='SLA'):
    """
    Rorients img to the desired order, which for our usage defaults to SLA, returning the roriented voxels.
    see nipy.org/nibabel/coordinate_systems
    :param orig_voxels: the 3D scan
    :param affine: the current affine transform
    :param order: desired orientation. defaults to SLA, as we use with tf.
    :return: the reordered 3D image voxels
    """
    ornt_trans = get_ornt_trans(affine, order)
    voxels = nib.orientations.apply_orientation(orig_voxels, ornt_trans)

    return voxels


def load_dcm_case(dirpath, order='SLA', ret_also_meta=False):
    """
    Loads all files in dirpath to an image data ready for tf.
    Currently uses dicom2nifti as a workaround.
    :param dirpath: input path that holds dicom files of a single case
    :param order: desired orientation. defaults to SLA, as we use with tf.
    :param ret_also_meta: whether to return also the whole file, including metadata.
    :return loaded image: reshaped tensorflow ready of the entire nii file
    """
    # bleh.
    tmpfile = tempfile.mktemp() + '.nii.gz'
    d2n.dicom_series_to_nifti(dirpath, tmpfile)
    return load_nii_file(tmpfile, order, ret_also_meta)


def load_nii_file(path, order='SLA', ret_also_meta=False):
    """
    Loads file to a reshaped image data ready for tf.
    :param path: path to nii file
    :param order: desired orientation. defaults to SLA, as we use with tf.
    :param ret_also_meta: whether to return also the whole file, including metadata.
    :return loaded image: reshaped tensorflow ready of the entire nii file
    """
    img = nib.load(path)
    affine = img.affine
    voxels = reorder_voxels(img.get_data(), affine, order)
    if ret_also_meta:
        return voxels, img
    else:
        return voxels


def update_progress(progress):
    print('\r[{0}] {1}%'.format('#'*int(progress/10), progress))


def load_nii_folder(nii_path):
    img_list = []
    number_of_files = 0
    for file in sorted(os.listdir(nii_path)):
        if file.endswith(".nii") or file.endswith(".nii.gz"):
            number_of_files += 1
    print("found " + str(number_of_files) + " files in " + nii_path)
    number_of_iteration = 0
    for file in sorted(os.listdir(nii_path)):
        if file.endswith(".nii") or file.endswith(".nii.gz"):
            number_of_iteration += 1
            update_progress(int((number_of_iteration/number_of_files)*100))
            nii_filename = os.path.join(nii_path, file)
            # print(os.path.join(nii_path, file))
            img = load_nii_file(nii_filename)
            img_list.append(img)
    print("Finished loading " + nii_path)
    return img_list
