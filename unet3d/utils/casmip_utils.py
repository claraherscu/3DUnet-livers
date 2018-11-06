"""
Utilities Adi's using in order keep Tensorflow simple as possible

"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import glob
import shutil
from scipy.misc import imsave
from scipy.misc import imresize
from scipy import ndimage
from PIL import Image, ImageDraw

from .casmip_load_utils import *


def empty_dir(path):
    for fn in os.listdir(path):
        file_path = os.path.join(path, fn)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(e)


def mkdir_prompt(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        if not os.listdir(path):
            return  # empty dir, as if just created
        response = input('Dir '+path+' exists. Empty it? ([y]/n): ')
        if response in ('', 'y', 'Y'):
            empty_dir(path)
        else:
            raise


def nii_to_slices(nii_image, nii_seg, rect_size, crop_steps):
    # print(nii_image.shape)
    nii_slices_true = []
    nii_slices_seg_true = []
    nii_slices_false = []
    nii_slices_seg_false = []
    num_of_img, x, y, channels = nii_image.shape
    x -= rect_size
    y -= rect_size
    minimum = rect_size*rect_size/2
    for k in range(num_of_img):
        for i in range(0, x, crop_steps):
            for j in range(0, y, crop_steps):
                # print(i,j,k)
                liver_pieces = nii_image[k, i:i+rect_size, j:j+rect_size, :]
                segmentation_pieces = nii_seg[k, i:i+rect_size, j:j+rect_size, :]
                flipped_liver_pieces = np.fliplr(liver_pieces)
                if np.count_nonzero(liver_pieces) < minimum:
                    continue
                if np.count_nonzero(segmentation_pieces) == 0:
                    nii_slices_true.append(liver_pieces)
                    nii_slices_seg_true.append([0, 1])
                    nii_slices_true.append(flipped_liver_pieces)
                    nii_slices_seg_true.append([0, 1])
                else:
                    nii_slices_false.append(liver_pieces)
                    nii_slices_seg_false.append([1, 0])
                    nii_slices_false.append(flipped_liver_pieces)
                    nii_slices_seg_false.append([1, 0])
#    nii_slices = np.asarray(nii_slices)
#    nii_slices_seg = np.asarray(nii_slices_seg)
    return np.array(nii_slices_true), np.array(nii_slices_seg_true), np.array(nii_slices_false), np.array(nii_slices_seg_false)


def release_list(a):
   del a[:]
   del a


def nii_to_np(path_to_nii, standardize=True, overwrite=False):
    """Convert nii files to raw numpy arrays and save them in another file, named nii_filename + '.npz'

    Parameters
    ----------
    path_to_nii : str
        Path to nii file

    standardize : bool
        Whether to standardize using standardize_images_globally

    overwrite : bool
        Wheter to overwrite or skip already converted files
    """
    if not overwrite and os.path.isfile(path_to_nii + '.npz'):
        return

    nii = load_nii_file(path_to_nii)
    if standardize:
        nii = standardize_images_globally(nii)

    np.savez_compressed(path_to_nii, *nii)


def all_nii_to_np(path_to_nii_dir, *args, **kwargs):
    files = glob.glob(os.path.join(path_to_nii_dir, '**', '*.nii'))
    files += glob.glob(os.path.join(path_to_nii_dir, '**', '*.nii.gz'))
    for f in files:
        nii_to_np(f, *args, **kwargs)


def nii_to_slices_to_np(path_to_nii, *args, **kwargs):
    nii_slices_true, nii_slices_seg_true, nii_slices_false, nii_slices_seg_false = nii_to_slices(*args, **kwargs)
    for name in ['nii_slices_true', 'nii_slices_seg_true', 'nii_slices_false', 'nii_slices_seg_false']:
        datum = np.array(eval(name))
        np.savez_compressed(path_to_nii + '_' + name, *datum)


def slices_from_np(path_to_np, indices=None):
    try: data = np.load(path_to_np)
    except FileNotFoundError: raise
    except OSError:
        if indices is None:
            return np.array([])
        else:
            # file is empty - nothing was saved
            raise

    keys = np.array(sorted(data.keys(), key=lambda x: int(x.split('_')[1])))
    if indices is not None:
        keys = keys[indices]

    all_data = [data[k] for k in keys]
    ret = np.array([None] * len(all_data), dtype=object)
    ret[:] = all_data[:]
    return ret


def imcrop_tosquare(img):
    """Make any image a square image.

    Parameters
    ----------
    img : np.ndarray
        Input image to crop, assumed at least 2d.

    Returns
    -------
    crop : np.ndarray
        Cropped image.
    """
    size = np.min(img.shape[:2])
    extra = img.shape[:2] - size
    crop = img
    for i in np.flatnonzero(extra):
        crop = np.take(crop, extra[i] // 2 + np.r_[:size], axis=i)
    return crop


def slice_montage(montage, img_h, img_w, n_imgs):
    """Slice a montage image into n_img h x w images.

    Performs the opposite of the montage function.  Takes a montage image and
    slices it back into a N x H x W x C image.

    Parameters
    ----------
    montage : np.ndarray
        Montage image to slice.
    img_h : int
        Height of sliced image
    img_w : int
        Width of sliced image
    n_imgs : int
        Number of images to slice

    Returns
    -------
    sliced : np.ndarray
        Sliced images as 4d array.
    """
    sliced_ds = []
    for i in range(int(np.sqrt(n_imgs))):
        for j in range(int(np.sqrt(n_imgs))):
            sliced_ds.append(montage[
                1 + i + i * img_h:1 + i + (i + 1) * img_h,
                1 + j + j * img_w:1 + j + (j + 1) * img_w])
    return np.array(sliced_ds)


def montage(images, saveto='montage.png'):
    """Draw all images as a montage separated by 1 pixel borders.

    Also saves the file to the destination specified by `saveto`.

    Parameters
    ----------
    images : numpy.ndarray
        Input array to create montage of.  Array should be:
        batch x height x width x channels.
    saveto : str
        Location to save the resulting montage image.

    Returns
    -------
    m : numpy.ndarray
        Montage image.
    """
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))
    if len(images.shape) == 4 and images.shape[3] == 3:
        m = np.ones(
            (images.shape[1] * n_plots + n_plots + 1,
             images.shape[2] * n_plots + n_plots + 1, 3)) * 0.5
    elif len(images.shape) == 4 and images.shape[3] == 1:
        m = np.ones(
            (images.shape[1] * n_plots + n_plots + 1,
             images.shape[2] * n_plots + n_plots + 1, 1)) * 0.5
    elif len(images.shape) == 3:
        m = np.ones(
            (images.shape[1] * n_plots + n_plots + 1,
             images.shape[2] * n_plots + n_plots + 1)) * 0.5
    else:
        raise ValueError('Could not parse image shape of {}'.format(
            images.shape))
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                m[1 + i + i * img_h:1 + i + (i + 1) * img_h,
                  1 + j + j * img_w:1 + j + (j + 1) * img_w] = this_img
    imsave(arr=np.squeeze(m), name=saveto)
    return m


def montage_npz(path, *args, **kwargs):
    npz = np.load(path)
    data = [item[1] for item in npz.items()]
    return montage(data, *args, **kwargs)


def show_images(images, rows=1, titles=None):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    rows (Default = 1): Number of rows in figure (number of cols is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(rows, np.ceil(n_images/float(rows)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()


def gauss(mean, stddev, ksize):
    """Use Tensorflow to compute a Gaussian Kernel.

    Parameters
    ----------
    mean : float
        Mean of the Gaussian (e.g. 0.0).
    stddev : float
        Standard Deviation of the Gaussian (e.g. 1.0).
    ksize : int
        Size of kernel (e.g. 16).

    Returns
    -------
    kernel : np.ndarray
        Computed Gaussian Kernel using Tensorflow.
    """
    g = tf.Graph()
    with tf.Session(graph=g):
        x = tf.linspace(-3.0, 3.0, ksize)
        z = (tf.exp(tf.negative(tf.pow(x - mean, 2.0) /
                           (2.0 * tf.pow(stddev, 2.0)))) *
             (1.0 / (stddev * tf.sqrt(2.0 * 3.1415))))
        return z.eval()


def gauss2d(mean, stddev, ksize):
    """Use Tensorflow to compute a 2D Gaussian Kernel.

    Parameters
    ----------
    mean : float
        Mean of the Gaussian (e.g. 0.0).
    stddev : float
        Standard Deviation of the Gaussian (e.g. 1.0).
    ksize : int
        Size of kernel (e.g. 16).

    Returns
    -------
    kernel : np.ndarray
        Computed 2D Gaussian Kernel using Tensorflow.
    """
    z = gauss(mean, stddev, ksize)
    g = tf.Graph()
    with tf.Session(graph=g):
        z_2d = tf.matmul(tf.reshape(z, [ksize, 1]), tf.reshape(z, [1, ksize]))
        return z_2d.eval()


def convolve(img, kernel):
    """Use Tensorflow to convolve a 4D image with a 4D kernel.

    Parameters
    ----------
    img : np.ndarray
        4-dimensional image shaped N x H x W x C
    kernel : np.ndarray
        4-dimensional image shape K_H, K_W, C_I, C_O corresponding to the
        kernel's height and width, the number of input channels, and the
        number of output channels.  Note that C_I should = C.

    Returns
    -------
    result : np.ndarray
        Convolved result.
    """
    g = tf.Graph()
    with tf.Session(graph=g):
        convolved = tf.nn.conv2d(img, kernel, strides=[1, 1, 1, 1], padding='SAME')
        res = convolved.eval()
    return res


def linear(x, n_output, name=None, activation=None, reuse=None):
    """Fully connected layer.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor to connect
    n_output : int
        Number of output neurons
    name : None, optional
        Scope to apply

    Returns
    -------
    op : tf.Tensor
        Output of fully connected layer.
    """
    if len(x.get_shape()) != 2:
        x = flatten(x, reuse=reuse)

    n_input = x.get_shape().as_list()[1]

    with tf.variable_scope(name or "fc", reuse=reuse):
        # W = tf.get_variable(
        #     name='W',
        #     shape=[n_input, n_output],
        #     dtype=tf.float32,
        #     initializer=tf.contrib.layers.xavier_initializer())
        W = weight_variable(shape=[n_input, n_output])

        # b = tf.get_variable(
        #     name='b',
        #     shape=[n_output],
        #     dtype=tf.float32,
        #     initializer=tf.constant_initializer(0.0))

        b = bias_variable(shape=[n_output])

        h = tf.nn.bias_add(
            name='h',
            value=tf.matmul(x, W),
            bias=b)

        if activation:
            h = activation(h)

        return h, W


def flatten(x, name=None, reuse=None):
    """Flatten Tensor to 2-dimensions.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor to flatten.
    name : None, optional
        Variable scope for flatten operations

    Returns
    -------
    flattened : tf.Tensor
        Flattened tensor.
    """
    with tf.variable_scope('flatten'):
        dims = x.get_shape().as_list()
        if len(dims) == 4:
            flattened = tf.reshape(
                x,
                shape=[-1, dims[1] * dims[2] * dims[3]])
        elif len(dims) == 2 or len(dims) == 1:
            flattened = x
        else:
            raise ValueError('Expected n dimensions of 1, 2 or 4.  Found:',
                             len(dims))

        return flattened


def weight_variable(shape):
    try: weight_variable.counter += 1
    except AttributeError: weight_variable.counter = 1
    return tf.get_variable('W' + str(weight_variable.counter), shape=shape, dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())


def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape, dtype=tf.float32)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(x, ksize=2, stride=2):
    return tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1], padding='SAME')


def standardize_images_globally(x):
    """convert the images to float32 and normalize them w.r.t. mean and std of all images, so the mean is 0 and std is 1."""
    x = x.astype(np.float32)
    std = np.std(x)
    adjusted_stddev = np.maximum(std, 1.0/np.sqrt(x.size))  # avoid division by zero in next line
    return (x - np.mean(x)) / adjusted_stddev


def standardize_images(x):
    """convert the images to float32 and normalize them, so the mean is 0 and std is 1."""
    x = x.astype(np.float32)
    r_x = x.reshape(x.shape[0], -1)
    std = np.std(r_x, 1)  # might contain zeros
    num_pixels = r_x.shape[1]
    adjusted_stddev = np.maximum(std, 1.0/np.sqrt(num_pixels))  # avoid division by zero in next line
    r_x = ((r_x.T - np.mean(r_x, 1)) / adjusted_stddev).T
    x = r_x.reshape(x.shape)

    return x


def conv2dpool(x, shape, activation=tf.nn.relu, use_bias=False):
    W = weight_variable(shape)
    c = conv2d(x, W)
    if use_bias:
        b = bias_variable([shape[-1]])
        c += b

    h = max_pool(activation(c))

    return h, W


def free_bytes(path):
    stat = os.statvfs(path)
    return stat.f_frsize * stat.f_bavail


def rotate_image(img, angle, pivot, crop=True):
    """rotate image at angle (degrees) around a pivot ([row, col]). maybe squeeze, expand_dims occur"""
    dim = img.ndim
    # pad so pivot is center
    pad_row = [img.shape[0] - pivot[0], pivot[0]]
    pad_col = [img.shape[1] - pivot[1], pivot[1]]
    imgP = np.pad(np.squeeze(img), [pad_row, pad_col], 'constant')
    # rotate around pivot
    imgR = ndimage.rotate(imgP, angle, reshape=False)

    # return cropped
    if crop:
        imgR = imgR[pad_row[0]: -pad_row[1], pad_col[0]: -pad_col[1]]

    if imgR.ndim < dim:
        imgR = np.expand_dims(imgR, dim)

    return imgR


def bounding_rect(polygon):
    """takes a polygon and returns a rectangle parallel to the axes"""
    xs = [q[0] for q in polygon]
    ys = [q[1] for q in polygon]
    return [[min(xs), min(ys)], [max(xs), max(ys)]]


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in degrees between vectors 'v1' and 'v2'"""
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))


def rotate2d(point, degrees, pivot=[0, 0]):
    """rotates point at angle degrees around pivot"""
    radians = np.radians(degrees)
    crad = np.cos(radians)
    srad = np.sin(radians)
    x = point[0] - pivot[0]
    y = point[1] - pivot[1]
    newx = x * crad - y * srad
    newy = x * srad + y * crad
    newx += pivot[0]
    newy += pivot[1]

    return newx, newy


def rotate_polygon(poly, *args, **kwargs):
    return [rotate2d(p, *args, **kwargs) for p in poly]


def translate_polygon(poly, xy):
    return [[p[0] + xy[0], p[1] + xy[1]] for p in poly]


def expand_rects(rects, clip_shapes=None, factor=2):
    widths = abs(rects[:, 1][:, 0] - rects[:, 0][:, 0])
    heights = abs(rects[:, 1][:, 1] - rects[:, 0][:, 1])
    mul = (factor - 1) / 2
    add_width = widths * mul
    add_height = heights * mul
    rects[:, 0][:, 0] -= add_width
    rects[:, 1][:, 0] += add_width
    rects[:, 0][:, 1] -= add_height
    rects[:, 1][:, 1] += add_height

    if clip_shapes is not None:
        rects[:, 0][:, 0] = rects[:, 0][:, 0].clip(0)
        rects[:, 1][:, 0] = rects[:, 1][:, 0].clip(0, clip_shapes[:, 0])
        rects[:, 0][:, 1] = rects[:, 0][:, 1].clip(0)
        rects[:, 1][:, 1] = rects[:, 1][:, 1].clip(0, clip_shapes[:, 1])

    return rects


def _extract_quads(slices, quads, slopes=None, mode='bounding', expand=False, angle=0):
    """angle is given for augmentation"""
    if slopes is None:
        slopes = [0] * len(quads)
    if mode == 'bounding':
        rects = np.array([bounding_rect(q) for q in quads]).astype(np.float32)
    elif mode == 'rotated':
        # NOTE: ignoring given slopes atm as they are fishy
        # assuming quads are convex... after sorting, the first and second points are adjacent, and have the largest y.
        # this represents the short side of the quad (hopefully a rotated rectangle)
        squads = np.array([sorted(quad, key=lambda q:q[1]) for quad in quads])
        # find the angle between adjacent points
        first_points = squads[:, 0]
        second_points = squads[:, 1]
        diff = second_points - first_points
        angles = np.arctan(diff[:, 1] / diff[:, 0]) + angle

        # rotate the slices and the quads to the opposite direction so the quads will be parallel to the axes
        slices = [rotate_image(s, -a, p, crop=False) for s, a, p in zip(slices, angles, first_points)]
        rotated_squads = [np.array(rotate_polygon(q, -a, p)) for q, a, p in zip(squads, angles, first_points)]
        rotated_squads = [translate_polygon(q, np.array(np.squeeze(s).shape) / 2 - p) for q, p, s in zip(rotated_squads, first_points, slices)]
        # force the quads to be rects parallel to the axes
        rects = np.array([bounding_rect(q) for q in rotated_squads])
    elif mode == 'bounded':
        raise NotImplementedError
    else:
        raise BaseException('wtf')

    if expand:
        rects = expand_rects(rects, np.array([s.shape for s in slices]))

    # return slices, rects
    return [s[rect[0][0]:rect[1][0], rect[0][1]:rect[1][1]] for s, rect in zip(slices, rects.round().astype(int))]


def extract_quads(slices, quads, angle_interval=[-5, 6], angle_delta=1, **kwargs):
    angle_range = np.arange(angle_interval[0], angle_interval[1], angle_delta)  # e.g. [-5, 6], 1 -> [-5, -4, ... , 4, 5]
    ret = []
    for angle in angle_range:
        ret += _extract_quads(slices, quads, angle=angle, **kwargs)
    return ret


def one_hot_encode(x, n_classes):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
     """
    return np.eye(n_classes)[x]


def imresize_batch(images, size, fn=''):
    ret = []
    for i, image in enumerate(images):
        try:
            ret.append(imresize(np.squeeze(image), size=size, mode='F').reshape([size[0], size[1], 1]))
        except ValueError as e:
            print(fn, i, repr(e))
            continue
    return ret


def rgb_change_hs_batch(images_rgb, hue, sat):
    ret = []
    for rgb in images_rgb:
        hsv = rgb_to_hsv(rgb)
        hsv[:, :, 0] = hue
        hsv[:, :, 1] = sat
        ret.append(hsv_to_rgb(hsv))

    return ret


def rgb_to_hsv(rgb):
    # Translated from source of colorsys.rgb_to_hsv
    # r,g,b should be a numpy arrays with values between 0 and 255.0
    # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
    rgb = rgb.astype('float')
    hsv = np.zeros_like(rgb)
    # in case an RGBA array was passed, just copy the A channel
    hsv[..., 3:] = rgb[..., 3:]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.max(rgb[..., :3], axis=-1)
    minc = np.min(rgb[..., :3], axis=-1)
    hsv[..., 2] = maxc
    mask = maxc != minc
    hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
    rc = np.zeros_like(r)
    gc = np.zeros_like(g)
    bc = np.zeros_like(b)
    rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
    gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
    bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
    hsv[..., 0] = np.select(
        [r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
    hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
    return hsv


def hsv_to_rgb(hsv):
    # Translated from source of colorsys.hsv_to_rgb
    # h,s should be a numpy arrays with values between 0.0 and 1.0
    # v should be a numpy array with values between 0.0 and 255.0
    # hsv_to_rgb returns an array of uints between 0 and 255.0
    rgb = np.empty_like(hsv)
    rgb[..., 3:] = hsv[..., 3:]
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = (h * 6.0).astype('uint8')
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i %= 6
    conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
    rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
    rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
    rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
    return rgb


def softmax(x, axis=None):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def get_dir_size(path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


def fill_poly(shape2d, poly, outline=True, fill=True):
    """return a 2d mask on an array with shape shape2d. poly is list of x,y tuple-like pairs"""
    poly = list(map(tuple, poly))  # PIL...

    img = Image.new('L', shape2d, 0)
    ImageDraw.Draw(img).polygon(poly, outline=outline, fill=fill)
    mask = np.array(img).T  # PIL works with x,y and not row,col

    return mask

