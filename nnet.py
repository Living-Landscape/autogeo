# interactive segmentation https://github.com/saic-vul/ritm_interactive_segmentation

import os
import random

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow_addons import layers as alayers
from PIL import Image, ImageEnhance, ImageOps


def gaussian_blur(x, kernel_size=11, sigma=5):
    """
    Blur images
    """
    ax = tf.range(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
    xx, yy = tf.meshgrid(ax, ax)
    kernel = tf.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel = kernel / tf.reduce_sum(kernel)
    kernel = tf.tile(kernel[..., tf.newaxis], [1, 1, 1])

    x = x[..., tf.newaxis]
    x_pad = (kernel_size // 2, kernel_size // 2)
    x = tf.pad(x, [(0, 0), x_pad, x_pad, (0, 0)], 'reflect')
    x = tf.nn.depthwise_conv2d(x, kernel[..., tf.newaxis], [1, 1, 1, 1], padding='VALID')

    return x[..., 0]


def detect_edges(x):
    """
    Detect edges inside the images
    """
    grad_components = tf.image.sobel_edges(x)
    grad_mag_components = grad_components**2
    grad_mag_square = tf.reduce_sum(grad_mag_components,axis=-1)

    return tf.sqrt(grad_mag_square)


def compute_pixel_weights(x):
    """
    Compute pixel weights
    """
    # TODO figure out the shape
    #shape = tf.shape(x)
    shape = (16, 256, 256, 3)

    # quantize inputs to (256 // 16) ** 3 = 4069 values
    quantized = x // 16
    if shape[-1] == 1:
        quantized = quantized[..., 0]
    elif shape[-1] == 3:
        quantized = quantized[..., 0] + 16 * quantized[..., 1] + 256 * quantized[..., 2]
    else:
        assert False
    flattened = tf.reshape(quantized, (-1,))
    values, indices, counts = tf.unique_with_counts(flattened)
    pixel_count = shape[0] * shape[1] * shape[2]
    min_counts = pixel_count // 16384
    max_counts = pixel_count // 256
    counts = tf.clip_by_value(counts, min_counts, max_counts)
    sum_counts = tf.reduce_sum(counts)
    weights = (min_counts + max_counts - counts) / sum_counts
    weights = tf.cast(weights, tf.float32)

    # create pixel mask
    pixel_mask = tf.gather(weights, indices)
    pixel_mask = tf.reshape(pixel_mask, shape[:3])
    pixel_mask = pixel_mask / tf.reduce_max(pixel_mask)

    return pixel_mask


def compute_edge_weights(x):
    """
    Compute edge weights
    """
    if x.shape[-1] == 3:
        x = rgb_to_oklab(x)

    edge_mask = detect_edges(x[..., :1])[..., 0]
    edge_mask = tf.cast(edge_mask, tf.float32)
    edge_mask = edge_mask / tf.reduce_max(edge_mask, axis=(1, 2), keepdims=True)

    return edge_mask


def compute_image_weights(x):
    """
    Compute image weights
    """
    pixel_mask = compute_pixel_weights(x)
    edge_mask = compute_edge_weights(x)

    mask = gaussian_blur(pixel_mask + edge_mask)
    mask = mask / tf.reduce_sum(mask, axis=(1, 2), keepdims=True)

    return mask


def unsupervised_loss(y_true, y_pred):
    """
    Weighted L1 loss
    """
    y_true = tf.cast(y_true, tf.float32)

    #l1 = tf.abs(y_true - y_pred)
    #weights = compute_image_weights(y_true)

    #return tf.reduce_mean(l1 * weights, axis=(1, 2))

    l1 = tf.reduce_sum(tf.abs(y_true - y_pred), axis=-1)
    return tf.reduce_mean(l1, axis=(1, 2))


def l1_loss(y_true, y_pred):
    """
    L1 loss
    """
    y_true = tf.cast(y_true, tf.float32)

    l1 = tf.reduce_sum(tf.abs(y_true - y_pred), axis=-1)

    return tf.reduce_mean(l1, axis=(1, 2))


def dice_loss(y_true, y_pred, weights=None, weights_sum=None):
    """
    Dice loss
    """
    eps = 1e-6

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.sigmoid(y_pred)

    if weights is None:
        weights = 1
        weights_sum = 1

    nominator = 2 * tf.reduce_sum(weights * y_true * y_pred, axis=(1, 2)) / weights_sum
    denominator = tf.reduce_sum(weights * (y_true + y_pred), axis=(1, 2)) / weights_sum

    return 1 - (nominator + eps) / (denominator + eps)


def weak_loss(y_true, y_pred):
    """
    Combined dice and cross entropy loss
    """
    eps = 1e-6

    y_true = tf.cast(y_true, tf.float32)

    # pixel weights
    positive = y_true
    negative = (1 - y_true)

    shape = tf.shape(y_true)
    all_sum = tf.cast(shape[0] * shape[1] * shape[2], tf.float32)
    pos_weight = tf.reduce_sum(positive, axis=(0, 1, 2), keepdims=True) / all_sum
    neg_weight = tf.reduce_sum(negative, axis=(0, 1, 2), keepdims=True) / all_sum
    weights = eps + (1 - pos_weight) * positive + (1 - neg_weight) * negative
    weights_sum = tf.reduce_sum(weights, axis=(0, 1, 2))

    # cross entropy
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred)
    cross_entropy = tf.reduce_sum(weights * cross_entropy, axis=(0, 1, 2)) / weights_sum

    return cross_entropy


def mask_edges(masks):
    """
    Return mask of widen edges
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    filters = tf.convert_to_tensor(kernel, masks.dtype)
    filters = tf.expand_dims(filters, axis=-1)

    eroded = tf.nn.erosion2d(
        masks,
        filters=filters,
        strides=(1, 1, 1, 1),
        padding='SAME',
        data_format='NHWC',
        dilations=(1, 1, 1, 1),
    ) + 1
    dilated = tf.nn.dilation2d(
        masks,
        filters=filters,
        strides=(1, 1, 1, 1),
        padding='SAME',
        data_format='NHWC',
        dilations=(1, 1, 1, 1),
    ) - 1

    return dilated - eroded


def supervised_loss(y_true, y_pred):
    """
    Supervised loss
    """
    eps = 1e-6

    y_true = tf.cast(y_true, tf.float32)

    # area masks
    edges = mask_edges(y_true)
    positive = (1 - edges) * y_true
    negative = (1 - edges) * (1 - y_true)

    # area weights
    shape = tf.shape(y_true)
    all_sum = tf.cast(shape[0] * shape[1] * shape[2], tf.float32)
    pos_weight = tf.reduce_sum(positive, axis=(0, 1, 2), keepdims=True) / all_sum
    neg_weight = tf.reduce_sum(negative, axis=(0, 1, 2), keepdims=True) / all_sum
    edge_weight = tf.reduce_sum(edges, axis=(0, 1, 2), keepdims=True) / all_sum

    # combine weights
    edge_coef = 4
    weights = eps + (1 - pos_weight) * positive + (1 - neg_weight) * negative + edge_coef * (1 - edge_weight) * edges
    weights_sum = tf.reduce_sum(weights, axis=(0, 1, 2), keepdims=True)

    """
    weights = compute_image_weights(y_true)[..., tf.newaxis]
    weights_sum = tf.reduce_sum(weights, axis=(1, 2), keepdims=True)
    """

    # cross entropy
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred)
    cross_entropy = tf.reduce_sum(weights * cross_entropy, axis=(0, 1, 2)) / weights_sum
    #cross_entropy = tf.reduce_sum(weights * cross_entropy, axis=(1, 2)) / weights_sum

    return cross_entropy


def rgb_to_oklab(images):
    """
    Convert image to oklab colorspace
    https://bottosson.github.io/posts/oklab/
    """
    images = tf.convert_to_tensor(images, dtype=tf.float32)

    l = 0.4122214708 * images[..., 0] + 0.5363325363 * images[..., 1] + 0.0514459929 * images[..., 2]
    m = 0.2119034982 * images[..., 0] + 0.6806995451 * images[..., 1] + 0.1073969566 * images[..., 2]
    s = 0.0883024619 * images[..., 0] + 0.2817188376 * images[..., 1] + 0.6299787005 * images[..., 2]

    l = tf.pow(l, 1 / 3)
    m = tf.pow(m, 1 / 3)
    s = tf.pow(s, 1 / 3)

    return tf.stack([
            0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
        ],
        axis=-1,
    )


def rai_initializer(kernel_width, kernel_height, input_features, output_features):
    """Randomized asymmetric initializer"""
    receptive_field = kernel_height * kernel_width
    fan_in = input_features * receptive_field
    fan_out = output_features * receptive_field

    V = np.random.randn(fan_out, fan_in + 1) * 0.6007 / fan_in ** 0.5
    for j in range(fan_out):
        k = np.random.randint(0, high=fan_in + 1)
        V[j, k] = np.random.beta(2, 1)
    W = V[:, :-1].T
    b = V[:, -1]

    return W.astype(np.float32), b.astype(np.float32)


class RAIConv2D(layers.Conv2D):

    def build(self, input_shape):
        kernel_weights, bias_weights = rai_initializer(*self.kernel_size, input_shape[-1], self.filters)
        self.kernel_initializer = tf.constant_initializer(kernel_weights)
        self.bias_initializer = tf.constant_initializer(bias_weights)
        super().build(input_shape)


class RAIDepthwiseConv2D(layers.DepthwiseConv2D):

    def build(self, input_shape):
        kernel_weights, bias_weights = rai_initializer(*self.kernel_size, input_shape[-1], input_shape[-1])
        self.kernel_initializer = tf.constant_initializer(kernel_weights)
        self.bias_initializer = tf.constant_initializer(bias_weights)
        #print(kernel_weights.shape, bias_weights.shape)
        super().build(input_shape)


def conv_block(x, output_filters):
    expansion = 2
    inputs = x

    # expansion
    x = RAIConv2D(expansion * inputs.shape[3], 1, padding='valid', use_bias=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # depthwise
    x = RAIDepthwiseConv2D(3, 1, padding='same', use_bias=False)(x)
    x = layers.ReLU()(x)

    # contraction
    x = RAIConv2D(output_filters, 1, padding='valid', use_bias=False)(x)

    # results
    if inputs.shape[3] == output_filters:
        return layers.add([inputs, x])
    else:
        return x


def double_conv_block(x, n_filters):
    x = conv_block(x, n_filters)
    x = conv_block(x, n_filters)

    return x


def downsample_block(x, n_filters):
    # covolutions
    f = double_conv_block(x, n_filters)

    # downsample
    p = layers.MaxPool2D(2)(f)

    return f, p


def upsample_block(x, conv_features, n_filters):
   # upsample
   x = layers.Conv2DTranspose(n_filters, 3, 2, padding='same')(x)

   # concatenate
   x = layers.concatenate([x, conv_features])

   # convolutions
   x = double_conv_block(x, n_filters)

   return x


def unet(mode):
    img_size = (256, 256)

    # input
    inputs = layers.Input(shape=img_size + (3,), dtype=tf.float32)
    inputs = rgb_to_oklab(inputs)

    # downsample
    f1, p1 = downsample_block(inputs, 32)
    f2, p2 = downsample_block(p1, 32)
    f3, p3 = downsample_block(p2, 64)
    f4, p4 = downsample_block(p3, 64)

    #  bottleneck
    b = double_conv_block(p4, 64)
    b = double_conv_block(b, 64)
    b = double_conv_block(b, 64)
    b = double_conv_block(b, 64)

    # upsample
    u4 = upsample_block(b, f4, 64)
    u3 = upsample_block(u4, f3, 64)
    u2 = upsample_block(u3, f2, 32)
    features = upsample_block(u2, f1, 32)

    # outputs and model
    if mode == 'unsupervised':
        outputs = RAIConv2D(3, 1, padding='same')(features)
        model = tf.keras.Model(inputs, outputs, name='unet')
        model.compile(optimizer=optimizers.Adam(), loss=unsupervised_loss)
    elif mode == 'weak':
        outputs = RAIConv2D(1, 1, padding='same')(features)
        model = tf.keras.Model(inputs, outputs, name='unet')
        model.compile(optimizer=optimizers.Adam(), loss=weak_loss, metrics=dice_loss)
    elif mode == 'supervised':
        outputs = RAIConv2D(1, 1, padding='same')(features)
        model = tf.keras.Model(inputs, outputs, name='unet')
        model.compile(optimizer=optimizers.Adam(), loss=supervised_loss, metrics=dice_loss)
    else:
        assert False, f'unknown mode {mode}'

    return model


def create_blobs(width, height, fill=0.5):
    """
    Create blob mask
    """
    # create random noise image
    noise = np.random.randint(0, 255, (height, width), np.uint8)

    # blur the noise image to control the size
    blur = cv2.GaussianBlur(noise, (0,0), sigmaX=4, sigmaY=4, borderType=cv2.BORDER_DEFAULT)
    blur = blur ** 2

    # stretch image
    stretch = blur - np.min(blur)
    stretch = stretch / np.max(stretch) * 255
    stretch = stretch.astype(np.uint8)

    # threshold stretched image to control the size
    thresh = cv2.threshold(stretch, 50 * (1 - fill), 255, cv2.THRESH_BINARY)[1]

    # apply morphology open and close to smooth out shapes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    result = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)

    return result / 255


class RandAug:
    """
    Random augmentations
    """

    def __init__(self, count=3, transforms=None):
        # create count function
        if isinstance(count, int):
            self.count = lambda: count
        else:
            self.count = count

        # compile transformation list
        self.transforms = {
            'identity': [1, RandAug.augment_identity, None],
            #'invert': [1, RandAug.augment_invert, None],
            'contrast': [0.8, RandAug.augment_contrast, None],
            'brightness': [0.3, RandAug.augment_brightness, None],
            'sharpness': [0.9, RandAug.augment_sharpness, None],
            #'affine': [1, RandAug.augment_affine, None],
            'blobs': [1, RandAug.augment_blobs, None],
            'flip': [1, RandAug.augment_flip, None],
            'rotate': [1, RandAug.augment_rotate, None],
            'noise': [0.5, RandAug.augment_noise, None],
        }

        if transforms is not None:
            for name, magnitude in transforms.items():
                assert name in self.transforms, f'{name} is unknown transformation'
                assert 0 <= magnitude <= 1, f'{name} has invalid magnitude {magnitude}'
                self.transforms[name][0] = magnitude

        self.generated = None


    @staticmethod
    def augment_identity(mode, image, magnitude, param):
        """
        Augment image with blobs
        """
        return None, image


    @staticmethod
    def augment_invert(mode, image, magnitude, param):
        """
        Augment image with blobs
        """
        if mode == 'mask':
            return None, image
        else:
            return None, np.array(ImageOps.invert(Image.fromarray(image)))


    @staticmethod
    def augment_contrast(mode, image, magnitude, contrast):
        """
        Augment image with blobs
        """
        if mode == 'mask':
            return None, image
        else:
            if contrast is None:
                contrast = 1 + 2 * magnitude * random.random()
                if random.random() < 0.5:
                    contrast = 1 / contrast

            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)

            v_mean = np.mean(v)
            v = (v - v_mean) * contrast + v_mean
            v = np.maximum(0, np.minimum(255, v))
            v = v.astype(np.uint8)

            hsv = cv2.merge((h, s, v))
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

            return contrast, image


    @staticmethod
    def augment_brightness(mode, image, magnitude, brightness):
        """
        Augment image with blobs
        """
        if mode == 'mask':
            return None, image
        else:
            if brightness is None:
                brightness = magnitude * (2 * random.random() - 1)

            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)

            v = v.astype(np.int16)
            v += int(255 * brightness)
            v = np.maximum(0, np.minimum(255, v))
            v = v.astype(np.uint8)

            hsv = cv2.merge((h, s, v))
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

            return brightness, image


    @staticmethod
    def augment_sharpness(mode, image, magnitude, sharpness):
        """
        Augment image with blobs
        """
        if mode == 'mask':
            return None, image
        elif mode == 'preserve_features':
            return None, image
        else:
            if sharpness is None:
                sharpness = 1 + magnitude * (2 * random.random() - 1)

            return sharpness, np.array(ImageEnhance.Sharpness(Image.fromarray(image)).enhance(sharpness))


    @staticmethod
    def augment_affine(mode, image, magnitude, matrix):
        """
        Augment image with blobs
        """
        return matrix, image


    @staticmethod
    def augment_blobs(mode, image, magnitude, param):
        """
        Augment image with blobs
        """
        if mode == 'mask':
            return None, image
        elif mode == 'preserve_features':
            return None, image
        else:
            blob_mask = create_blobs(*image.shape[:2], fill=0.75)[..., np.newaxis]
            noise = np.random.randint(0, 255, (*image.shape[:2], 1))
            composed = blob_mask * noise + (1 - blob_mask) * image

            return None, composed.astype(np.uint8)


    @staticmethod
    def augment_flip(mode, image, magnitude, flip):
        """
        Flip image
        """
        if flip is None:
            flip = random.random() < 0.5

        if flip:
            image = np.fliplr(image)

        return flip, image


    @staticmethod
    def augment_rotate(mode, image, magnitude, angle):
        """
        Rotate image
        """
        assert image.shape[0] == image.shape[1]

        if angle is None:
            angle = random.choice([0, 1, 2, 3])

        if angle:
            image = np.rot90(image, angle)

        return angle, image


    @staticmethod
    def augment_noise(mode, image, magnitude, std):
        """
        Augment image with blobs
        """
        if mode == 'mask':
            return None, image
        elif mode == 'preserve_features':
            return None, image
        else:
            if std is None:
                std = magnitude * 64 * random.random()

            noise = np.random.normal(0, std, image.shape)
            image = np.clip(image + noise, 0, 255)
            image = image.astype(np.uint8)

            return std, image


    def augment(self, mode, image):
        """
        Generate augmented image
        """
        # generate transforms
        if self.generated is None:
            generated = [name for name, transform in self.transforms.items() if transform[0] > 0]
            random.shuffle(generated)
            self.generated = generated[:self.count()]

        for name in self.generated:
            transform = self.transforms[name]
            magnitude, fn, param = transform
            param, image = fn(mode, image, magnitude, param)
            transform[2] = param

        return image


def iterate_unsupervised(dataset_path, batch_size, shuffle=True, augment=True):
    """
    Itearate through the whole unsupervised dataset
    """
    # read dataset
    image_paths = {}
    for filename in os.listdir(dataset_path):
        name, type = filename.split('_', 1)
        image_paths.setdefault(name, {})[type] = os.path.join(dataset_path, filename)
    samples = list(image_paths.keys())

    transforms = {
        'blobs': 0,
    }

    batch_inputs = []
    batch_outputs = []
    while True:
        random.shuffle(samples)
        for name in samples:
            image_file = image_paths[name]['image']

            # read image
            image = cv2.imread(image_file, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            width, height = image.shape[:2]

            # augment
            if augment:
                randaug = RandAug(2, transforms)
                preserved_image = randaug.augment('preserve_features', image)
                augmented_image = randaug.augment('image', image)
            else:
                preserved_image = image
                augmented_image = image

            blob_mask = create_blobs(*image.shape[:2], fill=0.75)[..., np.newaxis]
            noise = np.random.randint(0, 255, image.shape)
            masked_image = blob_mask * noise + (1 - blob_mask) * augmented_image
            masked_image = masked_image.astype(np.uint8)

            # add to batch
            batch_inputs.append(masked_image)
            batch_outputs.append(preserved_image / 255.0)

            # yield batch
            if len(batch_inputs) == batch_size:
                yield np.asarray(batch_inputs), np.asarray(batch_outputs)
                batch_inputs = []
                batch_outputs = []


def iterate_supervised(dataset_path, batch_size, shuffle=True, augment=True):
    """
    Itearate through the whole supervised dataset
    """
    # read dataset
    image_paths = {}
    for filename in os.listdir(dataset_path):
        name, type = filename.split('_', 1)
        image_paths.setdefault(name, {})[type] = os.path.join(dataset_path, filename)
    samples = list(image_paths.keys())

    batch_inputs = []
    batch_outputs = []
    while True:
        random.shuffle(samples)
        for name in samples:
            image_file = image_paths[name]['image']
            if 'map' not in image_paths[name]:
                continue
            map_file = image_paths[name]['map']

            # read images
            image = cv2.imread(image_file, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            map_mask = cv2.imread(map_file, cv2.IMREAD_GRAYSCALE)
            width, height = image.shape[:2]

            # augment
            if augment:
                randaug = RandAug(2)
                augmented_image = randaug.augment('image', image)
                map_mask = randaug.augment('mask', map_mask)
                map_mask = (map_mask > 128).astype(np.uint8) * 255
            else:
                augmented_image = image

            # add to batch
            batch_inputs.append(augmented_image)
            batch_outputs.append(map_mask[..., np.newaxis] / 255)

            # yield batch
            if len(batch_inputs) == batch_size:
                yield np.asarray(batch_inputs), np.asarray(batch_outputs)
                batch_inputs = []
                batch_outputs = []
