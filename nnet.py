
import os
import random
import json

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from PIL import Image, ImageEnhance, ImageOps


def lr_schedule(max_lr, min_lr, batch, epoch_batch_count, training_batch_count):
    """
    Log-quadratic learning rate scheduler with warmup
    """
    if batch < epoch_batch_count:
        step = batch / epoch_batch_count
        decay = step * (max_lr - min_lr) + min_lr
    else:
        step = (batch - epoch_batch_count) / (training_batch_count - epoch_batch_count)
        b = np.log(min_lr)
        a = np.log(max_lr) - b
        decay = np.exp(a * (1 - step ** 2) + b)

    return decay


def prewitt_kernel():
    """
    Return Prewitt kernel
    """
    prewitt = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1],
    ], dtype=np.float32)
    prewitt = prewitt[..., None, None]

    return prewitt


def gauss_kernel(size, sigma):
    """
    Return gaussian kernel
    """
    x = np.linspace(-0.5 * (size - 1), 0.5 * (size - 1), size)
    kernel = np.exp((-0.5 * x ** 2) / sigma ** 2)
    kernel = kernel.astype(np.float32)
    kernel = kernel[:, None, None, None]

    return kernel / np.sum(kernel)


def mask_edges(masks):
    """
    Return mask of widen edges
    """
    shape = tf.shape(masks)

    # build prewitt kernel
    prewitt = prewitt_kernel()
    prewitt = tf.convert_to_tensor(prewitt, masks.dtype)
    prewitt = prewitt * tf.ones((3, 3, shape[-1], 1), tf.float32)

    # find edges
    gx = tf.nn.depthwise_conv2d(masks, prewitt, (1, 1, 1, 1), 'SAME')
    gy = tf.nn.depthwise_conv2d(masks, tf.transpose(prewitt, (1, 0, 2, 3)), (1, 1, 1, 1), 'SAME')
    gradient = (gx ** 2 + gy ** 2) ** 0.5
    gradient_max = tf.reduce_max(gradient, axis=(1, 2), keepdims=True)
    gradient = tf.where(gradient_max > 0, gradient / gradient_max, 0)

    # draw edges around the border
    ones = 1 + 0 * gradient[:, :1, ...]
    gradient = tf.concat([ones, gradient[:, 1:-1, ...], ones], axis=1)
    ones = 1 + 0 * gradient[..., :1, :]
    gradient = tf.concat([ones, gradient[..., 1:-1, :], ones], axis=2)

    # blurring kernel
    kernel_size = 17
    sigma = 5
    kernel = gauss_kernel(kernel_size, sigma)
    kernel = tf.convert_to_tensor(kernel, masks.dtype)
    kernel = kernel * tf.ones((kernel_size, 1, shape[-1], 1), tf.float32)

    # blur
    gradient = tf.nn.depthwise_conv2d(gradient, kernel, (1, 1, 1, 1), 'SAME')
    gradient = tf.nn.depthwise_conv2d(gradient, tf.transpose(kernel, (1, 0, 2, 3)), (1, 1, 1, 1), 'SAME')

    return tf.cast(gradient > 0.03, tf.float32)


def boosted_edges_loss(targets, predictions):
    """
    Computes loss with boosted mask edges
    """
    # edges
    edge_weight = tf.random.uniform((1, 1, 1, predictions.shape[-1]), 1, 5)
    edges = mask_edges(targets)

    # weights
    weights = 1 + (edge_weight - 1) * edges
    weights = weights / tf.reduce_sum(weights, axis=(1, 2), keepdims=True)

    # l2 loss
    loss = tf.where(targets == 1, tf.maximum(0.0,  1 - predictions) ** 2, tf.maximum(0.0, predictions) ** 2)
    loss = tf.reduce_sum(weights * loss, axis=(1, 2))

    return loss


class BlendedLoss(tf.keras.layers.Layer):
    """
    Loss layer, computes reconstruction loss for blended images
    """

    def build(self, input_shapes):
        """
        Create variables
        """
        pass


    def call(self, in_blended, in_a, in_b, p_reconstruction):
        """
        Compute losses
        """
        # dominant losss
        l1_a = tf.reduce_mean(tf.abs(in_a / 255 - p_reconstruction), axis=(1, 2))
        l1_a = tf.reduce_mean(l1_a, axis=1)

        # losses
        self.add_loss(l1_a)

        return p_reconstruction


class SegmentationLoss(tf.keras.layers.Layer):
    """
    Loss layer, l2 segmentation loss
    """

    def __init__(self, logvars_init=None):
        super().__init__()
        self.logvars_init = logvars_init


    def build(self, input_shapes):
        """
        Create variables
        """
        self.logvars = self.add_weight(shape=(1, input_shapes[-1]), initializer=tf.constant_initializer(value=self.logvars_init), name='logvars', trainable=True)


    def call(self, in_masks, in_targets, p_targets):
        """
        Compute losses
        """
        mask_types = self.logvars.shape[-1]

        # losses
        raw_losses = boosted_edges_loss(in_targets, p_targets)
        losses = tf.math.exp(-self.logvars) * raw_losses + self.logvars
        losses = tf.reduce_mean(in_masks * losses, axis=1)

        # losses
        self.add_loss(losses)

        # metrics
        noise = (tf.math.exp(self.logvars) ** 0.5)
        for n in range(mask_types):
            self.add_metric(noise[0, n], name=f'n{n}')
        for n in range(mask_types):
            self.add_metric(1 - raw_losses[0, n], name=f'j{n}')

        return p_targets


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


def msf_block(inputs, filters, scale='same'):
    """
    Mutli-scale fusion step
    """
    assert len(inputs) == len(filters)
    assert scale in ('down', 'up', 'same')

    scales = len(inputs)
    x = scales * [None]
    for n in range(scales):
        # scale
        if scale == 'down':
            size = inputs[n].shape[1]
            x[n] = tf.image.resize(inputs[n], (size // 2, size // 2))
        elif scale == 'up':
            size = inputs[n].shape[1]
            x[n] = tf.image.resize(inputs[n], (size * 2, size * 2))
        else:
            x[n] = inputs[n]

        # convolution
        x[n] = layers.Conv2D(filters[n], 1, padding='valid')(x[n])
        x[n] = layers.BatchNormalization()(x[n])
        x[n] = layers.LeakyReLU()(x[n])
        x[n] = layers.DepthwiseConv2D(3, 1, padding='same')(x[n])
        x[n] = layers.LeakyReLU()(x[n])

    # merge
    merged = scales * [None]
    for n in range(scales):
        size = x[n].shape[1]

        merged[n] = x[n]
        filter_count = x[n].shape[-1]
        if n > 0:
            # upsample
            if filter_count == x[n - 1].shape[-1]:
                merged[n] += tf.image.resize(x[n - 1], (size, size))
            elif filter_count > x[n - 1].shape[-1]:
                merged[n] = tf.concat([
                    merged[n][..., :x[n - 1].shape[-1]] + tf.image.resize(x[n - 1], (size, size)),
                    merged[n][..., x[n - 1].shape[-1]:],
                ], axis=-1)
            else:
                merged[n] += tf.image.resize(x[n - 1][..., :filter_count], (size, size))
        if n < scales - 1:
            # downsample
            if filter_count == x[n + 1].shape[-1]:
                merged[n] += tf.image.resize(x[n + 1], (size, size))
            elif filter_count > x[n + 1].shape[-1]:
                merged[n] = tf.concat([
                    merged[n][..., :x[n + 1].shape[-1]] + tf.image.resize(x[n + 1], (size, size)),
                    merged[n][..., x[n + 1].shape[-1]:],
                ], axis=-1)
            else:
                merged[n] += tf.image.resize(x[n + 1][..., :filter_count], (size, size))

    for n in range(scales):
        # projection
        x[n] = layers.Conv2D(filters[n], 1, padding='valid')(merged[n])
        x[n] = layers.BatchNormalization()(x[n])
        x[n] = layers.LeakyReLU()(x[n])

        # convoluton
        x[n] = layers.DepthwiseConv2D(3, 1, padding='same')(x[n])
        x[n] = layers.LeakyReLU()(x[n])
        x[n] = layers.Conv2D(filters[n], 1, padding='valid')(x[n])

        # output
        if inputs[n].shape == x[n].shape:
            x[n] = layers.add([inputs[n], x[n]])

    return x


def fast_model_parameters():
    """
    Return parameters for fast model
    """
    return {
        'img_size': (512, 512),
        'filters': [24, 32, 64, 64, 64, 64, 128, 256],
        'depth': 6,  # model trunk layers count
        'resolution': 3,  # scaling factor of model trunk (2 ** resolution)
    }


def strong_model_parameters():
    """
    Return parameters for strong model
    """
    return {
        'img_size': (512, 512),
        'filters': [24, 32, 64, 64, 64, 64, 128, 256],
        'depth': 12,  # model trunk layers count
        'resolution': 2,  # scaling factor of model trunk (2 ** resolution)
    }


class EncoderNN:
    """
    Encoder for NN
    """

    def __init__(self, model=None, parameters=None):
        """
        Initialize net
        """
        self.img_size = parameters['img_size']
        self.filters = parameters['filters']
        self.depth = parameters['depth']
        self.resolution = parameters['resolution']
        self.scales = len(self.filters) - self.resolution

        if model is None:
            self.model = self.build()
        else:
            self.model = model


    def build(self):
        """
        Build encoder
        """
        scales = self.scales
        filters = self.filters

        # inputs
        in_img = layers.Input(shape=self.img_size + (3,), dtype=tf.float32, name='image')

        # intro
        #oklab = rgb_to_oklab(in_img)
        oklab = in_img

        # downscale features
        x = scales * [None]
        size = oklab.shape[1]
        x[0] = oklab
        for n in range(1, scales):
            size = oklab.shape[1] // (2 ** n)
            x[n] = tf.image.resize(oklab, (size, size))

        # downscale
        for scale in range(self.resolution):
            x = msf_block(x, filters[scale:scales + scale], 'down')

        # body
        for _ in range(self.depth):
            x = msf_block(x, filters[self.resolution:scales + self.resolution], 'same')
        x = msf_block(x, filters[self.resolution - 1:scales + self.resolution - 1], 'same')

        # upscale
        for scale in range(self.resolution - 2, -1, -1):
            x = msf_block(x, filters[scale:scales + scale], 'up')

        # encoder
        self.model = tf.keras.Model(
            name='encoder',
            inputs=[in_img],
            outputs=[x],
        )

        return self.model


class UnsupervisedNN:
    """
    Unsupervised NN, reconstruction
    """
    def __init__(self, path=None, parameters=None):
        """
        Initialize net
        """
        self.model = None
        self.training_model = None

        # create / load model
        if path is None:
            self.build_training(EncoderNN(parameters=parameters))
        else:
            self.load(path)


    def build_model(self, encoder, in_img):
        """
        Build detector
        """
        # add detection head
        scales = encoder.scales
        filters = encoder.filters
        embedding =  encoder.model(in_img)[0]
        reconstruction = msf_block(embedding, [3] + filters[0:scales - 1], 'up')[0]

        return tf.keras.Model(
            name='unsupervised_model',
            inputs=[in_img],
            outputs=[reconstruction],
        )


    def build_training(self, encoder):
        """
        Build training models
        """
        # inputs
        in_blended = layers.Input(shape=encoder.img_size + (3,), dtype=tf.float32, name='combined_image')
        in_a = layers.Input(shape=encoder.img_size + (3,), dtype=tf.float32, name='a_image')
        in_b = layers.Input(shape=encoder.img_size + (3,), dtype=tf.float32, name='b_image')

        # models
        self.model = self.build_model(encoder, in_blended)

        # outputs
        p_reconstruction = self.model(in_blended)

        # losses
        p_unsupervised = BlendedLoss()(in_blended, in_a, in_b, p_reconstruction)

        # model for training
        self.training_model = tf.keras.Model(
            name='training_model',
            inputs=[in_blended, in_a, in_b],
            outputs=[p_unsupervised],
        )

        # unsupervised optimizer
        self.training_model.compile(optimizer=tf.keras.optimizers.Adam(0.001))


    def predict(self, inputs):
        """
        Predict outputs
        """
        return self.model.predict_on_batch(inputs)


    def eval(self, inputs, outputs):
        """
        Evaluate inputs
        """
        return self.training_model.test_on_batch(inputs, outputs)


    def save(self, path):
        """
        Save model to file
        """
        self.training_model.save(path, save_format='tf')


    def load(self, path):
        """
        Load model from saved file
        """
        self.training_model = tf.keras.models.load_model(
            path,
            custom_objects={'BlendedLoss': BlendedLoss},
        )
        self.model = self.training_model.get_layer('unsupervised_model')


class DetectorNN:
    """
    Detector NN, semantic segmentation
    """
    def __init__(self, load=None, parameters=None, logvars_init=None):
        """
        Initialize net
        """
        self.target_count = 4  # map, water, wet meadow, dry meadow

        self.model = None
        self.training_model = None
        self.logvars_init = logvars_init

        # create / load model
        if load is None:
            self.build_training(EncoderNN(parameters=parameters))
        else:
            model_type, path = load
            if model_type == 'detector':
                self.load(path)
            elif model_type == 'unsupervised':
                self.load_unsupervised(path)
            else:
                assert False


    def build_model(self, encoder, in_img):
        """
        Build detector
        """
        # add detection head
        scales = encoder.scales
        filters = encoder.filters
        embedding =  encoder.model(in_img)[0]
        predictions = msf_block(embedding, [self.target_count] + filters[0:scales - 1], 'up')[0]

        return tf.keras.Model(
            name='detector',
            inputs=[in_img],
            outputs=[predictions],
        )


    def build_training(self, encoder):
        """
        Create training models
        """
        assert self.logvars_init is not None

        # inputs
        in_img = layers.Input(shape=encoder.img_size + (3,), dtype=tf.float32, name='image')
        in_targets = layers.Input(shape=encoder.img_size + (self.target_count,), dtype=tf.float32, name='targets')
        in_masks = layers.Input(shape=(self.target_count,), dtype=tf.float32, name='masks')

        # models
        self.model = self.build_model(encoder, in_img)

        # outputs
        p_targets = self.model(in_img)

        # losses
        p_targets = SegmentationLoss(self.logvars_init)(in_masks, in_targets, p_targets)

        # model for training
        self.training_model = tf.keras.Model(
            name='training_model',
            inputs=[in_img, in_masks, in_targets],
            outputs=[p_targets],
        )

        # optimizer
        self.training_model.compile(optimizer=tf.keras.optimizers.Adam(0.001), jit_compile=True)


    def predict(self, inputs):
        """
        Predict outputs
        """
        return self.model.predict_on_batch(inputs)


    def eval(self, inputs, outputs):
        """
        Evaluate inputs
        """
        return self.training_model.test_on_batch(inputs, outputs)


    def save(self, path):
        """
        Save model to file
        """
        self.training_model.save(path, save_format='tf')


    def load(self, path):
        """
        Load model from saved file
        """
        self.training_model = tf.keras.models.load_model(
            path,
            custom_objects={
                'SegmentationLoss': SegmentationLoss,
            },
        )
        self.training_model.compile(jit_compile=True)
        self.model = self.training_model.get_layer('detector')


    def load_unsupervised(self, path):
        """
        Load from unsupervised model file
        """
        training_model = tf.keras.models.load_model(
            path,
            custom_objects={'BlendedLoss': BlendedLoss},
        )
        unsupervised_model = training_model.get_layer('unsupervised_model')
        encoder = EncoderNN(unsupervised_model.get_layer('encoder'))
        self.build_training(encoder)


def create_blobs(width, height, fill=0.5, seed=None):
    """
    Create blob mask
    """
    generator = np.random.default_rng(seed=seed)

    # create random noise image
    noise = generator.integers(0, 255, (height, width), np.uint8)

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
            #'blobs': [1, RandAug.augment_blobs, None],
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


def dataset_info(dataset_path, types):
    """
    Return list of files in dataset
    """
    assert types is not None

    if isinstance(types, str):
        types = {types}
    else:
        types = set(types)

    # read dataset info
    with open(os.path.join(dataset_path, 'info.json')) as fp:
        info = json.load(fp)

    # filter dataset
    filtered_samples = []
    filtered_counts = {}
    for samples in info['samples']:
        filtered_types = set(samples.keys()) & types
        if not filtered_types:
            continue
        filtered_samples.append({types: os.path.join(dataset_path, samples[types][0]) for types in filtered_types})
        for current_type in filtered_types:
            filtered_counts[current_type] = filtered_counts.get(current_type, 0) + 1

    return {
        'samples': filtered_samples,
        'counts': filtered_counts,
    }


def blend_images(batch_images):
    """
    Blend images
    """
    batch_a = []
    batch_b = []
    batch_blended = []
    bs = len(batch_images)
    for n in range(bs // 2):
        image_a = batch_images[n]
        image_b = batch_images[bs // 2 + n]

        # a dominant
        alpha = 0.7 + np.random.random() * 0.1
        image_blended = alpha * image_a + (1 - alpha) * image_b
        image_blended = image_blended.astype(np.uint8)
        batch_a.append(image_a)
        batch_b.append(image_b)
        batch_blended.append(image_blended)

        # b dominant
        alpha = 0.7 + np.random.random() * 0.1
        image_blended = alpha * image_b + (1 - alpha) * image_a
        image_blended = image_blended.astype(np.uint8)
        batch_a.append(image_b)
        batch_b.append(image_a)
        batch_blended.append(image_blended)

    return batch_blended, batch_a, batch_b


def iterate_unsupervised(dataset_path, batch_size, shuffle=True, augment=True):
    """
    Itearate through the whole unsupervised dataset
    """
    # read dataset
    info = dataset_info(dataset_path, {'image'})
    samples = info['samples']

    batch_images = []
    while True:
        if shuffle:
            random.shuffle(samples)
        for paths in samples:
            image_file = paths['image']

            # read image
            image = Image.open(image_file)
            assert image.mode == 'RGB'
            image = np.asarray(image)
            width, height = image.shape[:2]

            # augment
            if augment:
                randaug = RandAug(2)
                image = randaug.augment('image', image)

            batch_images.append(image)

            # yield batch
            if len(batch_images) ==  batch_size:
                batch_blended, batch_a, batch_b = blend_images(batch_images)
                zeros = np.zeros((batch_size,))
                yield [
                    np.asarray(batch_blended),
                    np.asarray(batch_a),
                    np.asarray(batch_b),
                ], [zeros]
                batch_images = []

        # yield the restbatch
        if batch_images:
            batch_blended, batch_a, batch_b = blend_images(batch_images)
            zeros = np.zeros((batch_size,))
            yield [
                np.asarray(batch_blended),
                np.asarray(batch_a),
                np.asarray(batch_b),
            ], [zeros]
            batch_images = []


def iterate_supervised(dataset_path, batch_size, masks, shuffle=True, augment=True):
    """
    Itearate through the whole supervised dataset
    """
    assert masks is not None

    # read dataset info
    info = dataset_info(dataset_path, set(masks) | {'image'})
    samples = info['samples']

    # iterate through dataset
    batch_images = []
    batch_masks = []
    batch_targets = []
    while True:
        if shuffle:
            random.shuffle(samples)
        for paths in samples:
            image_file = paths['image']

            # read images
            image = Image.open(image_file)
            assert image.mode == 'RGB'
            image = np.asarray(image)
            width, height = image.shape[:2]

            # augment image
            if augment:
                randaug = RandAug(2)
                image = randaug.augment('image', image)

            # read masks
            masks = []
            mask_targets = []
            contains_map = True
            for mask in ['map', 'water', 'wetmeadow', 'drymeadow']:  # map must be first (contais_map)
                mask_file = paths.get(mask)
                if mask_file is None:
                    if contains_map:
                        masks.append(0)
                    else:
                        masks.append(1)  # no map == no other regions
                    mask_target = np.zeros((width, height), np.float32)
                else:
                    masks.append(1)
                    mask_target = Image.open(mask_file)
                    assert mask_target.mode == 'L'
                    mask_target = np.asarray(mask_target)
                assert mask_target.shape == (width, height)

                # augment
                if augment:
                    if mask_file is not None:
                        mask_target = randaug.augment('mask', mask_target)
                        mask_target = mask_target.astype(np.float32) / 255
                else:
                    if mask_file is not None:
                        mask_target = mask_target.astype(np.float32) / 255

                # if there is no map region, other masks are not present as well
                if mask == 'map' and mask_file is not None and np.sum(mask_target) == 0:
                    contains_map = False

                mask_targets.append(mask_target[..., None])

            # add to batch
            batch_images.append(image.astype(np.float32))
            batch_masks.append(np.asarray(masks, dtype=np.float32))
            batch_targets.append(np.concatenate(mask_targets, axis=-1))

            # yield batch
            if len(batch_images) == batch_size:
                yield {
                    'image': np.asarray(batch_images),
                    'masks': np.asarray(batch_masks),
                    'targets': np.asarray(batch_targets),
                }
                batch_images = []
                batch_masks = []
                batch_targets = []

        # yield the rest batch
        if batch_images:
            yield {
                'image': np.asarray(batch_images),
                'masks': np.asarray(batch_masks),
                'targets': np.asarray(batch_targets),
            }
            batch_images = []
            batch_masks = []
            batch_targets = []
