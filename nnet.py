
import os
import random
import json

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from PIL import Image, ImageEnhance, ImageOps


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


def weak_loss(y_true, y_pred):
    """
    Combined dice and cross entropy loss
    """
    # TODO: water and grass not yet implemented
    y_true = y_true[..., :1]
    y_pred = y_pred[..., :1]
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
    kernel_size = 17
    shape = tf.shape(masks)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    filters = tf.convert_to_tensor(kernel, masks.dtype)
    filters = tf.expand_dims(filters, axis=-1) * tf.ones((kernel_size, kernel_size, shape[-1]), tf.float32)

    # invert borders to create an edge
    masks = tf.concat([1 - masks[:, :1, ...], masks[:, 1:-1, ...], 1 - masks[:, -1:, ...]], axis=1)
    masks = tf.concat([1 - masks[..., :1, :], masks[..., 1:-1, :], 1 - masks[..., -1:, :]], axis=2)

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


def boosted_edges_loss(detections, predictions):
    """
    Computes loss with boosted mask edges
    """
    # edges
    edge_weight = tf.random.uniform((1, 1, 1, 3), 1, 40)
    edges = mask_edges(detections)

    # weights
    weights = 1 + (edge_weight - 1) * edges
    weights = weights / tf.reduce_sum(weights, axis=(1, 2), keepdims=True)

    """
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(weights[0, ..., 0])
    plt.colorbar()
    plt.show()
    """

    # loss
    loss = tf.where(detections == 1, tf.maximum(0.0,  1 - predictions) ** 2, tf.maximum(0.0, predictions) ** 2)
    loss = tf.reduce_sum(weights * loss, axis=(1, 2))

    return loss


class Loss(tf.keras.layers.Layer):
    """
    Loss layer
    """

    def build(self, input_shapes):
        """
        Create variables
        """
        self.logvars = self.add_weight(shape=(1, input_shapes[-1]), initializer=tf.constant_initializer(value=-3), name='logvars', trainable=True)


    def call(self, in_masks, in_detections, p_detections):
        """
        Compute losses
        """
        raw_losses = boosted_edges_loss(in_detections, p_detections)
        losses = tf.math.exp(-self.logvars) * raw_losses + self.logvars
        losses = tf.reduce_mean(in_masks * losses, axis=1)

        # losses
        self.add_loss(losses)

        # metrics
        noise = (tf.math.exp(self.logvars) ** 0.5)[0]
        self.add_metric(noise[0], name='mv')
        self.add_metric(noise[1], name='wv')
        self.add_metric(noise[2], name='gv')

        return p_detections


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


class Net:
    """
    Neural network for semantic segmentation
    """

    def __init__(self, path=None):
        """
        Initialize net
        """
        self.img_size = (512, 512)
        self.detection_count = 3  # map, water, grass

        self.training_model = None
        self.production_model = None

        if path is None:
            self.build()
        else:
            self.load(path)


    def build_detector(self, in_img):
        """
        Build detector
        """
        scales = 5
        filters = [24, 32, 64, 64, 64, 64, 128, 256]

        # intro
        oklab = rgb_to_oklab(in_img)

        # downscale features
        x = scales * [None]
        size = oklab.shape[1]
        x[0] = oklab
        for n in range(1, scales):
            size = size // 2
            x[n] = tf.image.resize(oklab, (size, size))

        # downscale
        x = msf_block(x, filters[0:scales + 0], 'down')
        x = msf_block(x, filters[1:scales + 1], 'down')
        x = msf_block(x, filters[2:scales + 2], 'down')

        # body
        x = msf_block(x, filters[3:scales + 3], 'same')
        x = msf_block(x, filters[3:scales + 3], 'same')
        x = msf_block(x, filters[3:scales + 3], 'same')
        x = msf_block(x, filters[3:scales + 3], 'same')
        x = msf_block(x, filters[3:scales + 3], 'same')
        x = msf_block(x, filters[3:scales + 3], 'same')

        # upscale
        x = msf_block(x, filters[2:scales + 2], 'same')
        x = msf_block(x, filters[1:scales + 1], 'up')
        x = msf_block(x, filters[0:scales + 0], 'up')

        # outputs
        outputs = msf_block(x, [self.detection_count] + filters[0:scales - 1], 'up')[0]

        return outputs


    def build(self):
        """
        Create models
        """
        # inputs
        in_img = layers.Input(shape=self.img_size + (3,), dtype=tf.float32, name='img')
        in_detections = layers.Input(shape=self.img_size + (self.detection_count,), dtype=tf.float32, name='detections')
        in_masks = layers.Input(shape=(self.detection_count,), dtype=tf.float32, name='masks')

        # detector
        p_detections = self.build_detector(in_img)

        # model for production
        self.production_model = tf.keras.Model(
            name='detector',
            inputs=[in_img],
            outputs=[p_detections],
        )

        # losses
        p_detections = self.production_model(in_img)
        p_detections = Loss()(in_masks, in_detections, p_detections)

        # model for training
        self.training_model = tf.keras.Model(
            name='training_model',
            inputs=[in_img, in_masks, in_detections],
            outputs=[p_detections],
        )

        # optimizer
        self.training_model.compile(optimizer=tf.keras.optimizers.Adam(0.001))


    def predict(self, inputs):
        """
        Predict outputs
        """
        return self.production_model.predict_on_batch(inputs)


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
            custom_objects={'Loss': Loss},
        )
        self.production_model = self.training_model.get_layer('detector')


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
            image = Image.open(image_file)
            assert image.mode == 'RGB'
            image = np.asarray(image)
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


def dataset_info(dataset_path, masks=None):
    """
    Return list of files in dataset
    """
    if masks is None:
        masks = {'map', 'water', 'grass'}
    elif isinstance(masks, str):
        masks = {masks}
    else:
        masks = set(masks)
    masks.add('image')

    # read dataset info
    with open(os.path.join(dataset_path, 'info.json')) as fp:
        info = json.load(fp)

    # filter dataset
    filtered_samples = []
    filtered_positive = {}
    filtered_count = {}
    for samples in info['samples']:
        filtered_masks = set(samples.keys()) & masks
        if len(filtered_masks) <= 1:
            continue
        filtered_samples.append({mask: os.path.join(dataset_path, samples[mask][0]) for mask in filtered_masks})
        for mask in filtered_masks:
            filtered_positive[mask] = filtered_positive.get(mask, 0) + samples[mask][1]
            filtered_count[mask] = filtered_count.get(mask, 0) + samples['image'][1]

    return {
        'samples': filtered_samples,
        'positive': filtered_positive,
        'count': filtered_count,
    }


def iterate_supervised(dataset_path, batch_size, shuffle=True, augment=True, masks=None):
    """
    Itearate through the whole supervised dataset
    """
    # read dataset info
    info = dataset_info(dataset_path, masks)
    samples = info['samples']

    # iterate through dataset
    batch_images = []
    batch_masks = []
    batch_detections = []
    batch_count = None
    while batch_count is None or batch_count > 0:
        batch_count = 0
        random.shuffle(samples)
        for paths in samples:
            image_file = paths['image']
            map_file = paths.get('map')
            water_file = paths.get('water')
            grass_file = paths.get('grass')

            # read images
            image = Image.open(image_file)
            assert image.mode == 'RGB'
            image = np.asarray(image)
            width, height = image.shape[:2]

            masks = []

            # read map
            if map_file is None:
                masks.append(0)
                map_detection = np.zeros((width, height), np.uint8)
            else:
                masks.append(1)
                map_detection = Image.open(map_file)
                assert map_detection.mode == 'L'
                map_detection = np.asarray(map_detection)

            # read water
            if water_file is None:
                masks.append(0)
                water_detection = np.zeros((width, height), np.uint8)
            else:
                masks.append(1)
                water_detection = Image.open(water_file)
                assert water_detection.mode == 'L'
                water_detection = np.asarray(water_detection)

            # read grass
            if grass_file is None:
                masks.append(0)
                grass_detection = np.zeros((width, height), np.uint8)
            else:
                masks.append(1)
                grass_detection = Image.open(grass_file)
                assert grass_detection.mode == 'L'
                grass_detection = np.asarray(grass_detection)

            assert map_detection.shape == (width, height)
            assert map_detection.shape == water_detection.shape
            assert map_detection.shape == grass_detection.shape

            # augment
            if augment:
                randaug = RandAug(2)
                image = randaug.augment('image', image)

                if map_file is not None:
                    map_detection = randaug.augment('mask', map_detection)
                    map_detection = (map_detection >= 128).astype(np.float32)

                if water_file is not None:
                    water_detection = randaug.augment('mask', water_detection)
                    water_detection = (water_detection >= 128).astype(np.float32)

                if grass_file is not None:
                    grass_detection = randaug.augment('mask', grass_detection)
                    grass_detection = (grass_detection >= 128).astype(np.float32)
            else:
                if map_file is not None:
                    map_detection = map_detection.astype(np.float32) / 255
                if water_file is not None:
                    water_detection = water_detection.astype(np.float32) / 255
                if grass_file is not None:
                    grass_detection = grass_detection.astype(np.float32) / 255

            # add to batch
            batch_images.append(image)
            batch_masks.append(masks)
            batch_detections.append(
                np.concatenate([
                    map_detection[..., np.newaxis],
                    water_detection[..., np.newaxis],
                    grass_detection[..., np.newaxis],
                ], axis=-1)
            )

            # yield batch
            if len(batch_images) == batch_size:
                zeros = np.zeros((batch_size,))
                yield [
                    np.asarray(batch_images),
                    np.asarray(batch_masks),
                    np.asarray(batch_detections),
                ], [zeros]
                batch_count += 1
                batch_images = []
                batch_masks = []
                batch_detections = []

        # yield the rest batch
        if batch_images:
            zeros = np.zeros((batch_size,))
            yield [
                np.asarray(batch_images),
                np.asarray(batch_masks),
                np.asarray(batch_detections),
            ], [zeros]
            batch_count += 1
            batch_images = []
            batch_masks = []
            batch_detections = []
