# author: Jan Prochazka
# license: public domain

import numpy as np
import cv2
import tflite_runtime.interpreter as tflite
from scipy import ndimage

from model import Detector


def prediction_confidence(predictions, thresholds):
    """
    Compute prediction confidence
    """
    shape = predictions.shape
    topn = int(0.05 * shape[1] * shape[2])

    predictions = np.reshape(predictions, (shape[0], shape[1] * shape[2], shape[3]))
    errors = 1 - 2 * np.abs(thresholds - np.maximum(0, np.minimum(1, predictions)))

    return 1 - np.mean(np.sort(errors, axis=1)[:, -topn:, :], axis=1)


class TFLiteModel:

    def __init__(self, path):
        self.thresholds = (0.56, 0.41, 0.5)

        self.interpreter = tflite.Interpreter(model_path=path)
        self.interpreter.allocate_tensors()

        # input details
        inputs = {}
        for inp in self.interpreter.get_input_details():
            inputs[inp['name']] = inp

        # output details
        outputs = {}
        for out in self.interpreter.get_output_details():
            outputs[out['name']] = out

        self.in_img = inputs['serving_default_img:0']
        self.p_out = outputs['StatefulPartitionedCall:0']


    def predict(self, image):
        """
        Predict map parts
        """
        image = image.astype(np.float32)[np.newaxis, ...]
        self.interpreter.set_tensor(self.in_img['index'], image)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.p_out['index'])
        confidence = prediction_confidence(out, self.thresholds)

        return out[0], confidence[0]


def remove_thin_structures(mask):
    """     
    Remove thin structures
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


class NNetDetector(Detector):
    """
    Detect map parts
    """

    def __init__(self, model_path, image):
        self.image = image
        self.image_height, self.image_width = image.shape[:2]
        self.model = TFLiteModel(model_path)


    def detect(self, progress_callback=None):
        """
        Detect map segments
        Returns segments, masks
        """
        chunk_size = 512
        overlap = chunk_size // 2

        image = self.image

        # check for too small images
        if self.image_width < chunk_size or self.image_height < chunk_size:
            masks = np.zeros((*image.shape[:2], 3), np.uint8)
            segments = [[], [], []]
            return segments, masks, [1.0, 1.0, 1.0]

        # run nnet inference on chunks, averaging results
        masks = np.zeros((*image.shape[:2], 3), np.float32)
        counts = np.zeros((*image.shape[:2], 1), np.uint8)
        confidences = []
        iterations = 0
        for x in range(0, image.shape[1], chunk_size - overlap):
            for y in range(0, image.shape[0], chunk_size - overlap):
                if progress_callback is not None:
                    progress_current = y + x * image.shape[0]
                    progress_total = image.shape[0] * image.shape[1]
                    progress_callback(progress_current / progress_total)
                left = x
                right = min(x + chunk_size, image.shape[1])
                top = y
                bottom = min(y + chunk_size, image.shape[0])
                left = min(left, right - chunk_size)
                top = min(top, bottom - chunk_size)

                chunk = image[top:bottom, left:right, :3]
                outputs, chunk_confidence = self.model.predict(chunk)

                masks[top:bottom, left:right] += outputs
                counts[top:bottom, left:right] += 1
                confidences.append(chunk_confidence)
                iterations += 1
        masks = masks / counts
        masks = masks > self.model.thresholds
        masks = masks.astype(np.uint8)
        topn = int(iterations / 20)
        confidence = np.mean(np.sort(confidences, axis=0)[:topn], axis=0)

        min_map_ratio = 20
        min_water_ratio = 200
        min_grass_ratio = 1

        # post-processing maps
        masks[..., 0] = ndimage.binary_fill_holes(masks[..., 0])
        masks[..., 0] = masks[..., 0].astype(np.uint8)
        masks[..., 0] = remove_thin_structures(masks[..., 0])

        # find map segments
        segments = [
            list(self.find_segments(masks[..., n], min_area_ratio))
            for n, min_area_ratio in enumerate([min_map_ratio, min_water_ratio, min_grass_ratio])
        ]

        return segments, masks, confidence
