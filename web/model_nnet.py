# author: Jan Prochazka
# license: public domain

import numpy as np
import cv2
import tflite_runtime.interpreter as tflite
from scipy import ndimage

from model import Detector


class TFLiteModel:

    def __init__(self, path):
        self.thresholds = (0.60, 0.64, 0.58)

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
        predictions = self.interpreter.get_tensor(self.p_out['index'])

        return predictions[0]


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


    def detect(self, progress_callback=None, return_confidence_masks=False):
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
            if return_confidence_masks:
                return segments, masks, np.ones((3,)), np.zeros((*image.shape[:2], 3), np.uint8)
            else:
                return segments, masks, np.ones((3,))

        # run nnet inference on chunks, averaging results
        masks = np.zeros((*image.shape[:2], 3), np.float32)
        counts = np.zeros((*image.shape[:2], 1), np.uint8)
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
                chunk_outputs = self.model.predict(chunk)

                masks[top:bottom, left:right] += chunk_outputs
                counts[top:bottom, left:right] += 1
                iterations += 1
        masks = masks / counts

        # compute confidences
        confidence_margin = 0.3
        confidences = 0.5 - masks
        confidences = np.abs(confidences)
        confidences = confidences <= 0.5 * confidence_margin
        confidence_values = 1 - (np.sum(confidences, axis=(0, 1)) / (image.shape[0] * image.shape[1])) ** 0.5
        confidence_values = np.maximum(0, confidence_values - 0.5) * 2
        if return_confidence_masks:
            confidences = confidences.astype(np.uint8)
        else:
            confidences = None

        # convert masks
        masks = masks > self.model.thresholds
        masks = masks.astype(np.uint8)

        min_map_ratio = 20
        min_water_ratio = 200
        min_wetmeadow_ratio = 200

        # maps post-processing
        masks[..., 0] = ndimage.binary_fill_holes(masks[..., 0])
        masks[..., 0] = masks[..., 0].astype(np.uint8)
        masks[..., 0] = remove_thin_structures(masks[..., 0])

        # find map segments
        segments = [
            list(self.find_segments(masks[..., n], min_area_ratio))
            for n, min_area_ratio in enumerate([min_map_ratio, min_water_ratio, min_wetmeadow_ratio])
        ]

        if return_confidence_masks:
            return segments, masks, confidence_values, confidences
        else:
            return segments, masks, confidence_values
