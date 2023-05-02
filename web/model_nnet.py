# author: Jan Prochazka
# license: public domain

import numpy as np
import cv2
import tflite_runtime.interpreter as tflite
from scipy import ndimage

from model import Detector


def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))


class TFLiteModel:

    def __init__(self, path):
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

        self.in_img = inputs['serving_default_input_1:0']
        self.p_out = outputs['StatefulPartitionedCall:0']

    def predict(self, image):
        """
        Predict map parts
        """
        image = image.astype(np.float32)[np.newaxis, ...]
        self.interpreter.set_tensor(self.in_img['index'], image)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.p_out['index'])

        return sigmoid(out[0])


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
        self.image_mask = None
        self.model = TFLiteModel(model_path)
        self.segments = None


    def detect(self, progress_callback=None):
        """
        Detect map segments
        """
        chunk_size = 512
        overlap = 256

        image = self.image

        # check for too small images
        if self.image_width < chunk_size or self.image_height < chunk_size:
            self.image_mask = np.zeros(image.shape[:2], np.uint8)
            self.segments = []
            return

        # run nnet inference on chunks, averaging results
        self.image_mask = np.zeros(image.shape[:2], np.float32)
        counts = np.zeros(image.shape[:2], np.uint8)
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
                outputs = self.model.predict(chunk)

                self.image_mask[top:bottom, left:right] += outputs[..., 0]
                counts[top:bottom, left:right] += 1
        self.image_mask = self.image_mask / counts
        self.image_mask = self.image_mask > 0.5
        self.image_mask = self.image_mask.astype(np.uint8)

        # post-processing
        self.image_mask = ndimage.binary_fill_holes(self.image_mask)
        self.image_mask = self.image_mask.astype(np.uint8)
        self.image_mask = remove_thin_structures(self.image_mask)

        # find map segments
        self.segments = list(self.find_segments())
