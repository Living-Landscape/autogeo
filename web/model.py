# author: Jan Prochazka
# license: public domain

import numpy as np
import cv2


class Detector:
    """
    Detect map parts
    """

    def __init__(self, image):
        self.image = image
        self.image_height, self.image_width = image.shape[:2]
        self.segments = None


    def find_segments(self, mask):
        """
        Find map contours
        """
        # find blob contours
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        # extract the biggest segments
        for contour in contours:
            # check area
            min_dimension = np.sqrt(self.image_width * self.image_height) / 20
            area = cv2.contourArea(contour)
            if area < min_dimension ** 2:
                continue

            left = np.min(contour[..., 0])
            right = np.max(contour[..., 0])
            top = np.min(contour[..., 1])
            bottom = np.max(contour[..., 1])

            yield (left, top, right, bottom), contour


    def draw_segment(self, segment, shrink=True, erase=True):
        """
        Draw segments into image
        """
        (left, top, right, bottom), contour = segment

        if erase:
            self.image[..., 3].fill(0)

        if shrink:
            segment_width = right - left + 1
            segment_height = bottom - top + 1
            mask = np.zeros((
                segment_height,
                segment_width,
            ), dtype=np.uint8)

            cv2.fillPoly(mask, [contour - [(left, top)]], color=255)
            self.image[top:bottom + 1, left:right + 1, 3] = mask

            return self.image[top:bottom + 1, left:right + 1, :]
        else:
            cv2.fillPoly(self.image[..., 3], [contour], color=255)

            return self.image


    def detect(self):
        """
        Detect map segments
        """
        raise NotImplementedError()
