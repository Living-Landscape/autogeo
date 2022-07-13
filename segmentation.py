# author: Jan Prochazka
# license: public domain

import gc

import cv2
import numpy as np
import numba


@numba.jit
def should_expand(x, y, image, width, height, mask, edges, min_tolerance, max_tolerance, seen):
    """
    Return if we should expand neighbors in floodfill
    """
    return (
        0 <= x < width and
        0 <= y < height and
        seen[y, x] == 0 and
        mask[y, x] > 0 and
        edges[y, x] == 0 and
        min_tolerance[0] < image[y, x, 0] < max_tolerance[0] and
        min_tolerance[1] < image[y, x, 1] < max_tolerance[1] and
        min_tolerance[2] < image[y, x, 2] < max_tolerance[2]
    )

@numba.jit(nopython=True)
def flood_fill_fast(image, mask, edges, starts, mean, std, scale):
    """
    Floodfill inside mask
    """
    height = image.shape[0]
    width = image.shape[1]
    min_tolerance = (
        mean[0] - scale * std[0],
        mean[1] - scale * std[1],
        mean[2] - scale * std[2],
    )
    max_tolerance = (
        mean[0] + scale * std[0],
        mean[1] + scale * std[1],
        mean[2] + scale * std[2],
    )
    seen = np.zeros_like(edges)
    stack = [coord for coord in starts if should_expand(*coord, image, width, height, mask, edges, min_tolerance, max_tolerance, seen)]
    while stack:
        x, y = stack.pop()

        seen[y, x] = 1
        mask[y, x] = 0
        if should_expand(x - 1, y, image, width, height, mask, edges, min_tolerance, max_tolerance, seen):
            stack.append((x - 1, y))
        if should_expand(x + 1, y, image, width, height, mask, edges, min_tolerance, max_tolerance, seen):
            stack.append((x + 1, y))
        if should_expand(x, y - 1, image, width, height, mask, edges, min_tolerance, max_tolerance, seen):
            stack.append((x, y - 1))
        if should_expand(x, y + 1, image, width, height, mask, edges, min_tolerance, max_tolerance, seen):
            stack.append((x, y + 1))

    return mask


def flood_fill(image, edges, starts, mean, std, scale):
    """
    Flood fill inside mask
    """
    mask = image[:, :, 3]
    starts = numba.typed.List(starts)
    return flood_fill_fast(image, mask, edges, starts, mean, std, scale)


@numba.jit(nopython=True)
def remove_background(image, mask, left, top, right, bottom, min_tolerance, max_tolerance):
    """
    Remove background
    """

    for x in range(right - left):
        for y in range(bottom - top):
            if (mask[y, x] > 0 and ((
                min_tolerance[0] < image[y + top, x + left, 0] < max_tolerance[0] and
                min_tolerance[1] < image[y + top, x + left, 1] < max_tolerance[1] and
                min_tolerance[2] < image[y + top, x + left, 2] < max_tolerance[2]
                ) or 0)):
                mask[y, x] = 0



class MapExtractor:
    """
    Extract map parts
    """

    def __init__(self, image):
        self.image = image
        self.image_mask = image[..., 3]
        self.image_height, self.image_width = image.shape[:2]
        self.segment_padding = 20


    def compute_borders(self):
        """
        Compute image borders
        """
        image_width = self.image_width
        image_height = self.image_height

        # get transparent border
        transparent_left = np.min(np.sum(self.image[:, :image_width // 2, 3] < 255, axis=1))
        transparent_right = np.min(np.sum(self.image[:, image_width // 2:, 3] < 255, axis=1))
        transparent_top = np.min(np.sum(self.image[:image_height // 2, :, 3] < 255, axis=0))
        transparent_bottom = np.min(np.sum(self.image[image_height // 2:, :, 3] < 255, axis=0))

        # get background color
        border_width = image_width // 100  + 1
        border_height = image_height // 100  + 1

        border_left = self.image[
            transparent_top:image_height - transparent_bottom,
            transparent_left:transparent_left + border_width,
            :3,
        ]
        border_right = self.image[
            transparent_top:image_height - transparent_bottom,
            image_width - transparent_right - border_width:image_width - transparent_right,
            :3,
        ]
        border_top = self.image[
            transparent_top:transparent_top + border_height,
            transparent_left:image_width - transparent_right,
            :3,
        ]
        border_bottom = self.image[
            image_height - transparent_bottom - border_height:image_height - transparent_bottom,
            transparent_left:image_width - transparent_right,
            :3,
        ]
        border_mean = (np.mean(border_left, axis=(0, 1)) + np.mean(border_right, axis=(0, 1)) + np.mean(border_top, axis=(0, 1)) + np.mean(border_bottom, axis=(0, 1))) / 4
        border_std = (np.std(border_left, axis=(0, 1)) + np.std(border_right, axis=(0, 1)) + np.std(border_top, axis=(0, 1)) + np.std(border_bottom, axis=(0, 1))) / 4
        border_std = np.maximum(border_std, 3)

        self.border_width = border_width
        self.border_height = border_height
        self.transparent_left = transparent_left
        self.transparent_right = transparent_right
        self.transparent_top = transparent_top
        self.transparent_bottom = transparent_bottom
        self.border_mean = border_mean
        self.border_std = border_std


    def strip_background(self):
        """
        Strip image background
        """
        std_scale = 3

        image_width = self.image_width
        image_height = self.image_height
        transparent_left = self.transparent_left
        transparent_right = self.transparent_right
        transparent_top = self.transparent_top
        transparent_bottom = self.transparent_bottom
        border_width = self.border_width
        border_height = self.border_height
        border_mean = self.border_mean
        border_std = self.border_std

        # compute border coordinates to start flood fill from
        border_coords = (
            [(transparent_left + border_width, y) for y in range(transparent_top + border_height, image_height - transparent_bottom - border_height)] +
            [(image_width - transparent_right - border_width, y) for y in range(transparent_top + border_height, image_height - transparent_bottom - border_height)] +
            [(x, transparent_top + border_height) for x in range(transparent_left + border_width, image_width - transparent_right - border_width)] +
            [(x, image_height - transparent_bottom - border_height) for x in range(transparent_left + border_width, image_width - transparent_right - border_width)]
        )

        # strengthen edges
        edges = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        edges = np.minimum(cv2.Canny(edges, 50, 90), 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # flood fill background
        flood_fill(self.image, edges, border_coords, border_mean, border_std, std_scale)


    def remove_thin_structures(self):
        """
        Remove thin structures
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.image[..., 3] = cv2.morphologyEx(self.image_mask, cv2.MORPH_OPEN, kernel)
        self.image[..., 3] = cv2.morphologyEx(self.image_mask, cv2.MORPH_CLOSE, kernel)


    def find_segments(self):
        """
        Find map contours
        """
        image = self.image

        # find blob contours
        contours = cv2.findContours(self.image_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        # extract the biggest segments
        self.segments = []
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

            # draw segment, reuse existing mask
            mask = np.zeros((bottom - top, right - left), dtype=np.uint8)
            cv2.fillPoly(mask, [contour], color=1, offset=(-left, -top))

            # check borders
            horizontal_border_bound = self.image_width // 100 + 1
            vertical_border_bound = self.image_height // 100 + 1

            if left <= self.transparent_left:
                left_pixels = np.sum(mask[:, 0])
                if left_pixels > horizontal_border_bound:
                    del mask
                    continue
            if right <= self.transparent_right:
                right_pixels = np.sum(mask[:, -1])
                if right_pixels > horizontal_border_bound:
                    del mask
                    continue
            if top <= self.transparent_top:
                top_pixels = np.sum(mask[0, :])
                if top_pixels > vertical_border_bound:
                    del mask
                    continue
            if bottom <= self.transparent_bottom:
                bottom_pixels = np.sum(mask[-1, :])
                if bottom_pixels > vertical_border_bound or bottom_pixels > vertical_border_bound:
                    del mask
                    continue

            # background segments
            scale = 3
            min_dimension = np.sqrt(self.image_width * self.image_height) // 30 + 1
            foreground_ratio = 0.35

            min_tolerance = (
                self.border_mean[0] - scale * self.border_std[0],
                self.border_mean[1] - scale * self.border_std[1],
                self.border_mean[2] - scale * self.border_std[2],
            )
            max_tolerance = (
                self.border_mean[0] + scale * self.border_std[0],
                self.border_mean[1] + scale * self.border_std[1],
                self.border_mean[2] + scale * self.border_std[2],
            )
            segment_area = np.sum(mask)
            remove_background(image, mask, left, top, right, bottom, min_tolerance, max_tolerance)
            foreground_area = np.sum(mask)
            if foreground_area < min_dimension ** 2 or foreground_area / segment_area < foreground_ratio:
                del mask
                continue

            # save segment
            self.segments.append((
                (left, top, right, bottom),
                contour,
            ))
            del mask


    def draw_segment(self, segment, shrink=False):
        """
        Draw segments into image
        """
        (left, top, right, bottom), contour = segment
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
            mask = np.zeros_like(self.image_mask)
            cv2.fillPoly(mask, [contour], color=255)

            return mask


    def extract(self):
        """
        Extract map segments
        """
        self.compute_borders()
        gc.collect()
        self.strip_background()
        gc.collect()
        self.remove_thin_structures()
        gc.collect()
        self.find_segments()
        gc.collect()
