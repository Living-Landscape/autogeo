#!/usr/bin/env python3

import sys
import os
import argparse
import json
import random
import subprocess
import threading

import numpy as np
import cv2
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

if 'web' not in sys.path:
    sys.path.append('web')

import utils
from model_simple import SimpleDetector
from model_nnet import NNetDetector


def iterate_images(objects_path, images_path, include_path=None):
    """
    Iterate through downloaded images
    """
    objects = [int(object_id) for object_id in os.listdir(objects_path)]

    if include_path is None:
        included = None
    else:
        included = set(int(name.split('_')[1]) for name in os.listdir(include_path))

    for object_id in objects:
        # read object info
        object_path = os.path.join(objects_path, f'{object_id}')
        print(f'reading object {object_path}')
        with open(object_path) as fp:
            object_info = json.load(fp)
        assert object_info['object_id'] == object_id

        # iterate through all images
        for image_info in object_info['images']:
            image_id = image_info['id']
            if included is not None and image_id not in included:
                continue

            image_path = os.path.join(images_path, f'{object_id}_{image_id}')
            yield {
                'image_path': image_path,
                'object_path': object_path,
                'object': object_info,
                'object_id': object_id,
                'image_id': image_id,
            }


def annotate_maps(objects_path, images_path, annotated_path, include_path=None):
    """
    Create annotations of map masks
    """
    annotated = {name for name in os.listdir(annotated_path)}

    for info in iterate_images(objects_path, images_path, include_path):
        object_id = info['object_id']
        image_id = info['image_id']
        image_path = info['image_path']

        output_name = f'{object_id}_{image_id}_map'
        if output_name in annotated:
            print(' * already done, skipping')
            continue

        # read image and extract segments
        print(f' * reading image {image_path}')
        try:
            image = utils.read_image(image_path)
        except cv2.error:
            print(' * failed to load the image, skipping')
            continue
        print(f' * extracting segments from image {image.shape}')
        detector = SimpleDetector(image)
        detector.detect()
        print(f' * extracted {len(detector.segments)} segmenents')

        # save images, compute mask
        print(' * writing results')
        image_mask = np.zeros_like(detector.image_mask)

        # combine masks of all segments
        for n, segment in enumerate(detector.segments):
            image_mask = np.maximum(image_mask, Image.fromarray(detector.draw_segment(segment, shrink=False)))

        # save mask
        with open(os.path.join(annotated_path, output_name), 'wb') as fp:
            image = Image.fromarray(image_mask)
            image.save(fp, format='PNG')


def annotate_nnet(objects_path, images_path, annotated_path, model_path, include_path=None):
    """
    Create annotations of masks with neural network
    """
    model_path = os.path.join(os.getcwd(), model_path)

    annotated = {name for name in os.listdir(annotated_path)}

    for info in iterate_images(objects_path, images_path, include_path):
        object_id = info['object_id']
        image_id = info['image_id']
        image_path = info['image_path']

        output_names = [
            f'{object_id}_{image_id}_map',
            f'{object_id}_{image_id}_water',
            f'{object_id}_{image_id}_wmeadow',
        ]
        if output_names[0] in annotated:
            print(' * already done, skipping')
            continue

        # read image and extract segments
        print(f' * reading image {image_path}')
        try:
            image = Image.open(image_path)
            image_width = image.width
            image_height = image.height
        except FileNotFoundError:
            print(' * failed to load the image, skipping')
            continue
        print(f' * detecting maps from image {image_width} x {image_height}')
        os.chdir('web')
        try:
            image = np.array(image.resize((image_width // 4, image_height // 4)))
            detector = NNetDetector(model_path, image)
            segments, masks, confidence = detector.detect()
        finally:
            os.chdir('..')
        print(f' * detected {len(segments[0])} segments')

        # save mask
        print(' * writing results')
        for n in range(len(output_names)):
            with open(os.path.join(annotated_path, output_names[n]), 'wb') as fp:
                image = Image.fromarray(masks[..., n] * 255)
                image = image.resize((image_width, image_height))
                image.save(fp, format='PNG')


def rgb_to_oklab(colors):
    """
    Convert image to oklab colorspace
    https://bottosson.github.io/posts/oklab/
    """
    colors = np.asarray(colors, dtype=np.float32)

    const_shape = np.ones(len((1,)), dtype=int)
    const_shape[-1] = 3
    coef = np.ones(const_shape, dtype=np.float32)

    lcoef = coef * np.asarray([0.4122214708, 0.5363325363, 0.0514459929], dtype=np.float32)
    mcoef = coef * np.asarray([0.2119034982, 0.6806995451, 0.1073969566], dtype=np.float32)
    scoef = coef * np.asarray([0.0883024619, 0.2817188376, 0.6299787005], dtype=np.float32)

    l = lcoef * colors
    l = np.sum(l, axis=-1)
    m = mcoef * colors
    m = np.sum(m, axis=-1)
    s = scoef * colors
    s = np.sum(s, axis=-1)

    l = l ** (1 / 3)
    m = m ** (1 / 3)
    s = s ** (1 / 3)

    ll = 0.2104542553 * l
    ll += 0.7936177850 * m
    ll -= 0.0040720468 * s
    aa = 1.9779984951 * l
    aa -= 2.4285922050 * m
    aa += 0.4505937099 * s
    bb = 0.0259040371 * l
    bb += 0.7827717662 * m
    bb -= 0.8086757660 * s

    return np.stack([ll, aa,  bb], axis=-1)


def compute_color_ranges(allowed_colors, forbidden_colors, show_pallete=False):
    """
    Compute color ranges
    """
    allowed_colors_oklab = rgb_to_oklab(allowed_colors)
    forbidden_colors_oklab = rgb_to_oklab(forbidden_colors)

    # show colors
    width = max(len(allowed_colors), len(forbidden_colors))
    pallete = np.zeros((2, width, 3), np.uint8)
    for n, color in enumerate(allowed_colors):
        pallete[0, n] = color
    for n, color in enumerate(forbidden_colors):
        pallete[1, n] = color

    if show_pallete:
        plt.figure()
        plt.title('original pallete')
        plt.imshow(pallete)
        plt.show()

    # compute distances
    distances = []
    for allowed, allowed_oklab in zip(allowed_colors, allowed_colors_oklab):
        distance = min(np.sum((allowed_oklab - forbidden_oklab) ** 2) ** 0.5 for forbidden_oklab in forbidden_colors_oklab)
        distances.append(0.5 * distance)

    # prune colors
    pruned = set()
    for n, (color, color_oklab, distance) in enumerate(zip(allowed_colors, allowed_colors_oklab, distances)):
        if tuple(color) in pruned:
            continue
        for candidate, candidate_oklab in zip(allowed_colors[n + 1:], allowed_colors_oklab[n + 1:]):
            if distance >= np.sum((color_oklab - candidate_oklab) ** 2) ** 0.5:
                pruned.add(tuple(candidate))

    colors = [
        (color, color_oklab, distance)
        for color, color_oklab, distance in zip(allowed_colors, allowed_colors_oklab, distances)
        if tuple(color) not in pruned
    ]

    # show resulting colors
    width = len(colors)
    pallete = np.zeros((1, width, 3), np.uint8)
    print('colors')
    for n, (color, color_oklab, distance) in enumerate(colors):
        print(' *', color, distance)
        pallete[0, n] = color

    if show_pallete:
        plt.figure()
        plt.title('optimized pallete')
        plt.imshow(pallete)
        plt.show()

    return colors


def remove_thin_structures(mask, size):
    """
    Remove thin structures
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def find_edges(image):
    """
    Find edges on image
    """
    # convert to oklab space
    image = rgb_to_oklab(image)
    edges = image[..., 0]
    edges = edges - np.min(edges)
    edges = edges / np.max(edges)
    edges = edges * 255
    edges = edges.astype(np.uint8)
    edges = np.minimum(cv2.Canny(edges, 50, 90), 1)

    # find edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    return edges


def annotate_water(objects_path, images_path, annotated_path, include_path=None):
    """
    Annotate water
    """
    # collect admisable colors
    allowed_colors = [
        [130, 167, 167],
        [80, 160, 168],
        [130, 166, 176],
        [135, 178, 169],
        [187, 214, 205],
        [220, 233, 224],
        [168, 201, 203],
        [209, 223, 207],
    ]
    forbidden_colors = [
        [239, 231, 208],
        [188, 175, 157],
        [181, 210, 167],
        [244, 219, 188],
        [167, 151, 127],
        [150, 195, 152],
        [222, 231, 195],
        [228, 228, 204],
        [223, 226, 200],
        [220, 225, 202],
    ]
    colors = compute_color_ranges(allowed_colors, forbidden_colors, show_pallete=False)
    print()

    # see what's been already done
    annotated = {name for name in os.listdir(annotated_path)}

    # iterate through images
    for info in iterate_images(objects_path, images_path, include_path):
        object_id = info['object_id']
        image_id = info['image_id']
        image_path = info['image_path']

        output_name = f'{object_id}_{image_id}_water'
        if output_name in annotated:
            print(' * already done, skipping')
            continue

        # read image and extract segments
        print(f' * reading image {image_path}')
        try:
            image = utils.read_image(image_path)
        except cv2.error:
            print(' * failed to load the image, skipping')
            continue

        # convert to oklab space
        print(' * masking')
        image = image[..., :3]
        shape = image.shape
        image = cv2.resize(image, (shape[1] // 2, shape[0] // 2))
        image = rgb_to_oklab(image)

        # mask out water
        mask = np.zeros(image.shape[:2], np.uint8)
        for color, color_oklab, distance in colors:
            diff = image - color_oklab
            diff = diff ** 2
            diff = np.sum(diff, axis=-1)
            diff = diff < distance ** 2
            mask[diff] = 255
            del diff
        del image
        mask = cv2.resize(mask, (shape[1], shape[0]))
        mask = mask > 127
        mask = mask.astype(np.uint8)
        mask = mask * 255

        # close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # save mask
        output_path = os.path.join(annotated_path, output_name)
        print(f' * saving to {output_path}')
        with open(output_path, 'wb') as fp:
            mask = Image.fromarray(mask)
            mask.save(fp, format='png')


def annotate_manually(images_path, preannotated_path, annotated_path, image_path, show, paint):
    """
    Annotate manually
    """
    assert show in ('mask', 'inverted_mask', 'edges', 'nothing')

    # gather annotations
    images = set(os.listdir(images_path))
    preannotations = set(os.listdir(preannotated_path))
    annotated = set(os.listdir(annotated_path))
    to_annotate = {name.rsplit('_', 1)[0]: name for name in preannotations - annotated}
    common = images & set(to_annotate.keys())
    print('already annotated', len(annotated))
    print('available annotations', len(common))
    print()

    if image_path is None:
        # pick random  image
        picked = random.choice(list(common))
    else:
        if image_path not in images:
            print(f'{image_path} is not in available images')
            return
        if image_path not in common and show not in ('edges', 'nothing'):
            print(f'{image_path} is not in available preannotations')
            return
        picked = image_path
    image_path = os.path.join(images_path, picked)
    if show in ('edges', 'nothing'):
        mask_path = None
        output_path = os.path.join(annotated_path, f'{picked}_generic')
        tmp_path = os.path.join('/tmp', f'{picked}_generic.png')
    else:
        mask_path = os.path.join(preannotated_path, to_annotate[picked])
        output_path = os.path.join(annotated_path, to_annotate[picked])
        tmp_path = os.path.join('/tmp', f'{to_annotate[picked]}.png')
    print('image', image_path)
    print('mask', mask_path)
    print('annotated', output_path)
    print('tmp', tmp_path)
    print()

    try:
        # open original image
        thread = threading.Thread(target=lambda: subprocess.run(['eog', image_path]), daemon=True)
        thread.start()

        print('masking original image')
        image = Image.open(image_path)
        if show == 'edges':
            #mask = np.array(Image.open(mask_path))
            mask = find_edges(image)
            mask = 1 - mask
            mask = 255 * mask
            mask = mask[..., np.newaxis]
            image = np.concatenate([image, mask], axis=2)
            """
            image = mask // 2
            image = image - 1
            image = np.maximum(mask, edges)
            """
        elif show in ('mask', 'inverted_mask'):
            mask = np.array(Image.open(mask_path))
            mask = mask[..., np.newaxis]
            if show == 'inverted_mask':
                mask = 255 - mask
            mask = np.maximum(mask, 90)
            image = np.array(image)
            image = np.concatenate([image, mask], axis=2)
        elif show == 'nothing':
            mask = 255 * np.ones((image.height, image.width, 1), np.uint8)
            image = np.array(image)
            image = np.concatenate([image, mask], axis=2)
        image = Image.fromarray(image)
        image.save(tmp_path, format='png')

        print('manual annotion')
        subprocess.run([paint, tmp_path])

        print('extracting mask')
        image = Image.open(tmp_path)
        assert image.mode == 'RGBA', image.mode
        if show == 'edges':
            _, _, image, _ = image.split()
            image = Image.fromarray(np.array(image) > 200)
        elif show == 'mask':
            _, _, _, image = image.split()
            image = Image.fromarray(np.array(image) > 127)
        elif show == 'inverted_mask':
            _, _, _, image = image.split()
            image = Image.fromarray(np.array(image) <= 127)
        elif show == 'nothing':
            _, _, _, image = image.split()
            image = Image.fromarray(np.array(image) <= 127)
        image.save(tmp_path, format='png')

        print('post-processing')
        image = Image.open(tmp_path)
        if show == 'edges':
            image = ImageOps.grayscale(image)
            image = np.asarray(image)
            image = 255 - image
            image = remove_thin_structures(image, 5)
            image = 255 - image
        elif show in ('mask', 'inverted_mask', 'nothing'):
            image = Image.open(tmp_path)
            image = ImageOps.grayscale(image)
            image = remove_thin_structures(np.asarray(image), 7)
        image = Image.fromarray(image)
        image.save(tmp_path, format='png')

        print('manual correction')
        subprocess.run(['pinta', tmp_path])

        print('converting to grayscale')
        image = Image.open(tmp_path)
        image = ImageOps.grayscale(image)
        image = Image.fromarray(np.asarray(image))
        image.save(output_path, format='png')
    finally:
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass  # ignore


def check_annotations(images_path, annotated_path, show_pair):
    """
    Check image annotatations
    """
    name = os.path.basename(annotated_path)
    image_name = name.rsplit('_', 1)[0]
    image_path = os.path.join(images_path, image_name)
    tmp_path = os.path.join('/tmp', f'{name}_check.png')

    print('image', image_path)
    print('annotated', annotated_path)
    print('tmp', tmp_path)

    try:
        print('masking original image')
        image = np.array(Image.open(image_path))
        mask = np.array(Image.open(annotated_path))
        assert image.shape[:2] == mask.shape[:2]

        shape = image.shape
        image = cv2.resize(image, (shape[1] // 2, shape[0] // 2))
        mask = cv2.resize(mask, (shape[1] // 2, shape[0] // 2))
        mask = mask[..., np.newaxis]
        mask = mask / 255

        color = (128, 0, 255)
        masked_original = (0.5 * mask * image).astype(np.uint8)
        masked_color = (0.5 * mask * [[color]]).astype(np.uint8)
        non_masked = ((1 - mask) * image).astype(np.uint8)
        image = masked_color + masked_original + non_masked

        if show_pair:
            mask = [[(255, 255, 255)]] * mask
            mask = mask.astype(np.uint8)
            image = np.hstack([image, mask])

        image = Image.fromarray(image)
        image.save(tmp_path, format='png')

        print('check annotations')
        subprocess.run(['eog', tmp_path])
    finally:
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass  # ignore


def check_annotations_dir(images_path, annotated_path, mask_type, excluded_path, random_order, show_pair):
    """
    Check image annotatations in the whole directory
    """
    # create exclusion list based on dir
    if excluded_path is None:
        excluded_names = set()
    else:
        excluded_names = {name for name in os.listdir(excluded_path)}
    print(f'excluding {len(excluded_names)} from {excluded_path}')

    # create exclusion list based on done list
    done_path = f'{annotated_path.strip("/")}.checked'
    try:
        with open(done_path) as fp:
            for line in fp:
                excluded_names.add(line.strip())
    except FileNotFoundError:
        pass  # nothing was checked already
    print(f'excluding {len(excluded_names)} from altogether')


    # iterate through the directory
    names = os.listdir(annotated_path)
    if random_order:
        random.shuffle(names)
    for mask_name in names:
        if not mask_name.endswith(f'_{mask_type}'):
            continue
        if mask_name in excluded_names:
            continue

        mask_path = os.path.join(annotated_path, mask_name)
        check_annotations(images_path, mask_path, show_pair)

        print(done_path)
        with open(done_path, 'a') as fp:
            fp.write(f'{mask_name}\n')


def main():
    """
    Anntations helper
    """
    # parse arguments
    argparser = argparse.ArgumentParser()
    subparsers = argparser.add_subparsers(dest='mode', required=True)

    parser_maps = subparsers.add_parser('maps', help='annotate maps')
    parser_maps.add_argument('objects_path', help='path with objects to annotate')
    parser_maps.add_argument('images_path', help='path with images to annotate')
    parser_maps.add_argument('annotated_path', help='path with annotations')
    parser_maps.add_argument('--include', default=None, help='use only included images')

    parser_water = subparsers.add_parser('water', help='annotate water')
    parser_water.add_argument('objects_path', help='path with objects to annotate')
    parser_water.add_argument('images_path', help='path with images to annotate')
    parser_water.add_argument('annotated_path', help='path with annotations')
    parser_water.add_argument('--include', default=None, help='use only inlcuded images')

    parser_nnet = subparsers.add_parser('nnet', help='annotate with nerual network')
    parser_nnet.add_argument('objects_path', help='path with objects to annotate')
    parser_nnet.add_argument('images_path', help='path with images to annotate')
    parser_nnet.add_argument('annotated_path', help='path with annotations')
    parser_nnet.add_argument('--include', default=None, help='use only included images')
    parser_nnet.add_argument('--model', default='nnet.tflite', help='path to nnet model')

    parser_manual = subparsers.add_parser('manual', help='manual annotation')
    parser_manual.add_argument('images_path', help='path with images to annotate')
    parser_manual.add_argument('preannotated_path', help='path with preannotated masks')
    parser_manual.add_argument('annotated_path', help='path with annotations')
    parser_manual.add_argument('--image', default=None, help='custom path to image')
    parser_manual.add_argument('--show', choices=['mask', 'inverted_mask', 'edges', 'nothing'], default='mask', help='how to show annotations')
    parser_manual.add_argument('--paint', default='pinta', help='application for annotations')

    parser_check = subparsers.add_parser('check', help='check annotated images')
    parser_check.add_argument('images_path', help='path with images')
    parser_check.add_argument('annotated_path', help='path with annotated mask')
    parser_check.add_argument('--show-pair', action='store_true', help='show image and mask')

    parser_check_dir = subparsers.add_parser('check-dir', help='check annotated images in directory')
    parser_check_dir.add_argument('images_path', help='path with images')
    parser_check_dir.add_argument('annotated_path', help='path with annotations')
    parser_check_dir.add_argument('--mask', choices=['map', 'water', 'wmeadow', 'dmeadow'], default='map', help='show chosen mask type')
    parser_check_dir.add_argument('--exclude', default=None, help='exclude images in given dir')
    parser_check_dir.add_argument('--random', action='store_true', help='show annotations in random order')
    parser_check_dir.add_argument('--show-pair', action='store_true', help='show image and mask')

    args = argparser.parse_args()

    if args.mode == 'maps':
        annotate_maps(args.objects_path, args.images_path, args.annotated_path, args.include)
    elif args.mode == 'water':
        annotate_water(args.objects_path, args.images_path, args.annotated_path, args.include)
    elif args.mode == 'nnet':
        annotate_nnet(args.objects_path, args.images_path, args.annotated_path, args.model, args.include)
    elif args.mode == 'manual':
        annotate_manually(args.images_path, args.preannotated_path, args.annotated_path, args.image, args.show, args.paint)
    elif args.mode == 'check':
        check_annotations(args.images_path, args.annotated_path, args.show_pair)
    elif args.mode == 'check-dir':
        check_annotations_dir(args.images_path, args.annotated_path, args.mask, args.exclude, args.random, args.show_pair)
    else:
        assert False


# entrypoint
if __name__ == '__main__':
    main()
