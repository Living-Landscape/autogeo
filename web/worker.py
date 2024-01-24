# author: Jan Prochazka
# license: public domain

import os
import io
import zipfile
import gc
import logging
import subprocess
import time

from redis import Redis
from rq.job import Job
from PIL import Image
import numpy as np
import cv2

from model_simple import SimpleDetector
from model_nnet import NNetDetector


# create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s')


class NoMapsFound(RuntimeError):
    """
    No maps found in the picture
    """
    pass


def save_image(image, name, output_format, zip_file):
    """
    Save image according to format
    """
    if output_format == 'png':
        with io.BytesIO() as image_bytes:
            image = Image.fromarray(image)
            image.save(image_bytes, format='png')

            # optimize output image - quantize + compress
            try:
                process = subprocess.Popen(['pngquant', '-'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
                compressed_bytes = process.communicate(input=image_bytes.getvalue())[0]
                process = subprocess.Popen(['oxipng', '--strip', 'safe', '-'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
                compressed_bytes = process.communicate(input=compressed_bytes)[0]
                if len(compressed_bytes) == 0:
                    compressed_bytes = image_bytes.getvalue()
            except Exception:
                compressed_bytes = image_bytes.getvalue()

        zip_file.writestr(f'{name}.png', compressed_bytes)
    elif output_format == 'webp':
        with io.BytesIO() as image_bytes:
            image = Image.fromarray(image)
            image.save(image_bytes, format='webp')
            zip_file.writestr(f'{name}.webp', image_bytes.getvalue())
    elif output_format == 'jpg':
        background = np.asarray([[(255, 255, 255)]], dtype=np.uint8)
        image = np.where(image[..., 3:] == 255, image[..., :3], background)
        image = Image.fromarray(image)
        #image = numpy_to_pil(image[..., :3], 'RGB')
        with io.BytesIO() as image_bytes:
            image.save(image_bytes, format='jpeg', quality=90, optimize=True)
            zip_file.writestr(f'{name}.jpg', image_bytes.getvalue())


def process(job_id, detector_type, output_format):
    """
    Extract map parts
    """
    assert detector_type in ('simple', 'nnet'), f'Unsupported detector type {detector_type}'
    assert output_format in ('png', 'webp', 'jpg'), f'Unsupported output format {output_format}'

    try:
        job_path = os.path.join('jobs', job_id)
        result_path = os.path.join('results', job_id)

        job = Job.fetch(job_id, connection=Redis())

        # decode image
        try:
            image = Image.open(job_path)
            image_width = image.width
            image_height = image.height
        except Exception:
            raise RuntimeError('Nedokážu přečíst nahrávaný soubor, zpracuju jenom .png a .jpg obrázky.')

        # check image mode
        if image.mode == 'L':
            raise RuntimeError('Čekal jsem barevný obrázek, ne šedivý')
        elif image.mode == 'RGB':
            pass
        elif image.mode == 'RGBA':
            pass
        else:
            raise RuntimeError('Obrázek má neznámý formát, umím pracovat jenom s rgb a rgba kanály v .png a .jpg obrázcích.')

        # resize
        dimension = max(image.width, image.height)
        if dimension > 4000:
            resize_ratio = 4
        elif dimension > 2000:
            resize_ratio = 2
        else:
            resize_ratio = None
        if resize_ratio is not None:
            image = image.resize((image.width // resize_ratio, image.height // resize_ratio), Image.BICUBIC)

        # add transparency if necessary
        if image.mode == 'RGB':
            image.putalpha(255)

        # convert to numpy
        image = np.array(image, dtype=np.uint8)
        gc.collect()

        def progress_fn(progress):
            """Keep an eye on computation progress."""
            nonlocal last_progress_update

            update_delay = 1  # minimum delay between updates in s

            now = time.time()
            if last_progress_update is None or now - last_progress_update > update_delay:
                job.meta['progress'] = progress * 0.95  # adjustment for zipping files
                job.save()
                last_progress_update = now

        # extract map blobs
        map_index = 0
        water_index = 1
        wetmeadow_index = 2
        names = ['mapa', 'voda', 'mokre_louky']

        if detector_type == 'simple':
            detector = SimpleDetector(image)
            segments, masks = detector.detect()
        elif detector_type == 'nnet':
            detector = NNetDetector('model_nnet.tflite', image)
            last_progress_update = None
            segments, masks, confidence = detector.detect(progress_fn)
            masks = 255 * masks
        if not segments[map_index]:  # map mask
            raise NoMapsFound('Nepodařilo se najít žádné mapy na obrázku.')

        # zip with images
        with zipfile.ZipFile(result_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for n, map_segment in enumerate(segments[map_index]):
                map_box, map_contour = map_segment
                map_left, map_top, map_right, map_bottom = map_box
                map_width = map_right - map_left + 1
                map_height = map_bottom - map_top + 1

                # create image for detected map segment
                map_image = detector.draw_segment(map_segment)
                if resize_ratio is not None:
                    map_image = cv2.resize(map_image, (resize_ratio * map_width, resize_ratio * map_height), interpolation=cv2.INTER_CUBIC)

                # write map image
                save_image(map_image, f'{names[map_index]}_{n + 1}', output_format, zip_file)

                del map_image
                gc.collect()

                # find all water chunks inside current map segment
                for mask_index in [water_index, wetmeadow_index]:
                    # draw mask image
                    mask_image = None
                    for p, mask_segment in enumerate(segments[mask_index]):
                        (left, top, right, bottom), contour = mask_segment
                        if (
                            (map_left <= left < map_right or map_left <= right < map_right) and
                            (map_top <= top < map_bottom or map_top <= bottom < map_bottom)
                        ):
                            mask_image = detector.draw_segment((map_box, contour), erase=mask_image is None)

                    # write mask image
                    if mask_image is not None:
                        mask_image[..., 3] = np.minimum(  # intersection of original mask and computed polygon
                            masks[map_top:map_bottom + 1, map_left:map_right + 1, mask_index],
                            mask_image[..., 3],
                        )
                        if resize_ratio:
                            mask_image = cv2.resize(mask_image, (resize_ratio * map_width, resize_ratio * map_height), interpolation=cv2.INTER_CUBIC)
                        save_image(mask_image, f'{names[mask_index]}_{n + 1}', output_format, zip_file)

                    del mask_image
                    gc.collect()

            # background colors
            if output_format == 'jpg':
                background_colors = '\n'.join(
                    f'{r} {g} {b} 100'
                    for r in range(250, 256)
                    for g in range(250, 256)
                    for b in range(250, 256)
                )
                zip_file.writestr('pozadi.txt', background_colors)

        del detector
        del image
        gc.collect()

        job.meta['download'] = True
        job.save()

        return True
    except Exception as exc:
        # write exception information to job meta
        message = str(exc)
        try:
            job = Job.fetch(job_id, connection=Redis())
            if isinstance(exc, NoMapsFound):
                job.meta['info'] = message
            else:
                job.meta['error'] = message
                if isinstance(exc, RuntimeError):
                    logger.error('%s processing error %s', job_id, message)
                else:
                    logger.exception('%s processing exception %s', job_id, message)
            job.save()
        except Exception:
            logger.exception('%s processing exception %s', job_id, message)

        return False
    finally:
        # remove image from temporary storage
        try:
            os.unlink(job_path)
        except Exception:
            pass
