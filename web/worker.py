# author: Jan Prochazka
# license: public domain

import os
import io
import zipfile
import gc
import logging
import subprocess

from redis import Redis
from rq.job import Job
from PIL import Image
import numpy as np

import segmentation


# create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s')


class NoMapsFound(RuntimeError):
    """
    No maps found in the picture
    """
    pass


def process(job_id, output_format):
    """
    Extract map parts
    """
    assert output_format in ('png', 'webp', 'jpg'), f'Unsupported output format {output_format}'

    try:
        job_path = os.path.join('jobs', job_id)
        result_path = os.path.join('results', job_id)

        # decode image
        try:
            image = Image.open(job_path)
        except Exception:
            raise RuntimeError('Nedokážu přečíst nahrávaný soubor, zpracuju jenom .png a .jpg obrázky.')
        gc.collect()

        # check image mode
        if image.mode == 'L':
            raise RuntimeError('Čekal jsem barevný obrázek, ne šedivý')
        elif image.mode == 'RGB':
            image.putalpha(255)
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
        if resize_ratio:
            image = image.resize((image.width // resize_ratio, image.height // resize_ratio), Image.BICUBIC)
            gc.collect()

        # convert to numpy
        image = np.array(image, dtype=np.uint8)
        gc.collect()

        # extract map blobs
        extractor = segmentation.MapExtractor(image)
        extractor.extract()
        if not extractor.segments:
            raise NoMapsFound('Nepodařilo se najít žádné mapy na obrázku.')

        # zip with images
        with zipfile.ZipFile(result_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for n, segment in enumerate(extractor.segments):
                segment_image = Image.fromarray(extractor.draw_segment(segment, shrink=True))
                gc.collect()
                if resize_ratio:
                    segment_image = segment_image.resize((segment_image.width * resize_ratio, segment_image.height * resize_ratio), Image.BICUBIC)
                    gc.collect()

                # create image
                if output_format == 'png':
                    with io.BytesIO() as image_bytes:
                        segment_image.save(image_bytes, format='png')

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

                    zip_file.writestr(f'mapa_{n + 1}.png', compressed_bytes)
                elif output_format == 'webp':
                    with io.BytesIO() as image_bytes:
                        segment_image.save(image_bytes, format='webp')
                        zip_file.writestr(f'mapa_{n + 1}.webp', image_bytes.getvalue())
                elif output_format == 'jpg':
                    background = Image.new('RGB', segment_image.size, (255, 255, 255))
                    segment_mask = segment_image.getchannel('A')
                    jpg_image = Image.composite(segment_image.convert('RGB'), background, segment_mask)
                    with io.BytesIO() as image_bytes:
                        jpg_image.save(image_bytes, format='jpeg', quality=90, optimize=True)
                        zip_file.writestr(f'mapa_{n + 1}.jpg', image_bytes.getvalue())

                del segment_image
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
        del extractor
        del image
        gc.collect()

        job = Job.fetch(job_id, connection=Redis())
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
