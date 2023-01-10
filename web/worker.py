# author: Jan Prochazka
# license: public domain

import os
import io
import zipfile
import gc
import logging

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


def process(job_id):
    """
    Extract map parts
    """
    try:
        job_path = os.path.join('jobs', job_id)
        result_path = os.path.join('results', job_id)

        # decode image
        try:
            image = Image.open(job_path)
        except Exception:
            raise RuntimeError('Nedokážu přečíst nahrávaný soubor, zpracuju jenom .png a .jpg obrázky.')
        gc.collect()

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

        if image.mode == 'L':
            raise RuntimeError('Čekal jsem barevný obrázek, ne šedivý')
        elif image.mode == 'RGB':
            image.putalpha(255)
        elif image.mode == 'RGBA':
            pass
        else:
            raise RuntimeError('Obrázek má neznámý formát, umím pracovat jenom s rgb a rgba kanály v .png a .jpg obrázcích.')
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
                with io.BytesIO() as image_bytes:
                    segment_image.save(image_bytes, format='png')
                    zip_file.writestr(f'mapa_{n + 1}.png', image_bytes.getvalue())
                del segment_image
                gc.collect()
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
