#!/usr/bin/env python3
# author: Jan Prochazka
# license: public domain
"""
Web application for map segmentation
"""

import os
import io
import gc
import secrets
import base64
import logging
from functools import wraps
import time
from datetime import datetime, timedelta

from flask import Flask, send_from_directory, request, make_response
from werkzeug.utils import secure_filename
import waitress
from redis import Redis
import rq
from rq import Queue
from rq.job import Job

import imagesize
import worker

# TODO cancel jobs, also on tab unload (navigator.sendBeacon)


# create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s')

# create flask application
app = Flask(__name__)

# create redis queue
redis = Redis()
queue = Queue(connection=redis)


def generic_error_message(error):
    """
    Return error message
    """
    return f'Při zpracování nastala chyba ({error}).'


def generate_random_id(length):
    """
    Generate random id
    """
    id = base64.b64encode(secrets.token_bytes(length)).decode('ascii')
    return id.upper().replace('+', 'a').replace('/', 'z')


def log_request(fn):
    """
    Log decorated reqeuests
    """
    @wraps(fn)
    def wrap(*args, **kwargs):
        start_time = time.monotonic()
        request_id = 'req' + generate_random_id(6)
        logger.info('%s new request %s %s', request_id, request.method, request.url)
        try:
            request.id = request_id
            return fn(*args, **kwargs)
        except Exception as exc:
            logger.exception('%s request exception: %s', request_id, str(exc))
        finally:
            logger.info('%s finished in %s', request_id, round(time.monotonic() - start_time, 3))

    return wrap


@app.route('/icon.png')
def image_icon():
    response = make_response(send_from_directory('.', 'web_icon.png'))
    response.headers['cache-control'] = 'public'
    response.headers['expires'] = (datetime.utcnow() + timedelta(30)).strftime("%a, %d %b %Y %H:%M:%S GMT")
    return response


@app.route('/preview.png')
def image_preview():
    response = make_response(send_from_directory('.', 'web_preview.png'))
    response.headers['cache-control'] = 'public'
    response.headers['expires'] = (datetime.utcnow() + timedelta(30)).strftime("%a, %d %b %Y %H:%M:%S GMT")
    return response


@app.route('/download')
@log_request
def download():
    """
    Download job results
    """
    try:
        # parse job id
        job_id = request.args.get('id')
        if job_id is None or len(job_id) != 43:
            return {'error': generic_error_message('Provided job id is invalid')}, 422

        # get the job
        try:
            job = Job.fetch(job_id, connection=redis)
        except rq.exceptions.NoSuchJobError:
            return {'error': generic_error_message(f'Job with id {job_id} expired')}, 404
        except Exception as exc:
            return {'error': generic_error_message(str(exc))}, 500

        # return response
        status = job.get_status(refresh=False)
        logger.info('%s %s job status %s', request.id, job_id, status)
        if status == 'failed':
            return {'status': status, 'error': generic_error_message(job.exc_info.strip().split('\n')[-1])}
        elif status == 'finished':
            download = job.meta.get('download')
            info = job.meta.get('info')
            error = job.meta.get('error')
            if download == True:
                response = make_response(send_from_directory('results', job_id))
                response.headers['Content-Type'] = 'application/zip;charset=UTF-8'
                response.headers['Content-Disposition'] = f'attachment;filename={job.meta["name"]}.zip'
                del job
                gc.collect()
                return response
            elif info is not None:
                return {'status': status, 'info': info}
            elif error is not None:
                return {'status': 'failed', 'error': generic_error_message(error)}
            else:
                return {'status': 'failed', 'error': generic_error_message('Application internal error')}
        else:
            return {'status': status, 'info': 'Not ready yet'}
    except Exception as exc:
        message = str(exc)
        logger.exception('%s request exception: %s', request.id, message)
        return {'error': generic_error_message(message)}, 500


@app.route('/status', methods=['POST'])
def status():
    """
    Get job status
    """
    try:
        # parse params
        job_ids = request.form.get('ids')
        if job_ids is None:
            return {'error': generic_error_message('Missing job ids parameter.')}, 422

        max_job_ids = 10
        job_ids = set(job_ids.split(','))
        if len(job_ids) > max_job_ids:
            return {'error': generic_error_message(f'You can request at most {max_job_ids} job statuses at once.')}, 422

        # fetch job ids
        try:
            jobs = Job.fetch_many(set(job_ids), connection=redis)
        except Exception as exc:
            return {'error': generic_error_message(str(exc))}, 500

        # fetch start of the queue
        max_queue_position = 10
        enqueued_ids = queue.get_job_ids(length=max_queue_position)

        # iterate through requested ids
        results = {}
        for job in jobs:
            # get response
            status = job.get_status(refresh=False)
            if status == 'failed':
                message = job.exc_info.strip().split('\n')[-1]
                results[job.id] = {'status': 'failed', 'error': generic_error_message(message)}
            elif status == 'finished':
                download = job.meta.get('download')
                info = job.meta.get('info')
                error = job.meta.get('error')
                if error is None:
                    result = {'status': 'finished'}
                    if download == True:
                        result['download'] = True
                    if info is not None:
                        result['info'] = info
                    results[job.id] = result
                else:
                    results[job.id] = {'status': 'failed', 'error': generic_error_message(error)}
            elif status == 'queued':
                try:
                    position = str(enqueued_ids.index(job.id) + 1)
                except ValueError:
                    position = f'{max_queue_position}+'
                results[job.id] = {'status': status, 'position': position}
            else:
                results[job.id] = {'status': status}

        return results
    except Exception as exc:
        message = str(exc)
        logger.exception('%s exception %s', request.i)
        return {'error': generic_error_message(message)}, 500


@app.route('/upload', methods=['POST'])
@log_request
def upload():
    """
    Return main page
    """
    try:
        # parse uploaded file
        if 'upload' not in request.files:
            return {'error': 'Chybí nahrávaný soubor.'}, 422
        file = request.files['upload']

        if file.filename == '':
            return {'error': 'Nahrávanému souboru chybí jméno.'}, 422

        # check queue size
        max_queue_size = 100
        if len(queue) > max_queue_size:
            return {'error': 'Server nyní zpracovává mnoho požadavků, počkejte chvíli a pak to zkuste znovu.'}, 422

        max_file_size = 20 * 1024 ** 2
        source_name = secure_filename(file.filename)
        name = os.path.splitext(source_name)[0]
        image_bytes = file.stream.read(max_file_size + 1)
        if len(image_bytes) == max_file_size + 1:
            return {'error': f'Maximální velikost souboru je {max_file_size // 1024 ** 2}MB.'}, 422

        # check dimensions
        max_dimension = 10000
        min_dimension = 100
        try:
            width, height = imagesize.get(io.BytesIO(image_bytes))
            logger.info('%s image size %sx%s', request.id, width, height)
        except ValueError:
            return {'error': 'Nedokážu přečíst nahrávaný soubor, zpracuju jenom .png a .jpg obrázky.'}, 422
        if width > max_dimension or height > max_dimension:
            return {'error': f'Maximální velikost obrázku je {max_dimension}x{max_dimension}, nahraný má {width}x{height}'}, 422
        if width < min_dimension or height < min_dimension:
            return {'error': f'Miminální velikost obrázku je {min_dimension}x{min_dimension}, nahraný má {width}x{height}'}, 422
        gc.collect()

        # check queue size again
        if len(queue) > max_queue_size:
            return {'error': 'Server nyní zpracovává mnoho požadavků, počkejte chvíli a pak to zkuste znovu.'}, 422

        # generate job id
        job_id = 'job' + generate_random_id(30)
        logger.info('%s %s new job for %s', request.id, job_id, name)

        # save image to temporary store
        with open(os.path.join('jobs', job_id), 'wb') as job_file:
            job_file.write(image_bytes)

        # enqueue job
        queue.enqueue(
            worker.process,
            args=(job_id,),
            job_id=job_id,
            job_timeout=600,
            result_ttl=600,
            failure_ttl=600,
            meta={
                'name': name,
            },
        )
        logger.info('%s %s job enqueued', request.id, job_id)

        # get queue position
        max_queue_position = 10
        position = max(1, len(queue))
        position = f'{max_queue_position}+' if position > max_queue_position else str(position)

        return {'status': 'queued', 'jobId': job_id, 'position': position}
    except Exception as exc:
        message = str(exc)
        logger.exception('%s exception %s', request.id, message)
        return {'error': generic_error_message(message)}, 500


@app.route('/')
def index():
    """
    Return main page
    """
    return web_template


# entrypoint
if __name__ == '__main__':
    # read web tamplate
    with open('web.html') as web_file:
        web_template = web_file.read()

    # serve flask application
    host = '0.0.0.0'
    port = 17859
    logger.info('serving on %s:%s', host, port)
    waitress.serve(app, host=host, port=port)
