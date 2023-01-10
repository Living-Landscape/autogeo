#!/usr/bin/env python3
# author: Jan Prochazka
# license: public domain
"""
Cleans up job data
"""

import os
import random
import logging
import time


# create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s')


def cleanup(path, ttl, cache):
    """
    Cleanup path for stale files
    """
    # TODO will fail to behave correctly when system time is changed
    now = time.time()
    for job_id in os.listdir(path):
        file = os.path.join(path, job_id)
        if file in cache:
            file_stat = cache[file]
        else:
            file_stat = os.stat(file)
            cache[file] = file_stat

        if now - file_stat.st_mtime > ttl:
            try:
                del cache[file]
                os.unlink(file)
                logger.info('%s removed from %s', job_id, path)
            except Exception:
                logger.exception('%s cannot remove from %s', job_id, path)


def check_cleanup():
    """
    Periodicaly checks and cleanup job data
    """
    initial_delay = 5
    cleanup_delay = 60
    jobs_ttl = 600
    results_ttl = 720

    job_cache = {}
    result_cache = {}

    time.sleep(initial_delay + random.random() * initial_delay / 2)
    while True:
        logger.info('cleaning up jobs and results')
        cleanup('jobs', jobs_ttl, job_cache)
        cleanup('results', results_ttl, result_cache)
        time.sleep(cleanup_delay + random.random() * cleanup_delay / 2)


# entrypoint
if __name__ == '__main__':
    check_cleanup()

