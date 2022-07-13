FROM debian:bullseye-slim

RUN apt-get update && \
	apt-get install --no-install-recommends --yes \
		python3 \
		python3-opencv \
		python3-pil \
		python3-numba \
		python3-waitress \
		python3-flask \
		redis-server \
		python3-rq \
		supervisor && \
	mkdir /app && \
	mkdir /app/jobs && \
	mkdir /app/results

COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY web.py web.html web_preview.png web_icon.png segmentation.py imagesize.py worker.py cleaner.py /app/

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/supervisord.conf"]
