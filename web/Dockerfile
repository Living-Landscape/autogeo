FROM debian:bullseye-slim

RUN apt-get update && \
	# install base packages \
	apt-get install --no-install-recommends --yes \
		python3 \
		python3-opencv \
		python3-pil \
		python3-numba \
		python3-waitress \
		python3-flask \
		redis-server \
		python3-rq \
		supervisor \
		pngquant \
		wget \
		ca-certificates && \
	# install oxipng \
	wget https://github.com/shssoichiro/oxipng/releases/download/v8.0.0/oxipng-8.0.0-x86_64-unknown-linux-musl.tar.gz && \
	tar -xzf oxipng-8.0.0-x86_64-unknown-linux-musl.tar.gz && \
	mv oxipng-8.0.0-x86_64-unknown-linux-musl/oxipng /usr/local/bin/oxipng && \
	rm -r oxipng-8.0.0-x86_64-unknown-linux-musl* && \
	apt-get autoremove --yes \
		wget \
		ca-certificates && \
	# prepare application folders \
	mkdir /app && \
	mkdir /app/jobs && \
	mkdir /app/results

# install main app
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY web.py web.html web_preview.png web_icon.png segmentation.py imagesize.py worker.py cleaner.py /app/

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/supervisord.conf"]
