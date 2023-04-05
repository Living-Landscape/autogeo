#!/bin/bash
set -e

# run
echo "building docker"
docker build -t autogeo "$(dirname "$(realpath "$0")")"

echo "stoping previous containers"
docker stop autogeo || /bin/true
docker rm autogeo || /bin/true

echo "running autogeo"
if [ "$1" == 'dev' ]; then
	docker run --publish 127.0.0.1:17859:17859 --cap-drop ALL --cap-add DAC_OVERRIDE --log-opt max-size=10m --log-opt max-file=1 --name autogeo --rm autogeo
else
	docker run --detach --publish 127.0.0.1:17859:17859 --cap-drop ALL --cap-add DAC_OVERRIDE --log-opt max-size=10m --log-opt max-file=1 --name autogeo --restart unless-stopped autogeo
fi
