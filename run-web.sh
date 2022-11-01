#!/bin/bash

./prepare-web.sh

echo "building docker image"
docker build -t autogeo "$(dirname "$(realpath "$0")")/web"
docker stop autogeo
docker rm autogeo

echo "running docker container"
docker run --publish 127.0.0.1:17859:17859 --cap-drop ALL --cap-add DAC_OVERRIDE --log-opt max-size=10m --log-opt max-file=1 --name autogeo --rm autogeo
