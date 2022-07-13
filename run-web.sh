#!/bin/bash

./prepare-web.sh

echo "building docker image"
docker build -t autogeo "$(dirname "$(realpath "$0")")/web"
docker stop autogeo
docker rm autogeo

echo "running docker container"
docker run --network host --cap-drop ALL --cap-add DAC_OVERRIDE --log-opt max-size=10m --log-opt max-file=1 --name autogeo --rm autogeo
