#!/bin/bash

# run
docker build -t autogeo "$(dirname "$(realpath "$0")")"
docker stop autogeo
docker rm autogeo
echo "running autogeo"
docker run --detach --publish 127.0.0.1:17859:17859 --cap-drop ALL --cap-add DAC_OVERRIDE --log-opt max-size=10m --log-opt max-file=1 --name autogeo --restart unless-stopped autogeo
