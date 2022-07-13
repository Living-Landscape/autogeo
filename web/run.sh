#!/bin/bash

# run
docker build -t autogeo "$(dirname "$(realpath "$0")")"
docker stop autogeo
docker rm autogeo
echo "running autogeo"
docker run --detach --network host --cap-drop ALL --cap-add DAC_OVERRIDE --log-opt max-size=10m --log-opt max-file=1 --name autogeo --rm autogeo
