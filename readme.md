# Automatic georeferencing of historical maps

Project aims to automatically georeference "Císařské otisky" from https://ags.cuzk.cz/archiv/. The process is multistage and in progress. The result will be the web service where an users can georeference their own maps.

 - [x] web service with task queues
 - [x] strip down maps background
 - [] actual georeferencing

## Installation

 * `autogeo` web server runs in docker container
 * `prepare-docker.sh` - prepare necessary files in `docker` folder
 * `cd docker && ./run.sh` - run production version of `autogeo` web service

## Development
 * `segmentation.{py,ipynb}` - script / notebook for automatic map foreground extraction
 * `docker` folder - sources for web service
 * `run-docker.sh` - runs `autogeo` web service locally
