# Automatic georeferencing of historical maps

Project aims to automatically georeference "Císařské otisky" from https://ags.cuzk.cz/archiv/. The process is multistage and in progress. The result will be the web service where an users can georeference their own maps.

 * [x] web service with task queues
 * [x] strip down maps background
 * [ ] actual georeferencing

## Installation

 * `autogeo` web service runs in docker container
 * `./prepare-web.sh` - prepare necessary files in `web` folder
 * `cd web && ./run.sh` - run production version of `autogeo` web service
 * `autogeo` web service runs inside docker on port 17859

## Development
 * `segmentation.{py,ipynb}` - script / notebook for automatic map foreground extraction
 * `web` folder - sources for web service
 * `./prepare-web.sh` - prepare `web` folder to be used for building docker image
 * `./run-web.sh` - runs `autogeo` web service locally
