# Automatic georeferencing of historical maps

Project aims to segment and georeference "Císařské otisky" from https://ags.cuzk.cz/archiv/. The process is multistage and in progress. The result will be the web service where an users can georeference their own maps.

![segmentation](https://github.com/Living-Landscape/autogeo/blob/d51055e019b3b07fc72cb8224718e6a1b502fc99/docs/segmentation.png)

 * [x] web service with task queues
 * [x] segmentation of maps, water, dry and wet meadows
 * [x] output in png, jpg or webp
 * [ ] georeferencing

## Installation

 * `autogeo` web service runs in docker container
 * `cd web && ./run.sh` - run production version of `autogeo` web service
 * `autogeo` web service runs inside docker on port 17859

## Development
 * `segmentation.{py,ipynb}` - script / notebook for automatic map foreground extraction
 * `web` folder - sources for web service
 * `cd web && ./run.sh dev` - run development version of `autogeo` web service
 * `autogeo` web service runs inside docker on port 17859
