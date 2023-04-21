#!/usr/bin/env python3
"""
OSM query
[out:json][timeout:25];
(
//way["natural"="water"]({{bbox}});
//way["natural"="grassland"]({{bbox}});
//way["natural"="wood"]({{bbox}});
//way["natural"="scrub"]({{bbox}});
//way["natural"="wetland"]({{bbox}});
//way["waterway"]({{bbox}});
//way["landuse"="forest"]({{bbox}});
//way["landuse"="meadow"]({{bbox}});
//way["landuse"="grass"]({{bbox}});
//way["building"]({{bbox}});
//way["highway"]({{bbox}});
//node["admin_level"="8"]({{bbox}});
//way["admin_level"="8"]({{bbox}});
relation["admin_level"="8"]({{bbox}});
);
out body;
>;
out skel qt;
"""

import argparse
import json
import pickle

import geopandas
from shapely.geometry import shape


def parse_map(geojson_map):
    """
    Parse and iterate throgh map features
    """
    for feature in geojson_map['features']:
        assert feature['type'] == 'Feature'
        properties = feature['properties']
        if 'building' in properties:
            yield ('building', shape(feature['geometry']))
        elif properties.get('landuse') == 'forest' or properties.get('natural') in ('wood', 'scrub'):
            yield ('forest', shape(feature['geometry']))
        elif properties.get('landuse') in ('grass', 'meadow') or properties.get('natural') == 'grassland':
            yield ('grass', shape(feature['geometry']))
        elif properties.get('natural') == 'water':
            yield ('water', shape(feature['geometry']))
        elif properties.get('natural') == 'wetland':
            yield ('wetland', shape(feature['geometry']))
        elif properties.get('highway'):
            yield ('road', shape(feature['geometry']))
        elif properties.get('waterway'):
            yield ('river', shape(feature['geometry']))
        elif properties.get('admin_level'):
            yield ('border', shape(feature['geometry']))
        else:
            assert False, feature


def main():
    """
    Anntations helper
    """
    # parse arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('geojson_map',help='geojson map')
    argparser.add_argument('pickle_map',help='pickle map')
    args = argparser.parse_args()

    # read geojson
    with open(args.geojson_map) as fp:
        geojson_map = json.load(fp)

    # parse features
    features = list(parse_map(geojson_map))
    1/0

    # write pickle
    with open(args.pickle_map, 'wb') as fp:
        pickle.dump(features, fp)


# entrypoint
if __name__ == '__main__':
    main()
