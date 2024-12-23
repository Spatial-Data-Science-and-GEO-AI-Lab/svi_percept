#!/usr/bin/env python3
#
# batch_rate.py - Run the svi_percept model on batches of CLIP features (mainly the output of the clip-retrieval tool).
#
#  Copyright (C) 2024 Matthew Danish
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
################################################################################
#
# This program makes more sense in the context of the Percept project
# https://github.com/Spatial-Data-Science-and-GEO-AI-Lab/percept
#
# It is possible to run it without including --tiles information and also
# without the --output-geojson, in which case you will simply get a numpy file
# with ratings, length N (the number of encoded images from the input) rows and
# 5 columns (walkability, bikeability, pleasantness, greenness, safety).
#
# The GeoJSON output puts various pieces of information from the Percept
# project together, namely:
#   - geometry (lat / lon coordinates)
#   - Mapillary image ID and sequence ID
#   - The five modelled ratings
#   - The angle of the 'camera' (simulated, in the case of panoramic sub-images)
#
# There can be several output features with the same Mapillary image ID because
# each panoramic image could have several subimages cropped and included in the
# dataset.
#
# Asking for the --average-panoramic-ratings means grouping by image ID and
# averaging the ratings of each panoramic subimage together.
#
# Pre-requisites:
#   pip install numpy torch svi_percept geopandas pandas clip_retrieval
#
#   make_tiles_db.py:
#     download https://github.com/Spatial-Data-Science-and-GEO-AI-Lab/percept-vsvi-filter/blob/main/make_tiles_db.py
#
# Example:
#
# Build the pickle file of Mapillary tiles information, also examining the image sequences within ams-seqs/:
#   make_tiles_db.py --seqs ams-seqs/ -o ams-tiles.pkl ams-tiles/
#
# Run the CLIP encoder on numerous images within the sequences directory, output to ams-embeddings/
# (we use open_clip:ViT-H-14-378-quickgelu for the svi_percept model on Huggingface by default)
#   clip-retrieval inference --input_dataset ams-seqs/ --output_folder ams-embeddings/ --clip_model open_clip:ViT-H-14-378-quickgelu
#
# Run the CLIP features through the model (downloaded from Huggingface), output aggregated and averaged GeoJSON features
#   batch_rate.py --embeddings ams-embeddings/ --tiles ams-tiles.pkl --average-panoramic-ratings --output-geojson ams-averaged.geojson
#

import argparse
import numpy as np
import torch
from svi_percept.model import SVIPerceptConfig, SVIPerceptModel
import sys
from pathlib import Path
import geopandas as gpd
import pandas as pd
from math import log10
import pickle
import lzma
from tqdm import tqdm

parser = argparse.ArgumentParser(prog='batch_rate', description='Run SVIPerceptModel on a batch of encoded features')
parser.add_argument('--embeddings', type=str, help='Embeddings dir (output from clip-retrieval) for input (img_emb/ subdir) and output (img_rate/ subdir)', default=None)
parser.add_argument('--tiles', type=str, help='Tiles database (pickled file from pickle_tiles.py)', default=None)
parser.add_argument('--input-numpy-file', type=str, help='Saved numpy file with encoded features (shape: Nx1024) instead of --embeddings', default=None)
parser.add_argument('--metadata-file', type=str, help='Metadata file instead of --embeddings', default=None)
parser.add_argument('--output-numpy-file', type=str, help='Numpy file to write with modelled ratings (shape: Nx5) instead of --embeddings', default=None)
parser.add_argument('--output-geojson', type=str, help='Output combined information to GeoJSON', default=None)
parser.add_argument('--limit', type=int, metavar='N', help='Only process up to N input files no matter how many there are.', default=None)
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files', default=False)
parser.add_argument('--average-panoramic-ratings', action='store_true', help='Average the ratings across the subimages of each panoramic image.', default=False)

# Ensure the input is converted to a numpy array
def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def main():
    args = parser.parse_args()

    if args.input_numpy_file and Path(args.input_numpy_file).exists():
        # Single input file
        input_numpy_files = [Path(args.input_numpy_file)]
    elif args.embeddings and Path(args.embeddings).is_dir():
        # Embeddings dir, with a set of input files under the img_emb/ subdir
        input_numpy_files = sorted((Path(args.embeddings) / 'img_emb').glob('*.npy'))
    else:
        print('One of --embeddings or --input-numpy-file must be specified')
        return 1

    if args.output_numpy_file is None:
        if args.embeddings is None:
            print('One of --embeddings or --output-numpy-file must be specified')
            return 1
        single_output_numpy_file = False

        # Output ratings will go into the img_rate/ subdir of the embeddings dir
        outputdir = Path(args.embeddings) / Path('img_rate')
        outputdir.mkdir(parents=True, exist_ok=True)

        # Enumerate the names of the output files with a fixed-length field for
        # the index of the file; ensure that the fixed-length is long enough to
        # count up to the needed amount (e.g. 2 digits for indices up to 99).
        n = len(input_numpy_files)
        if n == 0:
            print('No input files found.')
            return 1
        elif n == 1:
            digits = 1
        else:
            digits = int(log10(n - 1)) + 1
        output_numpy_files = [outputdir / ('img_rate_{:0'+str(digits)+'d}.npy').format(i) for i in range(n)]
    else:
        # Put all the ratings into a single output file
        single_output_numpy_file = True
        if not args.overwrite and Path(args.output_numpy_file).exists():
            print(f'{args.output_numpy_file} already exists, skipping.')
            return 0
 
    if args.metadata_file:
        # We were given a single metadata file to read
        metadata = pd.read_parquet(args.metadata_file)
        metadata_i = 0
    elif args.embeddings:
        # Get the metadata from the metadata subdir of the embeddings dir;
        # pandas can automatically read and combine all the parts.
        metadata = pd.read_parquet(Path(args.embeddings) / 'metadata')
        metadata_i = 0
    else:
        metadata = None
    if metadata is not None:
        metadata_len = len(metadata)
        print(f'Read metadata file with {metadata_len} rows.')

    if args.tiles:
        # Open our compressed Mapillary tile information database
        with lzma.open(args.tiles, 'rb') as fp:
            tiles = pickle.load(fp)
    else:
        tiles = None

    if args.output_geojson and tiles and metadata is not None:
        if not args.overwrite and Path(args.output_geojson).exists():
            output_geojson = None
        else:
            output_geojson = Path(args.output_geojson)
            geofeatures = []
    else:
        output_geojson = None
        if not tiles:
            print('WARNING: We cannot produce output GeoJSON without the tiles information.')

    # Construct the model directly, skip the pipeline
    model = SVIPerceptModel(SVIPerceptConfig())

    if single_output_numpy_file:
        # Accumulator var for all the results
        accum_results = []

    # Apply limit after generating output file names; all else should run as if
    # limit was not in place, but then we cut it off.
    if args.limit:
        input_numpy_files = input_numpy_files[:args.limit]

    for input_numpy_file in tqdm(input_numpy_files):
        if not args.overwrite and not single_output_numpy_file and output_numpy_files[0].exists():
            # Do not overwrite the existing output file, instead read it.
            results = np.load(output_numpy_files[0])
            output_numpy_files = output_numpy_files[1:]
        else:
            # Load the input CLIP features from the current input file
            clipfeatures = np.load(input_numpy_file)
            clipfeatures = torch.from_numpy(clipfeatures.astype(np.float32))
            # Run the model on all the features
            results = to_numpy(model(clipfeatures)['results'])

            if single_output_numpy_file:
                # Build up the results for a single file output
                accum_results.append(results)
            else:
                # Generate an output file for each input file
                np.save(output_numpy_files[0], results)
                output_numpy_files = output_numpy_files[1:]

        if output_geojson:
            # Analyze the current results for inclusion in the output GeoJSON file
            for i in tqdm(range(results.shape[0])):
                ratings = results[i]
                # The metadata has but one piece of information: the names of
                # the image files that correspond to each row in the input
                # numpy file(s).
                imagepath = Path(metadata.iloc[metadata_i].values[0])
                # We keep this index separately because this arrangement works
                # in both single- and multiple-input file cases.
                metadata_i += 1
                imagestem = imagepath.stem
                # Filename schema: <imgid>_x<pixel_offset>
                # Hence, the imgid can be read before the '_' in the filename
                imgid = int(imagestem[:imagestem.rfind('_')] if '_' in imagestem else imagestem)
                if imgid not in tiles:
                    print(f'Unable to find {imgid} in the tiles database.')
                    continue
                entry = tiles[imgid]
                # Assemble the 'properties' section of the GeoJSON Feature
                props = { 'imgid': imgid, 'imagepath': imagepath }
                for cat_i, cat in enumerate(model.categories):
                    props[cat] = round(ratings[cat_i], 1)
                for x in ['seqid', 'angle']: props[x] = entry[x]

                if entry['is_pano'] and 'x' in imagestem and 'image_width' in entry:
                    # Panoramic image files are stored in Mapillary with a
                    # convention that the lefthand-edge (pixel 0) of the image
                    # corresponds to due North (angle = 0).
                    #
                    # We keep track of the <pixel_offset> of a subimage of a
                    # panoramic image, in the filename, after the '_x'.
                    #
                    # Take the <pixel_offset> from the filename and convert it
                    # into an angle by dividing it by the total width of the
                    # original panoramic image.
                    #
                    # Subimages were cropped to be image_width / 4, therefore
                    # the central X coordinate in each subimage is image_width
                    # divided by 8. 360 / 8 = 45 degrees, hence the extra +45
                    # in the formula below.
                    x = int(imagestem[imagestem.rfind('x')+1:])
                    w = entry['image_width']
                    props['angle'] = round(props['angle'] + 45 + 360 * x / w, 1) % 360

                lat, lon = entry['lat'], entry['lon']
                geo = { 'type' : 'Point', 'coordinates': (lon, lat) }
                geofeatures.append({ 'type': 'Feature', 'properties': props, 'geometry': geo })

    if single_output_numpy_file:
        results = np.concatenate(accum_results, axis=0)
        np.save(args.output_numpy_file, results)

    if output_geojson:
        geofeaturecol = { 'type': 'FeatureCollection', 'features': geofeatures }
        gdf = gpd.GeoDataFrame.from_features(geofeaturecol)

        # Ensure that any cases of duplicate subimages (e.g. at angles
        # 5-degrees and 365-degrees) are resolved in favor of the first
        # occurrence.
        gdf['int_angle'] = (round(gdf['angle'], 1) * 10).astype(int) # avoid issues of floating-point comparison
        gdf['angle'] = gdf["int_angle"] / 10
        aggconditions = \
                {'geometry': 'first', 'seqid': 'first', 'angle': 'first'} | \
            { cat: 'first' for cat in model.categories }
        gdf = gdf.groupby(['imgid','int_angle']).agg(aggconditions)

        if args.average_panoramic_ratings:
            # Aggregate rows with the same image ID, as they were cropped from
            # the same panoramic image originally. Average the ratings
            # together, and discard the angle as it no longer makes sense.
            aggconditions = \
                {'geometry': 'first', 'seqid': 'first', 'angle': lambda x: set(x)} | \
                { cat: 'mean' for cat in model.categories }
            gdf = gdf.groupby('imgid').agg(aggconditions).reset_index()
        else:
            gdf = gdf.reset_index().drop('int_angle', axis=1)

        gdf.to_file(output_geojson, driver='GeoJSON')

    return 0

if __name__=='__main__':
    sys.exit(main())
