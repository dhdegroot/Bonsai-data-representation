from argparse import ArgumentParser
import os, sys

# quick fix for SARAH because I had OPENBLAS error...
# OpenBLAS blas_thread_init: pthread_create failed for thread 43 of 64: Resource temporarily unavailable
# OpenBLAS blas_thread_init: RLIMIT_NPROC 4096 current, 2062560 max
# os.environ['OPENBLAS_NUM_THREADS'] = '1'

from ruamel.yaml import YAML
import subprocess

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# Add the parent directory of this script-file to sys.path
sys.path.append(parent_dir)
os.chdir(parent_dir)

parser = ArgumentParser(
    description='Small wrapper around Bonsai Shiny app (shiny_bonsai.py) to parse some arguments.')

parser.add_argument('--results_folder', type=str, default=None,
                    help='Absolute (or relative to "bonsai-development") path to bonsai results folder that'
                         'contains the files created by vis_bonsai_preprocess: bonsai_vis_settings.json and'
                         'bonsai_vis_data.hdf')
parser.add_argument('--settings_filename', type=str, default='bonsai_vis_settings.json',
                    help='Filename of json-file that contains app settings. This file should be present in the '
                         'results_folder. Note that new json-files are created when one clicks "Store settings"'
                         'in the app.')
parser.add_argument('--port', type=int, default=-1,
                    help='Port at which the app will be running. Pick your favourite number over a 1000')

"""-----------------PARSE ARGUMENTS AND STORE IN SETTINGS FILE--------------------"""


args = parser.parse_args()

if args.results_folder is None:
    print("Path to Bonsai-results was not given as an argument.")

os.environ['BONSAI_DATA_PATH'] = os.path.abspath(os.path.join(args.results_folder, 'bonsai_vis_data.hdf'))
if not os.path.exists(os.environ['BONSAI_DATA_PATH']):
    print("Preprocessed results for Bonsai vis were not found. Make sure to run 'vis_bonsai_preprocess.py'.")

os.environ['BONSAI_SETTINGS_PATH'] = os.path.abspath(os.path.join(args.results_folder, args.settings_filename))
if not os.path.exists(os.environ['BONSAI_SETTINGS_PATH']):
    print("Settings file for Bonsai vis were not found at {}. Make sure to run 'vis_bonsai_preprocess.py'.".format(
        os.environ['BONSAI_SETTINGS_PATH']))

"""-------------------------------------Start shiny app----------------------------------"""
import random
if args.port == -1:
    port = random.randint(1025, 9999)
else:
    port = args.port
# port = 1234
print("Your app will shortly be running at: http://localhost:{}. Use your browser (not Safari) to view it.".format(port))
subprocess.run(['shiny', 'run', 'bonsai_scout/app.py', '--port={}'.format(port), '--host=0.0.0.0'])
