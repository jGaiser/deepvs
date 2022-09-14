import sys
import numpy
import os
import glob
import yaml

with open('config.yaml', 'r') as config_file:  
  config = yaml.safe_load(config_file)

ligand_dir = glob.glob(config['processed_pdbbind_dir'] + "*/")
protein_dir = config['pdbbind_dir']