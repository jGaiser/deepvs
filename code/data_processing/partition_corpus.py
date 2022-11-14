import pickle
import os
import re
import numpy as np
import sys
import glob
import yaml
from copy import deepcopy
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix

with open('config.yaml', 'r') as config_file:  
    config = yaml.safe_load(config_file)

pdb_dir = sorted(glob.glob(config['processed_pdbbind_dir'] + "*/"))
corpus_dir = config['corpus_dir']

# ----- SLURM BATCH ----- #

batch_number = int(sys.argv[1])
batch_count = int(sys.argv[2])

batch_size = int(len(pdb_dir) / batch_count)
batch_remainder = len(pdb_dir) % batch_count

if batch_number <= batch_remainder:
    batch_start_offset = batch_number-1
    batch_end_offset = batch_number 
else:
    batch_start_offset = batch_remainder
    batch_end_offset = batch_remainder

batch_start_index = (batch_number - 1) * batch_size + batch_start_offset
batch_end_index = (batch_number - 1) * batch_size + batch_size + batch_end_offset

if batch_number < batch_count:
    batch = pdb_dir[batch_start_index:batch_end_index]
else:
    batch = pdb_dir[batch_start_index:]

# ----------------------- #
batch_total = len(batch)
corpus_batch = [[], [], []]
corpus_output_file = corpus_dir + "pdbbind_corpus_%s_%s" % (batch_number, batch_count)

for t_idx, target_dir in enumerate(batch):
    target_id = target_dir.split('/')[-2]    
    print(t_idx, target_id)
     
    lg_file = "%s%s_ligand_graph.pkl" % (target_dir, target_id)
    pg_file = "%s%s_pocket_graph.pkl" % (target_dir, target_id)

    if os.path.exists(lg_file) == False:
        continue

    if os.path.exists(pg_file) == False:
        continue

    ligand_graph = pickle.load(open(lg_file,'rb'))
    pocket_graph = pickle.load(open(pg_file,'rb'))

    corpus_batch[0].append(target_id)
    corpus_batch[1].append(ligand_graph)
    corpus_batch[2].append(pocket_graph)

print(corpus_output_file)
pickle.dump(corpus_batch, open(corpus_output_file, 'wb'))



