import pickle
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

import rdkit_utils

with open('config.yaml', 'r') as config_file:  
    config = yaml.safe_load(config_file)

with open(config['protein_config_file'], 'r') as config_file:  
    protein_config = yaml.safe_load(config_file)

INTERACTION_LABELS = protein_config['interaction_labels']
# ['halogenbond', 'hbond_a', 'hbond_d', 'hydroph_interaction', 'pication_c', 'pication_r', 'pistack', 'saltbridge_n', 'saltbridge_p']
mol_graph_dir = config['mol_graph_dir']

def get_distance(v1,v2):
    total = 0

    for a,b in zip(v1,v2):
        total += (a-b)**2

    return total**0.5

def one_hot_update(reference, og_onehot, update_list):
    for item in update_list:
        og_onehot[reference.index(item)] = 1

    return og_onehot

pdb_dir = sorted(glob.glob(config['processed_pdbbind_dir'] + "*/"))

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

total = len(pdb_dir)

atom_types = []

for t_idx, target_dir in enumerate(batch):
    target_id = target_dir.split('/')[-2]    
    print("\n", target_id, "%s of %s" % (t_idx, batch_total))
    # if t_idx % 100 == 0:
    #     print(target_id, "%s of %s" % (t_idx, total))

    pdb_file_content = ""
    pdb_H_count = 0
    pdb_heavy_atom_data = []

    with open("%s%s_ligand.pdb" % (target_dir, target_id), 'r') as pdb_in:
        for line in pdb_in:
            pdb_file_content += line

            if line[:6].strip() in ['ATOM', 'HETATM']:
                if line[76:78].strip() == 'H':
                    pdb_H_count += 1
                    continue

                pdb_heavy_atom_data.append([line[12:16].strip(),
                                float(line[30:38].strip()),
                                float(line[38:46].strip()),
                                float(line[46:54].strip())])

    mol_y = np.zeros((len(pdb_heavy_atom_data), len(INTERACTION_LABELS)))
    ip = pickle.load(open("%s%s_ip.pkl" % (target_dir, target_id), 'rb'))

    for itype, icoords in ip.items():
        # print(itype.upper(), "\n---")

        for xyz in icoords:
            pdb_data_distances = np.array([get_distance(x[-3:], xyz) for x in pdb_heavy_atom_data])
            sorted_pdb_data_indices = np.argsort(pdb_data_distances)
            # print(sorted_pdb_data_indices)


            if itype in ['pication_r', 'pistack']:
                min_distance = pdb_data_distances[sorted_pdb_data_indices[0]]

                for a_i, atom_idx in enumerate(sorted_pdb_data_indices):
                    if pdb_data_distances[atom_idx] - min_distance > 0.5:
                        break

                interacting_atoms = sorted_pdb_data_indices[:a_i]
            else:
                interacting_atoms = [sorted_pdb_data_indices[0]]

            for atom_idx in interacting_atoms:
                mol_y[atom_idx] = one_hot_update(INTERACTION_LABELS, mol_y[atom_idx], [itype])

    mol_y = torch.vstack([torch.tensor(mol_y), torch.zeros((pdb_H_count, len(INTERACTION_LABELS)))])

    # with open("%s%s_ligand.smi" % (target_dir, target_id), 'r') as smi_in:
    #     for line in smi_in:
    #         smi_string = re.split(r'\s+', line)[0]

    try:
        molecule = Chem.rdmolfiles.MolFromPDBBlock(pdb_file_content, removeHs=False)

        # molecule = Chem.MolFromSmiles(smi_string)
        g = rdkit_utils.generate_mol_graph(molecule, mol_y)

        # admatrix = rdmolops.GetAdjacencyMatrix(molecule)
        # print(molecule)
        # sys.exit()

        # edge_index = [[],[]]

        # for b in molecule.GetBonds():
        #     edge_index[0].append(b.GetBeginAtomIdx())
        #     edge_index[1].append(b.GetEndAtomIdx())

        # edge_index[0].extend(list(edge_index[1]))
        # edge_index[1].extend(list(edge_index[0][:len(edge_index[1])]))

        # edge_file_content += " ".join([id_key[x] for x in edge_index[0]]) + "\n"
        # edge_file_content += " ".join([id_key[x] for x in edge_index[1]]) + "\n"

        # with open(output_dir + pdb_file.split('/')[-1].split('.')[0] + '.edges', 'w') as edges_out:
        #     edges_out.write(edge_file_content)
    except Exception as e:
        print(e, target_dir)

    # print("%s%s_ligand_graph.pkl" % (target_dir, target_id))
    pickle.dump(g, open("%s%s_mol_graph.pkl" % (mol_graph_dir, target_id), 'wb'))
