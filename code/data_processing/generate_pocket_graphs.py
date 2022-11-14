import pickle
import numpy as np
import sys
import glob
import yaml
from copy import deepcopy
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph
from graphein.protein.edges.atomic import add_atomic_edges, add_bond_order, add_ring_status 
from graphein.protein.edges.distance import node_coords
from graphein.protein.config import GraphAtoms 
import torch
from torch_geometric.data import Data

with open('config.yaml', 'r') as config_file:  
  config = yaml.safe_load(config_file) 

with open(config['protein_config_file'], 'r') as config_file:  
  protein_config = yaml.safe_load(config_file)

protein_atom_labels = protein_config['atom_labels']
protein_edge_labels = protein_config['edge_labels']
interaction_labels = protein_config['interaction_labels']

graphein_param_dict = {"granularity": "atom", 
                       "edge_construction_functions": [add_atomic_edges, add_bond_order, add_ring_status],
                       "deprotonate": True,
                       "verbose": False}

graphein_config = ProteinGraphConfig(**graphein_param_dict)

def get_distance(x,y):
    total = 0 

    for a,b in zip(x,y):
        total += (a-b)**2

    return total**(0.5)

def generate_node(node_type_list, node_type, coords):
    feature_vec = [0 for x in range(len(node_type_list))]
    feature_vec[node_type_list.index(node_type)] = 1
    feature_vec.extend(coords)
    return feature_vec

def generate_edge(edge_feature_list, edge_features, weight):
    feature_vec = [0 for x in range(len(edge_feature_list))]
    
    for item in edge_features:
        feature_vec[edge_feature_list.index(item)] = 1
        
    feature_vec.append(weight)
    return feature_vec

def update_edge_index(edge_index, v1, v2):
    source_list = edge_index[0] + [v1,v2]
    sink_list =   edge_index[1] + [v2,v1]
    return [source_list, sink_list]

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

for t_idx, target_dir in enumerate(batch):
    target_id = target_dir.split('/')[-2]
    print(target_id, "%s of %s" % (t_idx, batch_total))

    voxel_data = []
    protein_atom_nodes = []
    added_protein_atoms = [] 

    nodelabels = []
    nodelist = []
    edgelist = []
    edge_index = [[],[]]

<<<<<<< HEAD
=======
    with open("%s%s_2_1_0_3.vox" % (target_dir, target_id), 'r') as vox_in:
>>>>>>> ab469db92b2cf54761b464dc0461c7f707447d12
        for line in vox_in:
            line = line.rstrip()
            voxel_data.append([float(x) for x in line.split(' ')])

    protein_graph = construct_graph(config=graphein_config, pdb_path="%s%s_protein_25.pdb" % (target_dir, target_id))

    for n in protein_graph.nodes(data=True):
        if n[1]['atom_type'] not in protein_atom_labels:
            continue  

        protein_atom_nodes.append([n[0]] + generate_node(protein_atom_labels, n[1]['atom_type'], n[1]['coords']))

    for voxel_idx, voxel_xyz in enumerate(voxel_data):
        nodelabels.append("DUMMY_%s" % voxel_idx)
        nodelist.append(generate_node(protein_atom_labels, 'DUMMY', voxel_xyz))

    voxel_count = len(nodelist)

    for node_id, voxel_node in enumerate(nodelist[:voxel_count]):
        for node_id_2, voxel_node_2 in enumerate(nodelist[:voxel_count]):
            if node_id_2 <= node_id:
                continue 

            voxel_distance = get_distance(voxel_node[-3:], voxel_node_2[-3:])

            # Add voxel edges between neighboring voxels (face adjacent)
            if voxel_distance <= 2.01 and voxel_distance > 0.01:
                edge_index = update_edge_index(edge_index, node_id, node_id_2)
                edge_features = generate_edge(protein_edge_labels, ['voxel'], voxel_distance)
                edgelist += [edge_features, edge_features]

        neighbor_protein_atoms = sorted(protein_atom_nodes, key=lambda x: get_distance(voxel_node[-3:], x[-3:]))[:10]

        for protein_node in neighbor_protein_atoms:
            if protein_node[0] not in nodelabels:
                protein_node_id = len(nodelist)
                nodelabels.append(protein_node[0])
                nodelist.append(protein_node[1:])
            else:
                protein_node_id = nodelabels.index(protein_node[0])

            edge_index = update_edge_index(edge_index, node_id, protein_node_id)
            edge_features = generate_edge(protein_edge_labels, ['interaction'], get_distance(voxel_node[-3:], protein_node[-3:]))
            edgelist += [edge_features, edge_features]

        edge_source = np.array(edge_index[0])
        edge_sink = np.array(edge_index[1])
        
    for e in protein_graph.edges(data=True):
        if e[0] not in nodelabels:
            continue

        if e[1] not in nodelabels:
            continue

        v1 = nodelabels.index(e[0])
        v2 = nodelabels.index(e[1])

        edge_features = generate_edge(protein_edge_labels, e[2]['kind'], e[2]['distance'])        
        edge_index = update_edge_index(edge_index, v1, v2)
        edgelist += [edge_features, edge_features]

    edge_index = torch.tensor(edge_index, dtype=torch.long)
    node_features = torch.tensor(nodelist, dtype=torch.float)
    edge_features = torch.tensor(edgelist, dtype=torch.float)
        
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features)
    pickle.dump(data, open("%s%s_pocket_graph.pkl" % (target_dir, target_id), 'wb'))
    print("%s%s_pocket_graph.pkl" % (target_dir, target_id))

