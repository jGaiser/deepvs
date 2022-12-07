import pickle
import os
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
INTERACTION_LABELS = protein_config['interaction_labels']

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

def one_hot_update(reference, og_onehot, update_list):
    for item in update_list:
        og_onehot[reference.index(item)] = 1

    return og_onehot

pdb_dir = sorted(glob.glob(config['processed_pdbbind_dir'] + "*/"))
graph_dir = config['positive_voxel_graph_dir']


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

pruned_batch = []

for target_dir in batch:
    target_id = target_dir.split('/')[-2]
    if os.path.exists("%s%s_voxel_positive_graphs.pkl" % (graph_dir, target_id)):
        continue
    pruned_batch.append(target_dir)

batch = pruned_batch
batch_total = len(batch)
# ----------------------- #

for t_idx, target_dir in enumerate(batch):
    print("%s of %s" % (t_idx, batch_total), target_dir)
    target_id = target_dir.split('/')[-2]

    protein_atom_data = []         
    voxel_data = []

    with open("%s%s_2_1_0_3.vox" % (target_dir, target_id), 'r') as vox_in:
        for line in vox_in:
            line = line.rstrip()
            voxel_data.append([float(x) for x in line.split(' ')])

    ip_data = pickle.load(open("%s%s_ip.pkl" % (target_dir, target_id), 'rb'))

    if len(ip_data) == 0:
        continue

    interaction_count = 0

    for interaction_list in ip_data.values():
        interaction_count += len(interaction_list)

    voxel_graph_list = []
    positive_voxels = []
    voxel_y = []
    
    protein_graph = construct_graph(config=graphein_config, pdb_path="%s%s_protein_25.pdb" % (target_dir, target_id))

    for interaction_type, interaction_coords in ip_data.items():
        for interaction_xyz in interaction_coords: 
            closest_voxel = sorted(voxel_data, key=lambda x: get_distance(interaction_xyz, x))[0]
            skip_voxel = False

            if closest_voxel in positive_voxels:
                voxel_idx = positive_voxels.index(closest_voxel)
                skip_voxel = True
            else:
                positive_voxels.append(closest_voxel)
                voxel_y.append([0 for x in INTERACTION_LABELS])
                voxel_idx = -1

            voxel_y[voxel_idx][INTERACTION_LABELS.index(interaction_type)] = 1

            if skip_voxel: continue

            sorted_nodelist = [['DUMMY', 'DUMMY', interaction_xyz, 0]]
            sorted_node_labels = []
            node_features = []

            edge_features = []
            edge_index = [[],[]]

            for i, n in enumerate(protein_graph.nodes(data=True)):
                if n[1]['atom_type'] not in protein_atom_labels:
                    continue  

                n = [n[0], n[1]['atom_type'], n[1]['coords'], get_distance(interaction_xyz, n[1]['coords'])]
                sorted_nodelist.append(n)

            sorted_nodelist = sorted(sorted_nodelist, key=lambda x: x[-1])[:12]
            
            edge_check = [[0 for x in sorted_nodelist] for y in sorted_nodelist]

            for item in sorted_nodelist:
                sorted_node_labels.append(item[0])
                node_features.append(generate_node(protein_atom_labels, item[1], item[2]))

            for i,e in enumerate(protein_graph.edges(data=True)):
                if e[0] not in sorted_node_labels:
                    continue

                if e[1] not in sorted_node_labels:
                    continue
                    
                n1 = sorted_node_labels.index(e[0])
                n2 = sorted_node_labels.index(e[1])

                edge_index[0].extend([n1,n2])
                edge_index[1].extend([n2,n1])
                
                edge_check[n1][n2] = 1
                edge_check[n2][n1] = 1
                
                edge_feature_vec = generate_edge(protein_edge_labels, e[2]['kind'], e[2]['distance'])
                edge_features.extend([edge_feature_vec, edge_feature_vec])
                
            for n1 in range(len(edge_check)):
                for n2 in range(len(edge_check)):
                    if n1 == n2:
                        continue
                        
                    if n1 == 0 or n2 == 0:
                        edge_type = 'interaction'
                    else:
                        edge_type = 'spatial'
                        
                    if edge_check[n1][n2] == 0:
                        edge_index[0].extend([n1,n2])
                        edge_index[1].extend([n2,n1])

                        edge_check[n1][n2] = 1
                        edge_check[n2][n1] = 1

                        node_distance = get_distance(node_features[n1][-3:], node_features[n2][-3:])

                        edge_feature_vec = generate_edge(protein_edge_labels, [edge_type], node_distance)
                        edge_features.extend([edge_feature_vec, edge_feature_vec])
                        
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            node_features = torch.tensor(node_features, dtype=torch.float)
            edge_features = torch.tensor(edge_features, dtype=torch.float)
                
            voxel_graph_list.append({"x": node_features, 
                                     "edge_index": edge_index,
                                     "edge_attr": edge_features})

    final_target_data = []

    for graph_data, graph_y in zip(voxel_graph_list, voxel_y):
        graph_data['y'] = torch.tensor(graph_y, dtype=torch.float32) 
        final_target_data.append(Data(**graph_data))

    pickle.dump(final_target_data, open('%s%s_voxel_positive_graphs.pkl' % (graph_dir, target_id), 'wb'))



