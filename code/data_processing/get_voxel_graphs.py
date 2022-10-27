import pickle
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

pdb_dir = sorted(glob.glob(config['processed_pdbbind_dir'] + "*/"))
graph_dir = config['interaction_voxel_graph_dir']

# corpus = pickle.load(open(graph_dir + 'interaction_voxel_corpus.pkl', 'rb'))

target_id_list = []
voxel_graph_list = []

for t_idx, target_dir in enumerate(pdb_dir):
    if t_idx % 100 == 0:
        print(t_idx)

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

    protein_graph = construct_graph(config=graphein_config, pdb_path="%s%s_protein_25.pdb" % (target_dir, target_id))

    for interaction_type, interaction_coords in ip_data.items():
        for interaction_xyz in interaction_coords: 
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
                
            data = Data(x=node_features, edge_index=edge_index, 
                        edge_attr=edge_features, y=interaction_labels.index(interaction_type))
            
            target_id_list.append(target_id)
            voxel_graph_list.append(data)

pickle.dump([target_id_list, voxel_graph_list], open(graph_dir + 'interaction_voxel_corpus.pkl', 'wb'))