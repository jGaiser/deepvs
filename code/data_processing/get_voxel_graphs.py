import pickle
import sys
import glob
import yaml
from copy import deepcopy

with open('config.yaml', 'r') as config_file:  
  config = yaml.safe_load(config_file) 

with open(config['protein_config_file'], 'r') as config_file:  
  protein_config = yaml.safe_load(config_file)

protein_atom_symbols = protein_config['atom_symbols']
interaction_symbols = protein_config['interaction_symbols']

def get_distance(x,y):
    total = 0 

    for a,b in zip(x,y):
        total += (a-b)**2

    return total**(0.5)

pdb_dir = sorted(glob.glob(config['processed_pdbbind_dir'] + "*/"))
voxel_graph_dir = config['voxel_graph_dir']
target_id_list = []

for t_idx, target_dir in enumerate(pdb_dir):
    if t_idx % 100 == 0:
        print(t_idx)

    target_id = target_dir.split('/')[-2]
    target_id_list.append(target_id)
    protein_atom_data = []         

    with open("%s%s_protein_25.pdb" % (target_dir, target_id), 'r') as pdb_in:

        for line in pdb_in:
            line = line.rstrip()

            if line[:3].strip() == 'TER':
                break

            if line[:4].strip() != 'ATOM':
                continue
            
            atom_name = line[12:16].strip()       
            atom_element = line[76:78].strip()
            atom_xyz = [float(line[30:38].strip()), float(line[38:46].strip()), float(line[46:54].strip())]
            protein_atom_data.append([atom_name, atom_element, atom_xyz])

    interaction_profile = pickle.load(open("%s%s_ip.pkl" % (target_dir, target_id), 'rb'))

    for ip_type, coordinates in interaction_profile.items():
        for coords in coordinates:
            print(ip_type, coords)
            cutoff_index = 0
            heavy_atom_count = 0
            closest_protein_atoms = sorted(deepcopy(protein_atom_data), key=lambda atom_data: get_distance(coords, atom_data[2]))

            while heavy_atom_count <= 10:
                cutoff_index += 1

                if closest_protein_atoms[cutoff_index][1] != 'H':
                    heavy_atom_count += 1

            for thing in closest_protein_atoms[:cutoff_index]:
                print(thing, get_distance(coords, thing[2])) 

            print('------------')

    if t_idx == 5:
        sys.exit()











