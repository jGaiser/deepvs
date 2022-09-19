import sys
import numpy as np
import os
import glob
import yaml

batch_number = int(sys.argv[1])
batch_count = int(sys.argv[2])
PADDING = 1

with open('config.yaml', 'r') as config_file:  
  config = yaml.safe_load(config_file)

with open(config['ligand_stats_file'], 'r') as ligand_stats_file:  
  ligand_stats = yaml.safe_load(ligand_stats_file)

def get_distance(x,y):
    total = 0 

    for a,b in zip(x,y):
        total += (a-b)**2

    return total**(0.5)

input_dir = sorted(glob.glob(config['processed_pdbbind_dir'] + "/*/"))

#######
batch_size = int(len(input_dir) / batch_count)
batch_remainder = len(input_dir) % batch_count

if batch_number <= batch_remainder:
    batch_start_offset = batch_number-1
    batch_end_offset = batch_number 
else:
    batch_start_offset = batch_remainder
    batch_end_offset = batch_remainder

batch_start_index = (batch_number - 1) * batch_size + batch_start_offset
batch_end_index = (batch_number - 1) * batch_size + batch_size + batch_end_offset

if batch_number < batch_count:
    batch = input_dir[batch_start_index:batch_end_index]
else:
    batch = input_dir[batch_start_index:]
#######

for target_dir in batch:
    target_id = target_dir.split('/')[-2]
    protein_file = "%s%s_protein_25.pdb" % (target_dir, target_id)
    ligand_file = "%s%s_ligand.pdb" % (target_dir, target_id)

    ligand_atom_coords = []

    with open(ligand_file, 'r') as ligand_in:
        for line in ligand_in:
            line = line.rstrip()
    
            if line[:6].strip() not in ['HETATM', 'ATOM']:
                continue
    
            ligand_atom_coords.append([float(line[30:38].strip()), float(line[38:46].strip()), float(line[46:54].strip())])

    ligand_atom_coords = np.array(ligand_atom_coords)
    ligand_max = np.max(ligand_atom_coords, axis=0) 
    ligand_min = np.min(ligand_atom_coords, axis=0) 
    true_center = (ligand_min + ligand_max) / 2
    centermost_idx = np.argmin(np.linalg.norm(ligand_atom_coords-true_center, axis=1))
    centermost_atom_coord = ligand_atom_coords[centermost_idx]

    voxel_centers = []

    for i in np.arange(int(ligand_min[0])-PADDING, int(ligand_max[0]+1)+PADDING+1, 2):
        for j in np.arange(int(ligand_min[1])-PADDING, int(ligand_max[1]+1)+PADDING+1, 2):
            for k in np.arange(int(ligand_min[2])-PADDING, int(ligand_max[2]+1)+PADDING+1, 2):
                voxel_centers.append([i,j,k])

    protein_atom_coords = []
    with open(protein_file, 'r') as protein_in:
        for line in protein_in:
            line = line.rstrip()
    
            if line[:6].strip() not in ['ATOM']:
                continue
    
            protein_atom_coords.append([float(line[30:38].strip()), float(line[38:46].strip()), float(line[46:54].strip())])            

    voxel_file_content = ""

    for voxel in voxel_centers: 
        closest_protein_atom_distance = 999

        for protein_atom in protein_atom_coords:
            voxel_atom_distance = get_distance(voxel, protein_atom) 

            if voxel_atom_distance < closest_protein_atom_distance:
                closest_protein_atom_distance = voxel_atom_distance


        if closest_protein_atom_distance < ligand_stats['min_nearest_pl_atom_distance']:
            continue

        if closest_protein_atom_distance > ligand_stats['max_nearest_pl_atom_distance']-3:
            continue

        voxel_file_content += "%s %s %s\n" % tuple(voxel)

    with open("%s%s_2_1_0_3.vox" % (target_dir, target_id), 'w') as vox_file_out:
        vox_file_out.write(voxel_file_content)

    print(target_id)