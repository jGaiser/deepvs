import sys
import numpy
import os
import glob
import yaml

with open('config.yaml', 'r') as config_file:  
  config = yaml.safe_load(config_file)

with open(config['ligand_stats_file'], 'r') as ligand_stats_file:  
  ligand_stats = yaml.safe_load(ligand_stats_file)

input_dir = glob.glob(config['processed_pdbbind_dir'] + "/*/")
stats_file = config['ligand_stats_file']

ii=0

for target_dir in input_dir:
  target_id = target_dir.split('/')[-2]

  if target_id in ['index', 'readme']:
    continue

  ligand_file = target_dir + target_id + "_ligand.pdb"

  max_xyz = [-9999,-9999,-9999]
  min_xyz = [9999,9999,9999]

  with open(ligand_file, 'r') as ligand_in:
    for line in ligand_in:
      line = line.rstrip()

      if line[:6].strip() not in ['HETATM', 'ATOM']:
        continue

      xyz_coords = [float(line[30:38].strip()), float(line[38:46].strip()), float(line[46:54].strip())]

      for coord_idx in range(3):
        if max_xyz[coord_idx] < xyz_coords[coord_idx]:
          max_xyz[coord_idx] = xyz_coords[coord_idx]

        if min_xyz[coord_idx] > xyz_coords[coord_idx]:
          min_xyz[coord_idx] = xyz_coords[coord_idx]         

  w = abs(max_xyz[0] - min_xyz[0])
  l = abs(max_xyz[1] - min_xyz[1])
  h = abs(max_xyz[2] - min_xyz[2])
  v = w*l*h
  d = (w**2 + l**2 + h**2)**0.5

  if ligand_stats['max_w'] < w:
    ligand_stats['max_w'] = w

  if ligand_stats['max_l'] < l:
    ligand_stats['max_l'] = l

  if ligand_stats['max_h'] < h:
    ligand_stats['max_h'] = h

  if ligand_stats['max_v'] < v:
    ligand_stats['max_v'] = v 

  if ligand_stats['max_d'] < d:
    ligand_stats['max_d'] = d

  ii+=1
  print(ii)

with open(config['ligand_stats_file'], 'w') as ligand_stats_file:  
  yaml.dump(ligand_stats, ligand_stats_file)
