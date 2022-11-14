import sys
import os
import glob
import yaml

with open('config.yaml', 'r') as config_file:  
  config = yaml.safe_load(config_file)

obabel_command = "obabel -isdf %s -osmi > %s" 

output_dir = config['processed_pdbbind_dir']

input_dir = sorted(glob.glob(config['pdbbind_dir'] + "/*/"))

for dir_idx,input_target_dir in enumerate(input_dir):
	target_id = input_target_dir.split('/')[-2]

	if target_id in ['index', 'readme']:
		continue

	output_target_dir = output_dir + target_id + "/"

	input_ligand_file = input_target_dir + target_id + "_ligand.sdf"
	output_ligand_file = output_target_dir + target_id + "_ligand.smi"

	if os.path.exists(output_target_dir) == False:
		os.system("mkdir -p %s" % output_target_dir)

	if os.path.exists(output_ligand_file) == False:
		os.system(obabel_command % (input_ligand_file, output_ligand_file))

	if dir_idx % 500 == 0:
		print(dir_idx)