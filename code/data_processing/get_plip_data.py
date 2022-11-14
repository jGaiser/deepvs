import sys
import os
import glob
import re
import yaml
import pickle
from plip.structure.preparation import PDBComplex

with open('config.yaml', 'r') as config_file:  
  config = yaml.safe_load(config_file)

with open(config['protein_config_file'], 'r') as config_file:  
  protein_config = yaml.safe_load(config_file)

pdb_dir = sorted(glob.glob(config['processed_pdbbind_dir'] + "*/"))

##---------PROCESS BATCH
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
##---------PROCESS BATCH

def mean(l):
    return sum(l) / len(l)

def get_ligand_data(pl_interaction):
    ligand_interaction_data = {} 

    for interaction in pl_interaction.all_itypes:
        i_type = re.search(r".*\.(\S+)\'\>$", str(type(interaction))).group(1)

        if i_type == 'hbond':
            if interaction.protisdon == True:
                interaction_record = [i_type+"_a", interaction.a.coords]
            else:
                interaction_record = [i_type+"_d", interaction.h.coords]

        if i_type == 'hydroph_interaction':
            interaction_record = [i_type, interaction.ligatom.coords] 

        if i_type == 'halogenbond':
            interaction_record = [i_type, interaction.don.orig_x.coords] 

        if i_type == 'pistack':
            interaction_record = [i_type, tuple(interaction.ligandring.center)]

        if i_type == 'saltbridge':
            if interaction.protispos:
                interaction_record = ['saltbridge_n', tuple(interaction.negative.center)]
            else:
                interaction_record = ['saltbridge_p', tuple(interaction.positive.center)]

        if i_type == 'pication':
            if interaction.protcharged:
                interaction_record = [i_type + '_r', tuple(interaction.ring.center)]
            else:
                interaction_record = [i_type + '_c', tuple(interaction.charge.center)]

        if i_type in ['metal_complex', 'waterbridge']: 
            continue

        if interaction_record[0] not in ligand_interaction_data:
            ligand_interaction_data[interaction_record[0]] = []

        for coords in interaction_record[1:]:
            if coords not in ligand_interaction_data[interaction_record[0]]:
                ligand_interaction_data[interaction_record[0]].append(coords)

    return ligand_interaction_data 

def get_interaction_data(pdb_file):
    my_mol = PDBComplex()
    my_mol.load_pdb(pdb_file)
    my_mol.analyze()

    interaction_data = {}

    for object_ids, pl_interaction in my_mol.interaction_sets.items():
        plip_profile = get_ligand_data(pl_interaction)

        for k,v in plip_profile.items():
            if k not in interaction_data:
                interaction_data[k] = v
            else:
                for idx in v:
                    if idx not in interaction_data[k]:
                        interaction_data[k].append(idx)

    return interaction_data

def stringify_atom_idx(number, total_width):
    number = str(number)
    padding = total_width - len(number) 
    return " "*padding + number


for pdb_idx, target_dir in enumerate(pdb_dir):
    target_id = target_dir.split('/')[-2]
    if pdb_idx % 500 == 0:
        print(pdb_idx)
    # print(pdb_idx, target_id)

    if target_id in ['readme', 'index']:
        continue

    protein_pdb = target_dir + "%s_protein_25.pdb" % target_id
    ligand_pdb = target_dir + "%s_ligand.pdb" % target_id
    complex_pdb = target_dir + "%s_complex.pdb" % target_id
    interaction_profile = target_dir + "%s_ip.pkl" % target_id

    #DEBUG#
    if os.path.exists(interaction_profile) == False:
        print(target_id)
    #DEBUG#

    # complex_pdb_content = ""
    # atom_idx = -1

    # with open(protein_pdb, 'r') as protein_in:
    #     for line in protein_in:
    #         if line[:6].strip() in ['HETATM', 'ATOM']:
    #             atom_idx = int(line[6:11].strip())

    #         if line[:3] == 'END':
    #             continue

    #         complex_pdb_content += line 

    # with open(ligand_pdb, 'r') as ligand_in:
    #     for line in ligand_in:
    #         if line[:6].strip() not in ['HETATM', 'ATOM']: 
    #             continue
    #         atom_idx += 1
    #         line = line[:6] + stringify_atom_idx(atom_idx, 5) + line[11:] 
    #         complex_pdb_content += line

    # complex_pdb_content += "END\n"

    # with open(complex_pdb, 'w') as complex_out:
    #     complex_out.write(complex_pdb_content)

    # interaction_data = get_interaction_data(complex_pdb) 
    # pickle.dump(interaction_data, open(interaction_profile, 'wb'))
    # os.remove(complex_pdb)
