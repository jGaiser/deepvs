import sys
import os
import glob
import re
import yaml
import pickle
from plip.structure.preparation import PDBComplex


with open('config.yaml', 'r') as config_file:  
  config = yaml.safe_load(config_file)

pdb_dir = sorted(glob.glob(config['processed_pdbbind_dir'] + "*/"))

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

# def get_plip_atom_index_dict(pdb_file):
#     idx_dict = {}

#     with open(pdb_file,'r') as pdb_in:
#         heavy_atom_index = 1 

#         for line in pdb_in:
#             parsed_line = re.split(r"\s+", line)
#             line_type = parsed_line[0]

#             if line_type in ['ATOM', 'HETATM']:
#                 if line[77] == 'H':
#                     continue

#                 # if parsed_line[3] == 'HOH':
#                 #     continue

#                 idx_dict[heavy_atom_index] = int(parsed_line[1])
#                 heavy_atom_index += 1

#     print(idx_dict.keys())
#     return idx_dict

# def get_molecular_graph_data(pdb_file):
#     content_string = ""
#     capture = False

#     with open(pdb_file, 'r') as pdb_in:
#         for line in pdb_in:
#             if line[:6] == 'HETATM':
#                 capture = True 

#             if capture == True:
#                 if line[:3] == 'END':
#                     capture = False
#                     break

#                 content_string += line
#                 continue

#     return content_string


def stringify_atom_idx(number, total_width):
    number = str(number)
    padding = total_width - len(number) 
    return " "*padding + number


for pdb_idx, target_dir in enumerate(pdb_dir):
    target_id = target_dir.split('/')[-2]
    print(pdb_idx, target_id)

    if target_id in ['readme', 'index']:
        continue

    protein_pdb = target_dir + "%s_protein_25.pdb" % target_id
    ligand_pdb = target_dir + "%s_ligand.pdb" % target_id
    complex_pdb = target_dir + "%s_complex.pdb" % target_id
    interaction_profile = target_dir + "%s_ip.pkl" % target_id

    complex_pdb_content = ""
    atom_idx = -1

    with open(protein_pdb, 'r') as protein_in:
        for line in protein_in:
            if line[:6].strip() in ['HETATM', 'ATOM']:
                atom_idx = int(line[6:11].strip())

            if line[:3] == 'END':
                continue

            complex_pdb_content += line 

    with open(ligand_pdb, 'r') as ligand_in:
        for line in ligand_in:
            if line[:6].strip() not in ['HETATM', 'ATOM']: 
                continue
            atom_idx += 1
            line = line[:6] + stringify_atom_idx(atom_idx, 5) + line[11:] 
            complex_pdb_content += line

    complex_pdb_content += "END\n"

    with open(complex_pdb, 'w') as complex_out:
        complex_out.write(complex_pdb_content)

    interaction_data = get_interaction_data(complex_pdb) 
    pickle.dump(interaction_data, open(interaction_profile, 'wb'))
    os.remove(complex_pdb)
