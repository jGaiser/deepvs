import sys
import os
import glob
import yaml

with open('config.yaml', 'r') as config_file:  
  config = yaml.safe_load(config_file)

MAX_DISTANCE = 25

ligand_dir = sorted(glob.glob(config['processed_pdbbind_dir'] + "*/"))
protein_dir = config['pdbbind_dir']

def get_distance(x,y):
    total = 0 

    for a,b in zip(x,y):
        total += (a-b)**2

    return total**(0.5)


def stringify_atom_idx(number, total_width):
    number = str(number)
    padding = total_width - len(number) 
    return " "*padding + number


# ----- SLURM BATCHING ----- #

batch_number = int(sys.argv[1])
batch_count = int(sys.argv[2])

batch_size = int(len(ligand_dir) / batch_count)
batch_remainder = len(ligand_dir) % batch_count

if batch_number <= batch_remainder:
    batch_start_offset = batch_number-1
    batch_end_offset = batch_number 
else:
    batch_start_offset = batch_remainder
    batch_end_offset = batch_remainder

batch_start_index = (batch_number - 1) * batch_size + batch_start_offset
batch_end_index = (batch_number - 1) * batch_size + batch_size + batch_end_offset

if batch_number < batch_count:
    batch = ligand_dir[batch_start_index:batch_end_index]
else:
    batch = ligand_dir[batch_start_index:]

# -------------------------- #

for target_idx, target_dir in enumerate(batch):
    target_id = target_dir.split('/')[-2]

    print(target_idx, target_id)

    if target_id in ['index','readme']:
        continue

    ligand_file = target_dir + "%s_ligand.pdb" % target_id
    protein_file = protein_dir + "%s/%s_protein.pdb" % (target_id, target_id)
    processed_protein_file = target_dir + "%s_protein_%s.pdb" % (target_id, MAX_DISTANCE)
    processed_pdb_content = ""

    if(os.path.exists(processed_protein_file)):
        continue

    ligand_coords = []
    pdb_lines = []

    with open(ligand_file, 'r') as ligand_in:
        for line in ligand_in:
            line = line.rstrip()

            if line[:6].strip() not in ['HETATM', 'ATOM']:
                continue

            ligand_coords.append([float(line[30:38].strip()), float(line[38:46].strip()), float(line[46:54].strip())])

    with open(protein_file, 'r') as protein_in:
        atom_idx = 1
        

        for line in protein_in:
            line = line.rstrip()
            linetype = line[:6].strip()

            if linetype not in ['HETATM', 'ATOM', 'TER', 'END']:
                continue

            if linetype in ['HETATM', 'ATOM']:
                include_line = False
                protein_atom_xyz = [float(line[30:38].strip()), float(line[38:46].strip()), float(line[46:54].strip())]

                for ligand_atom_xyz in ligand_coords:
                    if get_distance(ligand_atom_xyz, protein_atom_xyz) < MAX_DISTANCE:
                        include_line = True
                        line = line[:6] + stringify_atom_idx(atom_idx, 5) + line[11:]
                        atom_idx += 1
                        break
            else:
                include_line = True

            if include_line == True:
                processed_pdb_content += line + "\n"

    with open(processed_protein_file, 'w') as processed_protein_out:
        processed_protein_out.write(processed_pdb_content)

