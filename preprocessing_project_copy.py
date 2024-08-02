import glob
import numpy as np
import pandas as pd
import subprocess
import os

import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm

import torch
from torch.nn import MSELoss
from sklearn.metrics import mean_squared_error

from adabelief_pytorch import AdaBelief
from sklearn.model_selection import train_test_split

import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from vision_mamba.model import Vim
import time
import time as time_module

from torch.nn import Linear, ReLU, Conv2d, MaxPool2d, Module
from torch.autograd import Variable

from sklearn.base import BaseEstimator, RegressorMixin
import json
from sklearn.model_selection import GridSearchCV

import optuna
import gc


def clean_df(df_path, pathological, affinity_entries_only=True):
    """
    Cleans the database containing the PDB entries.

    Parameters:
    - df_path: str
        Path to the database file.
    - pathological: list
        PDB identifiers of antibodies that need to be excluded.
    - affinity_entries_only: bool
        If True, only consider data with affinity values.

    Returns:
    - df_pdbs: list
        PDB entries.
    - df_kds: list
        Binding affinities.
    - df: pandas.DataFrame
        Cleaned database.
    """
    # Read the database
    df = pd.read_csv(df_path, sep='\t', header=0)[['pdb', 'antigen_type', 'affinity']]

    # Remove duplicates
    df.drop_duplicates(keep='first', subset='pdb', inplace=True)

    # Convert PDB identifiers to lowercase and remove '+' signs
    df['pdb'] = df['pdb'].str.lower().str.replace('+', '')

    # Filter entries based on antigen type
    df = df[(df.antigen_type.notna()) & (df.antigen_type != 'NA')][['pdb', 'affinity']]

    # If only entries with affinity are considered
    if affinity_entries_only:
        df = df[(df.affinity.notna()) & (df.affinity != 'None')]

    # Exclude pathological cases
    df = df[~df['pdb'].isin(pathological)]

    # df = df[:30]

    return list(df['pdb']), list(df['affinity']), df


def generate_fv_pdb(path, residues_path, stage='training', selection='_fv', affinity_entries_only=True, alphafold=False, ag_agnostic=False, cmaps=False,
                    keepABC=True, lresidues=False, hupsymchain=None, lupsymchain=None):
        r"""Generates a new PDB file going from the beginning of the CDR1 until the end of the CDR3.

        Parameters
        ----------
        path: str
            Path of a Chothia-numbered PDB file.
        keepABC: bool
            Keeps residues whose name ends with a letter from 'A' to 'Z'.
        lresidues: bool
            The names of each residue are stored in ``self.residues_path``.
        upsymchain: int
            Upper limit of heavy chain residues due to a change in the numbering convention. Only useful when using ``AlphaFold``.
        lupsymchain: int
            Upper limit of light chain residues due to a change in the numbering convention. Only useful when using ``AlphaFold``.

        """
        if stage == 'training':
            rpath = residues_path
        else:
            rpath = test_residues_path
        list_residues = ['START']

        with open(path, 'r') as f: # needs to be Chothia-numbered
            content = f.readlines()
            header_lines_important = range(4)
            header_lines = [content[i][0]=='R' for i in range(len(content))].count(True)
            h_range = range(1, 114)
            l_range = range(1, 108)
            start_chain = 21
            chain_range = slice(start_chain, start_chain+1)
            res_range = slice(23, 26)
            res_extra_letter = 26 #sometimes includes a letter 'A', 'B', 'C', ...
            h_chain_key = 'HCHAIN'
            l_chain_key = 'LCHAIN'
            antigen_chain_key = 'AGCHAIN'
            idx_list = list(header_lines_important)
            idx_list_l = []
            idx_list_antigen = []
            antigen_chains = []
            new_path = path[:-4] + selection + path[-4:]
            # Getting the names of the heavy and antigen chains
            line = content[header_lines_important[-1]]
            if line.find(h_chain_key) != -1:
                h_pos = line.find(h_chain_key) + len(h_chain_key) + 1
                h_chain = line[h_pos:h_pos+1]
                antigen_pos = line.find(antigen_chain_key) + len(antigen_chain_key) + 1
                antigen_chains.append(line[antigen_pos:antigen_pos+1])
                for i in range(3):
                    if line[antigen_pos+2*i+1] in [',', ';']:
                        antigen_chains.append(line[antigen_pos+2*i+2]) # If two (or more) interacting antigen chains present
            else:
                # useful when using AlphaFold
                h_chain = 'A'
                l_chain = 'B'
                antigen_chains = ['C', 'D', 'E']
                idx_list = [0]
                h_range = range(1-self.h_offset, hupsymchain-self.h_offset)
                l_range = range(1-self.l_offset, lupsymchain-self.l_offset)
                h_pos = start_chain
                l_pos = start_chain

            if line.find(l_chain_key) != -1:
                l_pos = line.find(l_chain_key) + len(l_chain_key) + 1
                l_chain = line[l_pos:l_pos+1]
            elif alphafold is False:
                l_chain = None

            # Checking if H and L chains have the same name
            if l_chain is not None and h_chain.upper() == l_chain.upper():
                pathologic = True
                h_chain = h_chain.upper()
                l_chain = h_chain.lower()
            elif antigen_chains is not None and affinity_entries_only is False and (h_chain.upper() in antigen_chains or (l_chain is not None and l_chain.upper() in antigen_chains)):
                pathologic = True
                h_chain = h_chain.lower()
                if l_chain is not None:
                    l_chain = l_chain.lower()
            else:
                pathologic = False

            # Checks for matching identifiers
            if pathologic:
                if 'X' not in antigen_chains:
                    new_hchain = 'X'
                else:
                    new_hchain = 'W'
                if 'Y' not in antigen_chains:
                    new_lchain = 'Y'
                else:
                    new_lchain = 'Z'
            else:
                new_hchain = h_chain
                new_lchain = l_chain

            # Obtaining lines for the heavy chain variable region first
            for i, line in enumerate(content[header_lines:]):
                if line[chain_range].find(h_chain) != -1 and int(line[res_range]) in h_range:
                    if (line[res_extra_letter] == ' ' or keepABC == True) and line.find('HETATM') == -1:
                        idx_list.append(i+header_lines)
                        if lresidues == True:
                            full_res = line[res_range] + line[res_extra_letter]
                            if pathologic:
                                full_res = new_hchain + full_res
                            else:
                                full_res = line[chain_range] + full_res
                            if full_res != list_residues[-1]:
                                list_residues.append(full_res)

            # This separation ensures that heavy chain residues are enlisted first
            if l_chain is not None:
                for i, line in enumerate(content[header_lines:]):
                    if line[chain_range].find(l_chain) != -1 and int(line[res_range]) in l_range:
                        if (line[res_extra_letter] == ' ' or keepABC == True) and line.find('HETATM') == -1:
                            idx_list_l.append(i+header_lines)
                            if lresidues == True:
                                full_res = line[res_range] + line[res_extra_letter]
                                if pathologic:
                                    full_res = new_lchain + full_res
                                else:
                                    full_res = line[chain_range] + full_res
                                if full_res != list_residues[-1]:
                                    list_residues.append(full_res)

            # Obtaining antigen(s)
            for i, line in enumerate(content[header_lines:]):
                if any(line[chain_range] in agc for agc in antigen_chains) and h_chain not in antigen_chains and l_chain not in antigen_chains:
                    idx_list_antigen.append(i+header_lines)

        # List with name of every residue is saved if selected
        if lresidues == True:
            list_residues.append('END')
            saving_path = rpath + path[-8:-4] + '.npy'
            #if not os.path.exists(saving_path):
            np.save(saving_path, list_residues)

        # Creating new file
        with open(new_path, 'w') as f_new:
            f_new.writelines([content[l] for l in idx_list[:header_lines_important[-1]]])
            if l_chain is not None and alphafold is False:
                f_new.writelines([content[l][:h_pos]+new_hchain+content[l][h_pos+1:l_pos]+new_lchain+content[l][l_pos+1:] for l in idx_list[header_lines_important[-1]:header_lines_important[-1]+1]])
            else:
                f_new.writelines([content[l][:h_pos]+new_hchain+content[l][h_pos+1:] for l in idx_list[header_lines_important[-1]:header_lines_important[-1]+1]])
            f_new.writelines([content[l][:start_chain-5]+' '+content[l][start_chain-4:start_chain]+new_hchain+content[l][start_chain+1:] for l in idx_list[header_lines_important[-1]+1:]])
            if l_chain is not None:
                f_new.writelines([content[l][:start_chain-5]+' '+content[l][start_chain-4:start_chain]+new_lchain+content[l][start_chain+1:] for l in idx_list_l])
            if not ag_agnostic:
                f_new.writelines([content[l] for l in idx_list_antigen])
            if not cmaps:
                f_new.writelines([content[l] for l in range(len(content)) if content[l][0:6] == 'HETATM' and content[l][chain_range] in [h_chain, l_chain] and l not in idx_list+idx_list_l+idx_list_antigen])


def generate_maps(entries, structures_path, residues_path, dccm_map_path, scripts_path, selection='_fv', file_type_input='.pdb', cmaps=False, modes=30,
                  cmaps_thr=8.0):
    """
    Generates the normal mode correlation maps.

    Parameters:
    - entries: list
        List of PDB entries to process.
    - selection: str
        Suffix to append to filenames indicating the selection of the structure.
    - structures_path: str
        Path to the directory containing structure files.
    - file_type_input: str
        File extension of the input structure files.
    - dccm_map_path: str
        Path where the generated maps should be saved.
    - scripts_path: str
        Path to the directory containing scripts for map generation.
    - cmaps: bool
        If True, use contact maps instead of normal mode correlation maps.
    - modes: int
        Number of modes to consider in the correlation map generation.
    - cmaps_thr: float
        Threshold for contact map generation.
    """
    for i, entry in enumerate(entries):
        file_name = entry + selection
        path = os.path.join(structures_path, file_name + file_type_input)
        new_path = os.path.join(dccm_map_path, entry)
        generate_fv_pdb(structures_path+entry+file_type_input, residues_path, lresidues=True)

        if not cmaps:
            # Call an external R script to generate normal mode correlation maps
            subprocess.call(['/usr/bin/Rscript', os.path.join(scripts_path, 'pdb_to_dccm.r'), path, new_path, str(modes)], shell=False, stdout=open(os.devnull, 'wb'))
        else:
            # Call a Python script to generate contact maps
            subprocess.call(['python', os.path.join(scripts_path, 'generate_contact_maps.py'), path, new_path, str(cmaps_thr)], stdout=open(os.devnull, 'wb'))

        if os.path.exists(path):
            os.remove(path)

        if i % 25 == 0:
            print('Map ' + str(i + 1) + ' out of ' + str(len(entries)) + ' processed.')


def get_lists_of_lengths(selected_entries, residues_path):
    """
    Retrieves lists with the lengths of the heavy and light chains.

    Parameters:
    - selected_entries: list
        List of PDB valid entries.
    - residues_path: str
        Path to the directory containing residue data files.

    Returns:
    - heavy: list
        Lengths of the heavy chains.
    - light: list
        Lengths of the light chains.
    - selected_entries: list
        PDB valid entries.
    """
    heavy = []
    light = []

    for entry in selected_entries:
        # Load the residue data for each entry
        list_of_residues = np.load(os.path.join(residues_path, entry + '.npy'))[1:-1]  # Exclude 'START' and 'END'
        h_chain = list_of_residues[0][0]  # Assume first residue chain label is heavy chain
        l_chain = list_of_residues[-1][0]  # Assume last residue chain label is light chain

        # Calculate the length of the heavy chain
        heavy_length = len([res for res in list_of_residues if res[0] == h_chain])
        heavy.append(heavy_length)

        # Calculate the length of the light chain if it exists and is different from the heavy chain
        if h_chain != l_chain:
            light_length = len([res for res in list_of_residues if res[0] == l_chain])
            light.append(light_length)
        else:
            light.append(0)  # No light chain or same as heavy chain

    return heavy, light, selected_entries


def get_max_min_chains(file_residues_paths, selected_entries, heavy, light):
        r"""Returns the longest and shortest possible chains.

        """
        max_res_list_h = []
        max_res_list_l = []

        for f in file_residues_paths:
            idx = selected_entries.index(f[-8:-4])
            current_list_h = np.load(f)[1:heavy[idx]+1]
            current_list_l = np.load(f)[heavy[idx]+1:heavy[idx]+light[idx]+1]
            current_list_h = [x[1:] for x in current_list_h]
            current_list_l = [x[1:] for x in current_list_l]
            max_res_list_h += list(set(current_list_h).difference(max_res_list_h))
            max_res_list_l += list(set(current_list_l).difference(max_res_list_l))

        max_res_list_h = sorted(max_res_list_h, key=remove_abc)
        min_res_list_h = list(dict.fromkeys([x for x in max_res_list_h]))
        max_res_list_h = [x.strip() for x in max_res_list_h]

        max_res_list_l = sorted(max_res_list_l, key=remove_abc)
        min_res_list_l = list(dict.fromkeys([x for x in max_res_list_l]))
        max_res_list_l = [x.strip() for x in max_res_list_l]

        for f in file_residues_paths:
            idx = selected_entries.index(f[-8:-4])
            current_list_h = np.load(f)[1:heavy[idx]+1]
            current_list_l = np.load(f)[heavy[idx]+1:heavy[idx]+light[idx]+1]
            current_list_h = [x[1:] for x in current_list_h]
            current_list_l = [x[1:] for x in current_list_l]
            min_res_list_h = sorted(list(set(current_list_h).intersection(min_res_list_h)))
            min_res_list_l = sorted(list(set(current_list_l).intersection(min_res_list_l)))

        min_res_list_h = [x.strip() for x in min_res_list_h]
        min_res_list_l = [x.strip() for x in min_res_list_l]

        return max_res_list_h, max_res_list_l, min_res_list_h, min_res_list_l


def remove_abc(residue):
    """
    Returns the residue names without the final letter that indicates extension positions.

    """
    if residue[-1] != ' ':
        residue = str(residue[:-1]) + '.' + '{0:0=2d}'.format(ord(residue[-1])-64)
    return float(residue)


def initialisation(entries, structures_path, dccm_map_path, scripts_path, residues_path, chain_lengths_path, renew_maps=False, renew_residues=True):
    """
    Computes the normal mode correlation maps and retrieves lists with the lengths of the heavy and light chains.

    Parameters:
    - renew_maps: bool
        Compute all the normal mode correlation maps.
    - renew_residues: bool
        Retrieve the lists of residues for each entry.
    - entries: list
        List of PDB entries.
    - selection: str
        Suffix to append to filenames indicating the selection of the structure.
    - structures_path: str
        Path to the directory containing structure files.
    - file_type_input: str
        File extension of the input structure files.
    - dccm_map_path: str
        Path where the generated maps should be saved.
    - scripts_path: str
        Path to the directory containing scripts for map generation.
    - cmaps: bool
        If True, use contact maps instead of normal mode correlation maps.
    - modes: int
        Number of modes to consider in the correlation map generation.
    - cmaps_thr: float
        Threshold for contact map generation.
    - residues_path: str
        Path to the directory containing residue data files.
    - chain_lengths_path: str
        Path to the directory where chain lengths data should be saved.

    Returns:
    - heavy: list
        Lengths of the heavy chains.
    - light: list
        Lengths of the light chains.
    - selected_entries: list
        PDB valid entries.
    """
    if renew_maps:
        generate_maps(entries, structures_path, residues_path, dccm_map_path, scripts_path, cmaps=True)

    dccm_paths = sorted(glob.glob(os.path.join(dccm_map_path, '*.npy')))
    selected_entries = [dccm_paths[i][-8:-4] for i in range(len(dccm_paths))]

    if renew_residues:
        heavy, light, selected_entries = get_lists_of_lengths(selected_entries, residues_path)
        np.save(os.path.join(chain_lengths_path, 'heavy_lengths.npy'), heavy)
        np.save(os.path.join(chain_lengths_path, 'light_lengths.npy'), light)
        np.save(os.path.join(chain_lengths_path, 'selected_entries.npy'), selected_entries)
    else:
        heavy = np.load(os.path.join(chain_lengths_path, 'heavy_lengths.npy')).astype(int)
        light = np.load(os.path.join(chain_lengths_path, 'light_lengths.npy')).astype(int)
        selected_entries = list(np.load(os.path.join(chain_lengths_path, 'selected_entries.npy')))

    assert len(selected_entries) == len(heavy) == len(light), "Mismatch in lengths of entries and chain lengths."

    return heavy, light, selected_entries


def generate_masked_image(img, idx, file_residues_paths, selected_entries, max_res_list_h, max_res_list_l, heavy, light, stage='training', test_h=None,
                          test_l=None, alphafold=False, test_residues_path=None, test_pdb_id='1t66', ag_residues=0):
        r"""Generates a masked normal mode correlation map

        Parameters
        ----------
        img: numpy.ndarray
            Original array containing no blank pixels.
        idx: int
            Input index.
        test_h: int
            Length of the heavy chain of an antibody in the test set.
        test_l: int
            Length of the light chain of an antibody in the test set.

        Returns
        -------
        masked: numpy.ndarray
            Masked normal mode correlation map.
        mask: numpy.ndarray
            Mask itself.

        """
        if stage == 'training':
            f = file_residues_paths[idx]
        elif alphafold is False:
            f = sorted(glob.glob(os.path.join(test_residues_path, '*'+test_pdb_id+'.npy')))[0]
        else:
            f = sorted(glob.glob(os.path.join(test_residues_path, '*'+test_pdb_id[:-3]+'.npy')))[0] # removing '_af' suffix
        antigen_max_pixels = ag_residues
        f_res = np.load(f)
        max_res_h = len(max_res_list_h)
        max_res_l = len(max_res_list_l)
        max_res = max_res_h + max_res_l
        masked = np.zeros((max_res+antigen_max_pixels, max_res+antigen_max_pixels))
        mask = np.zeros((max_res+antigen_max_pixels, max_res+antigen_max_pixels))

        if stage != 'training':
            h = test_h
            l = test_l
        else:
            current_idx = selected_entries.index(f[-8:-4])
            h = heavy[current_idx]
            l = light[current_idx]

        current_list_h = f_res[1:h+1]
        current_list_h = [x[1:].strip() for x in current_list_h]
        current_list_l = f_res[h+1:h+l+1]
        current_list_l = [x[1:].strip() for x in current_list_l]

        idx_list = [i for i in range(max_res_h) if max_res_list_h[i] in current_list_h]
        idx_list += [i+max_res_h for i in range(max_res_l) if max_res_list_l[i] in current_list_l]
        idx_list += [i+max_res_h+max_res_l for i in range(min(antigen_max_pixels, img.shape[-1]-(h+l)))]
        for k, i in enumerate(idx_list):
            for l, j in enumerate(idx_list):
                masked[i, j] = img[k, l]
                mask[i, j] = 1

        return masked, mask


def load_training_images(dccm_map_path, selected_entries, pathological, entries, affinity, df,
                         file_residues_paths, max_res_list_h, max_res_list_l, heavy, light, affinity_entries_only=True):
    """
    Returns the input/output pairs of the model and their corresponding labels.

    Parameters:
    - dccm_map_path: str
        Path where the DCCM maps are stored.
    - selected_entries: list
        List of selected PDB entries.
    - pathological: list
        List of entries that are considered pathological and should be excluded.
    - entries: list
        List of all entries.
    - affinity_entries_only: bool
        Flag to determine if only entries with affinity data should be processed.
    - affinity: list
        List of affinity values corresponding to the entries.
    - df: DataFrame
        DataFrame containing affinity data for validation.

    Returns:
    - imgs: numpy.ndarray
        Array of masked images.
    - kds: numpy.ndarray
        Array of log-transformed affinity values.
    - labels: list
        List of PDB IDs used.
    - raw_imgs: list
        List of raw images before masking.
    """
    imgs = []
    raw_imgs = []
    kds = []
    labels = []
    file_paths = sorted(glob.glob(os.path.join(dccm_map_path, '*.npy')))

    for f in file_paths:
        pdb_id = f[-8:-4]
        if pdb_id in selected_entries and pdb_id not in pathological:
            raw_sample = np.load(f)
            idx = entries.index(pdb_id)
            idx_new = selected_entries.index(pdb_id)
            labels.append(pdb_id)
            raw_imgs.append(raw_sample)
            imgs.append(generate_masked_image(raw_sample, idx_new, file_residues_paths, selected_entries, max_res_list_h, max_res_list_l, heavy, light)[0])
            if affinity_entries_only:
                kds.append(np.log10(np.float32(affinity[idx])))

    assert labels == [item for item in selected_entries if item not in pathological]

    # for pdb in selected_entries:
    #     if pdb not in pathological and affinity_entries_only:
    #         assert np.float16(10**kds[[item for item in selected_entries if item not in pathological].index(pdb)] == np.float16(df[df['pdb']==pdb]['affinity'])).all()

    for pdb in selected_entries:
        if pdb not in pathological and affinity_entries_only:
            idx = [item for item in selected_entries if item not in pathological].index(pdb)
            calculated_kd = np.float32(10**kds[idx])
            actual_kd = np.float32(df[df['pdb'] == pdb]['affinity'].values[0])
            assert np.isclose(calculated_kd, actual_kd, atol=1e-5), f"Mismatch in KD values for {pdb}: calculated {calculated_kd}, actual {actual_kd}"


    return np.array(imgs), np.array(kds), labels, raw_imgs


def create_test_set(train_x, train_y, test_size=None, random_state=0):
    r"""Creates the test set given a set of input images and their corresponding labels.

    Parameters
    ----------
    train_x: numpy.ndarray
        Input normal mode correlation maps.
    train_y: numpy.ndarray
        Labels.
    test_size: float
        Fraction of original samples to be included in the test set.
    random_state: int
        Set lot number.

    Returns
    -------
    train_x: torch.Tensor
        Training inputs.
    test_x: torch.Tensor
        Test inputs.
    train_y: torch.Tensor
        Training labels.
    test_y: torch.Tensor
        Test labels.

    """

    # Splitting
    indices = np.arange(len(train_x))
    train_x, test_x, train_y, test_y, indices_train, indices_test = train_test_split(train_x, train_y, indices, test_size=0.023, random_state=23)

    # Converting to tensors
    train_x = train_x.reshape(train_x.shape[0], 1, train_x.shape[1], train_x.shape[1])
    train_x = train_x.astype(np.float32)
    train_x  = torch.from_numpy(train_x)
    train_y = train_y.astype(np.float32).reshape(train_y.shape[0], 1)
    train_y = torch.from_numpy(train_y)

    test_x = test_x.reshape(test_x.shape[0], 1, test_x.shape[2], test_x.shape[2])
    test_x = test_x.astype(np.float32)
    test_x  = torch.from_numpy(test_x)
    test_y = test_y.astype(np.float32).reshape(test_y.shape[0], 1, 1)
    test_y = torch.from_numpy(test_y)

    return train_x, test_x, train_y, test_y, indices_train, indices_test


def training_vim(train_x, train_y):
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Assuming train_x and train_y are your input arrays
    train_x_t = torch.tensor(train_x, dtype=torch.float32).to(device)
    train_y_t = torch.tensor(train_y, dtype=torch.float32).to(device)

    # Create a TensorDataset and DataLoader
    dataset = TensorDataset(train_x_t, train_y_t)
    train_loader = DataLoader(dataset, batch_size=20, shuffle=True)

    # Initialize the Vim model
    model = Vim(
        dim=128,
        dt_rank=32,
        dim_inner=128,
        d_state=97,
        num_classes=1,  # For regression, typically the output is a single value per instance
        image_size=286,
        patch_size=13,
        channels=1,
        dropout=0.2677301595791723,
        depth=7,
    )

    # Move the model to the GPU
    model.to(device)

    # Using Mean Squared Error Loss for a regression task
    criterion = MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00015334092031733988)

    # Training loop
    model.train()  # Set the model to training mode
    num_epochs = 400  # Define the number of epochs
    verbose = True  # Set verbose to True to print correlation

    # Initialize lists to store the loss and correlation values for each epoch
    loss_values = []
    correlation_values = []

    # Record the start time
    start_time = time.time()

    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        outputs_all = []
        targets_all = []

        for batch_inputs, batch_targets in train_loader:
            # Move the inputs and targets to the GPU
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)

            optimizer.zero_grad()  # Zero the parameter gradients

            # Forward pass
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Accumulate loss
            total_loss += loss.item()
            num_batches += 1

            # Debugging shapes
            print("Output shape:", outputs.shape)
            print("Target shape:", batch_targets.shape)

            # Collect outputs and targets for correlation, ensure they are flattened
            outputs_all.append(outputs.view(-1).detach().cpu().numpy())
            targets_all.append(batch_targets.view(-1).detach().cpu().numpy())

        # Calculate average loss for the epoch
        average_loss = total_loss / num_batches
        print(f'Epoch {epoch + 1}: Average Loss {average_loss:.4f}')

        # Compute correlation
        outputs_flat = np.concatenate(outputs_all)
        targets_flat = np.concatenate(targets_all)
        corr = np.corrcoef(outputs_flat, targets_flat)[0, 1]
        if verbose:
            print('Epoch {}: Correlation: {:.4f}'.format(epoch + 1, corr))

        # Append loss and correlation values to the lists
        loss_values.append(average_loss)
        correlation_values.append(corr)

    # Record the end time
    end_time = time.time()

    # Calculate and print the total training time
    total_training_time = end_time - start_time
    print(f'Total Training Time: {total_training_time:.2f} seconds')

    # Save the trained model
    model_save_path = 'vim_model.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')

    return model, loss_values, correlation_values, num_epochs, total_training_time


def plot_vim(loss_values, correlation_values, num_epochs, time, name):
    # Plotting loss and correlation
    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), loss_values, label='Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot correlation
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), correlation_values, label='Correlation', color='orange')
    plt.title('R = '+str(correlation_values[-1]))
    plt.xlabel('Epoch')
    plt.ylabel('Correlation')
    plt.legend()

    # Add time annotation
    plt.text(0.05, 0.95, f'time = {time:.2f} s', transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top')

    # Save the plot to a file
    plt.savefig(name+'.png')
    plt.close()  # Close the figure to free up memory


# Hyperparameter search
def objective(trial, train_x, train_y):
    # Define the hyperparameters to tune
    dim = trial.suggest_int('dim', 64, 128)
    d_state = trial.suggest_int('d_state', 64, 128)
    depth = trial.suggest_int('depth', 4, 12)
    dropout = trial.suggest_categorical('dropout', [0.1, 0.3, 0.5])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
    # weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)

    # Initialize the Vim model
    model = Vim(
        dim=dim,
        dt_rank=32,
        dim_inner=dim,
        d_state=d_state,
        num_classes=1,  # For regression, typically the output is a single value per instance
        image_size=286,
        patch_size=13,
        channels=1,
        dropout=dropout,
        depth=depth,
    )

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Assuming train_x and train_y are your input arrays
    train_x_t = torch.tensor(train_x, dtype=torch.float32).to(device)
    train_y_t = torch.tensor(train_y, dtype=torch.float32).to(device)

    # Create a TensorDataset and DataLoader
    dataset = TensorDataset(train_x_t, train_y_t)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    # Using Mean Squared Error Loss for a regression task
    criterion = MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    # Training loop
    model.train()  # Set the model to training mode
    num_epochs = 10  # Reduce the number of epochs for faster hyperparameter search

    total_loss = 0.0
    num_batches = 0
    outputs_all = []
    targets_all = []

    for epoch in range(num_epochs):
        for batch_inputs, batch_targets in train_loader:
            # Move the inputs and targets to the GPU
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)

            optimizer.zero_grad()  # Zero the parameter gradients

            with torch.cuda.amp.autocast():
                # Forward pass
                outputs = model(batch_inputs)
                loss = criterion(outputs, batch_targets)

            # Backward pass and optimize
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Accumulate loss
            total_loss += loss.item() 
            num_batches += 1


    # Calculate average loss for the last epoch
    average_loss = total_loss / num_batches
    return average_loss


def test_vim(model, test_x, test_y):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_x_t = torch.tensor(test_x, dtype=torch.float32).to(device)
    test_y_t = torch.tensor(test_y, dtype=torch.float32).to(device)

    dataset = TensorDataset(test_x_t, test_y_t)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model.eval()  # Set the model to evaluation mode

    outputs_all = []
    targets_all = []

    with torch.no_grad():
        for batch_inputs, batch_targets in test_loader:
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)

            outputs = model(batch_inputs)

            outputs_all.append(outputs.view(-1).detach().cpu().numpy())
            targets_all.append(batch_targets.view(-1).detach().cpu().numpy())

    outputs_flat = np.concatenate(outputs_all)
    targets_flat = np.concatenate(targets_all)
    mse = mean_squared_error(targets_flat, outputs_flat)
    corr = np.corrcoef(outputs_flat, targets_flat)[0, 1]
    print(f'Test MSE: {mse:.4f}')
    print(f'Test Correlation: {corr:.4f}')

    return mse, corr, outputs_flat, targets_flat


def plot_test_results(outputs, targets, corr, mse, name):
    plt.figure(figsize=(10, 6))
    plt.scatter(targets, outputs, alpha=0.5)
    plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'k--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('R = '+str(corr))

    # Add MSE annotation
    plt.text(0.05, 0.95, f'MSE = {mse:.2f}', transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top')
    
    # Save the plot to a file
    plt.savefig(name + '.png')
    plt.close()  # Close the figure to free up memory


# def plot_vim_combined(loss_values_train, loss_values_val, correlation_values_train, num_epochs, name):
#     # Plotting loss and correlation
#     plt.figure(figsize=(12, 5))

#     # Plot training and validation loss
#     plt.subplot(1, 2, 1)
#     plt.plot(range(1, num_epochs + 1), loss_values_train, label='Training Loss')
#     plt.plot(range(1, num_epochs + 1), loss_values_val, label='Validation Loss')
#     plt.title('Loss over Epochs', size=20)
#     plt.xlabel('Epoch', size=14)
#     plt.ylabel('Loss', size=14)
#     plt.legend(prop={'size': 14})

#     # Plot training and validation correlation
#     plt.subplot(1, 2, 2)
#     plt.plot(range(1, num_epochs + 1), correlation_values_train, label='Training Correlation', color='blue')
#     # plt.plot(range(1, num_epochs + 1), correlation_values_val, label='Validation Correlation', color='orange')
#     plt.title('Train R = '+str(correlation_values_train[-1]), size=20)
#     plt.xlabel('Epoch', size=14)
#     plt.ylabel('Correlation', size=14)
#     plt.legend(prop={'size': 14})

#     # Save the plot to a file
#     plt.savefig(name + '.png')
#     plt.close()  # Close the figure to free up memory



class ANTIPASTI(Module):
    r"""Predicting the binding affinity of an antibody from its normal mode correlation map.

    Parameters
    ----------
    n_filters: int
        Number of filters in the convolutional layer.
    filter_size: int
        Size of filters in the convolutional layer.
    pooling_size: int
        Size of the max pooling operation.
    input_shape: int
        Shape of the normal mode correlation maps.
    l1_lambda: float
        Weight of L1 regularisation.
    mode: str
        To use the full model, provide ``full``. Otherwise, ANTIPASTI corresponds to a linear map.

    """
    def __init__(
            self,
            n_filters=2,
            filter_size=4,
            pooling_size=1,
            input_shape=281,
            l1_lambda=0.002,
            mode='full',
    ):
        super(ANTIPASTI, self).__init__()
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.pooling_size = pooling_size
        self.input_shape = input_shape
        self.mode = mode
        if self.mode == 'full':
            self.fully_connected_input = n_filters * ((input_shape-filter_size+1)//pooling_size) ** 2
            self.conv1 = Conv2d(1, n_filters, filter_size)
            self.pool = MaxPool2d((pooling_size, pooling_size))
            self.relu = ReLU()
        else:
            self.fully_connected_input = self.input_shape ** 2
        self.fc1 = Linear(self.fully_connected_input, 1, bias=False)
        self.l1_lambda = l1_lambda

    def forward(self, x):
        r"""Model's forward pass.

        Returns
        -------
        output: torch.Tensor
            Predicted binding affinity.
        inter_filter: torch.Tensor
            Filters before the fully-connected layer.

        """
        inter = x
        if self.mode == 'full':
            x = self.conv1(x) + torch.transpose(self.conv1(x), 2, 3)
            x = self.relu(x)
            inter = x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x.float(), inter

    def l1_regularization_loss(self):
        l1_loss = torch.tensor(0.0)
        for param in self.parameters():
            l1_loss += torch.norm(param, p=1)
        return self.l1_lambda * l1_loss
    
def training_step(model, criterion, optimiser, train_x, test_x, train_y, test_y, train_losses, test_losses, epoch, batch_size, verbose):
    r"""Performs a training step.

    Parameters
    ----------
    model: antipasti.model.model.ANTIPASTI
        The model class, i.e., ``ANTIPASTI``.
    criterion: torch.nn.modules.loss.MSELoss
        It calculates a gradient according to a selected loss function, i.e., ``MSELoss``.
    optimiser: adabelief_pytorch.AdaBelief.AdaBelief
        Method that implements an optimisation algorithm.
    train_x: torch.Tensor
        Training normal mode correlation maps.
    test_x: torch.Tensor
        Test normal mode correlation maps.
    train_y: torch.Tensor
        Training labels.
    test_y: torch.Tensor
        Test labels.
    train_losses: list
        The current history of training losses.
    test_losses: list
        The current history of test losses.
    epoch: int
        Of value ``e`` if the dataset has gone through the model ``e`` times.
    batch_size: int
        Number of samples that pass through the model before its parameters are updated.
    verbose: bool
        ``True`` to print the losses in each epoch.

    Returns
    -------
    train_losses: list
        The history of training losses after the training step.
    test_losses: list
        The history of test losses after the training step.
    inter_filter: torch.Tensor
        Filters before the fully-connected layer.
    y_test: torch.Tensor
        Ground truth test labels.
    output_test: torch.Tensor
        The predicted test labels.

    """
    tr_loss = 0
    tr_mse = 0
    x_train, y_train = Variable(train_x), Variable(train_y)
    x_test, y_test = Variable(test_x), Variable(test_y)

    # Filters before the fully-connected layer
    size_inter = int(np.sqrt(model.fully_connected_input/model.n_filters))
    inter_filter = np.zeros((x_train.size()[0], model.n_filters, size_inter, size_inter))
    if model.mode != 'full':
        inter_filter = np.zeros((x_train.size()[0], 1, model.input_shape, model.input_shape))
    permutation = torch.randperm(x_train.size()[0])

    for i in range(0, x_train.size()[0], batch_size):
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = x_train[indices], y_train[indices]

        # Training output
        output_train, inter_filters = model(batch_x)

        # Picking the appropriate filters before the fully-connected layer
        inter_filters_detached = inter_filters.detach().clone()
        inter_filter[i:i+batch_size] = inter_filters_detached.numpy()

        # Training loss, clearing gradients and updating weights
        optimiser.zero_grad()
        l1_loss = model.l1_regularization_loss()
        mse_loss = criterion(output_train[:, 0], batch_y[:, 0])
        loss_train = mse_loss + l1_loss
        if verbose:
            print(l1_loss)
        loss_train.backward()
        optimiser.step()
        # Adding batch contribution to training loss
        tr_loss += loss_train.item() * batch_size / x_train.size()[0]
        tr_mse += mse_loss * batch_size / x_train.size()[0]

    train_losses.append(tr_loss)
    loss_test = 0
    output_test = []

    with torch.no_grad():
        for i in range(x_test.size()[0]):
            optimiser.zero_grad()
            output_t, _ = model(x_test[i].reshape(1, 1, model.input_shape, model.input_shape))
            l1_loss = model.l1_regularization_loss()
            loss_t = criterion(output_t[:, 0], y_test[i][:, 0])
            loss_test += loss_t.item() / x_test.size()[0]
            if verbose:
                print(output_t)
                print(y_test[i])
                print('------------------------')
            output_test.append(output_t[:,0].detach().numpy())
    test_losses.append(loss_test)

    # Training and test losses
    if verbose:
        print('Epoch : ', epoch+1, '\t', 'train loss: ', tr_loss, 'train MSE: ', tr_mse, 'test MSE: ', loss_test)


    return train_losses, test_losses, inter_filter, y_test, output_test

def training_routine(model, criterion, optimiser, train_x, test_x, train_y, test_y, n_max_epochs=120, max_corr=0.87, batch_size=32, verbose=True):
    r"""Performs a chosen number of training steps.

    Parameters
    ----------
    model: antipasti.model.model.ANTIPASTI
        The model class, i.e., ``ANTIPASTI``.
    criterion: torch.nn.modules.loss.MSELoss
        It calculates a gradient according to a selected loss function, i.e., ``MSELoss``.
    optimiser: adabelief_pytorch.AdaBelief.AdaBelief
        Method that implements an optimisation algorithm.
    train_x: torch.Tensor
        Training normal mode correlation maps.
    test_x: torch.Tensor
        Test normal mode correlation maps.
    train_y: torch.Tensor
        Training labels.
    test_y: torch.Tensor
        Test labels.
    n_max_epochs: int
        Number of times the whole dataset goes through the model.
    max_corr: float
        If the correlation coefficient exceeds this value, the training routine is terminated.
    batch_size: int
        Number of samples that pass through the model before its parameters are updated.
    verbose: bool
        ``True`` to print the losses in each epoch.

    Returns
    -------
    train_losses: list
        The history of training losses after the training routine.
    test_losses: list
        The history of test losses after the training routine.
    inter_filter: torch.Tensor
        Filters before the fully-connected layer.
    y_test: torch.Tensor
        Ground truth test labels.
    output_test: torch.Tensor
        The predicted test labels.

    """
    train_losses = []
    test_losses = []

    for epoch in range(n_max_epochs):
        train_losses, test_losses, inter_filter, y_test, output_test = training_step(model, criterion, optimiser, train_x, test_x, train_y, test_y, train_losses, test_losses, epoch, batch_size, verbose)

        # Computing and printing the correlation coefficient
        corr = np.corrcoef(np.array(output_test).T, y_test[:,0].detach().numpy().T)[1,0]
        if verbose:
            print('Corr: ' + str(corr))
        if corr > max_corr:
            break

    return train_losses, test_losses, inter_filter, y_test, output_test

def plot_r_cnn(output_test, y_test, name):
    font_size = 14
    title_size = 20


    fig = plt.figure(figsize=(10, 8))
    plt.scatter(np.array(output_test), y_test[:,0].detach().numpy())
    corr = np.corrcoef(np.array(output_test).T, y_test[:,0].detach().numpy().T)[1,0]
    plt.plot([-11,-4],[-11,-4], c='r', linestyle='dashed')
    plt.title('R = '+str(corr), size=title_size)
    plt.xlabel('Predicted $log_{10}$($K_d$)', size=font_size)
    plt.ylabel('True $log_{10}$($K_d$)', size=font_size)
    # plt.show()

    # Save the plot to a file
    plt.savefig(name+'.png')
    plt.close()  # Close the figure to free up memory

def plot_loss_cnn(test_losses, train_losses, time, name):
    font_size = 14
    title_size = 20

    fig = plt.figure(figsize=(10, 8))
    plt.plot([test_losses[:][i] for i in range(len(test_losses[:]))])
    plt.plot([train_losses[:][i] for i in range(len(train_losses[:]))])
    plt.title('Loss', size=title_size)
    plt.xlabel('Number of epoch', size=font_size)
    plt.ylabel('MSE', size=font_size)
    plt.legend(['Test', 'Training'], prop={'size': font_size})
    # plt.show()

    # Add time annotation
    plt.text(0.05, 0.95, f'time = {time:.2f} s', transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top')

    # Save the plot to a file
    plt.savefig(name+'.png')
    plt.close()  # Close the figure to free up memory

def main():
    # Get the current working directory
    current_dir = os.getcwd()

    df_path = os.path.join(current_dir, 'notebooks', 'test_data', 'sabdab_summary_all.tsv')
    residues_path = os.path.join(current_dir, 'notebooks', 'test_data', 'lists_of_residues')
    structures_path = os.path.join(current_dir, 'all_structures', 'chothia')
    dccm_map_path = os.path.join(current_dir, 'notebooks', 'test_data', 'dccm_maps')
    scripts_path = os.path.join(current_dir, 'scripts')
    file_residues_paths = sorted(glob.glob(os.path.join(residues_path, '*.npy')))
    chain_lengths_path = os.path.join(current_dir, 'notebooks', 'test_data', 'chain_lengths')
    
    # Data to exclude
    pathological = ['5omm', '5i5k', '1uwx', '1mj7', '1qfw', '1qyg', '4ffz', '3ifl', '3lrh', '3pp4', '3ru8', '3t0w', '3t0x', '4fqr', '4gxu', '4jfx', '4k3h', '4jfz', '4jg0', '4jg1', '4jn2', '4o4y', '4qxt', '4r3s', '4w6y', '4w6y', '5ies', '5ivn', '5j57', '5kvd', '5kzp', '5mes', '5nmv', '5sy8', '5t29', '5t5b', '5vag', '3etb', '3gkz', '3uze', '3uzq', '4f9l', '4gqp', '4r2g', '5c6t', '3fku', '1oau', '1oay']
    scfv = ['4gqp', '3etb', '3gkz', '3uze', '3uzq', '3gm0', '4f9l', '6ejg', '6ejm', '1h8s', '5dfw', '6cbp', '4f9p', '5kov', '1dzb', '5j74', '5aaw', '3uzv', '5aam', '3ux9', '5a2j', '5a2k', '5a2i', '3fku', '5yy4', '3uyp', '5jyl', '1y0l', '1p4b', '3kdm', '4lar', '4ffy', '2ybr', '1mfa', '5xj3', '5xj4', '4kv5', '5vyf']
    pathological += scfv

    # Clean the dataframe
    entries, affinity, df = clean_df(df_path, pathological)

    # Generate the maps
    # generate_maps(entries, structures_path, residues_path, dccm_map_path, scripts_path, cmaps=False)

    # Initialise the heavy and light chains
    heavy, light, selected_entries = initialisation(entries, structures_path, dccm_map_path, scripts_path, residues_path,
                                                    chain_lengths_path)

    max_res_list_h, max_res_list_l, min_res_list_h, min_res_list_l = get_max_min_chains(file_residues_paths, selected_entries, heavy, light)

    # Load the training images
    train_x, train_y, labels, raw_imgs = load_training_images(dccm_map_path, selected_entries, pathological, entries, affinity, df,
                                                            file_residues_paths, max_res_list_h, max_res_list_l, heavy, light)

    # Create the test set
    train_x, test_x, train_y, test_y, idx_tr, idx_te = create_test_set(train_x, train_y, test_size=0.05)


    # Training VIM with fix parameters
    # loss_values, correlation_values, num_epochs = training_vim(train_x, train_y)

    # plot_vim(loss_values, correlation_values, num_epochs, name='training_performance_vim_50_epoch')

    


    # # Set CUDA_LAUNCH_BLOCKING to help with debugging
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


    # # Hyperparameter search using optuna
    # # Running the hyperparameter search
    # study = optuna.create_study(direction='minimize')
    # study.optimize(lambda trial: objective(trial, train_x, train_y), n_trials=30)  # Number of trials

    # # Get the best hyperparameters
    # best_params = study.best_params
    # print("Best hyperparameters:", best_params)

    # # Save the best hyperparameters to a JSON file
    # with open('best_hyperparameters.json', 'w') as f:
    #     json.dump(best_params, f, indent=4)

    # print("Best hyperparameters saved to best_hyperparameters.json")

    # # Initialize the best model
    # best_model = Vim(
    #     dim=best_params['dim'],
    #     # heads=8,
    #     dt_rank=32,
    #     dim_inner=best_params['dim'],
    #     d_state=best_params['d_state'],
    #     num_classes=1,  # For regression, typically the output is a single value per instance
    #     image_size=286,
    #     patch_size=13,
    #     channels=1,
    #     dropout=best_params['dropout'],
    #     depth=best_params['depth'],
    # )

    # # Print model architecture to a file
    # model_architecture_file = 'model_architecture.txt'
    # with open(model_architecture_file, 'w') as f:
    #     print(best_model, file=f)

    # print(f"Model architecture saved to {model_architecture_file}")

    # # Train the best model (use the objective function or similar training code)
    # # Assuming you have trained the best model here

    # # Save the best model
    # model_save_path = 'best_model.pth'
    # torch.save(best_model.state_dict(), model_save_path)
    # print(f"Best model saved to {model_save_path}")



    # Training CNN with fix parameters
    n_filters = 4
    filter_size = 4
    pooling_size = 2
    learning_rate = 1e-4
    input_shape = train_x.shape[-1]

    # Defining the model, optimiser and loss function
    model = ANTIPASTI(n_filters=n_filters, filter_size=filter_size, pooling_size=pooling_size, input_shape=input_shape, l1_lambda=0.002)
    criterion = MSELoss()
    optimiser = AdaBelief(model.parameters(), lr=learning_rate, weight_decay=False, eps=1e-8, print_change_log=False)
    # optimiser = Adam(model.parameters(), lr=learning_rate)
    print(model)

    train_losses = []
    test_losses = []

    model.train()
    n_max_epochs = 400 # This is just a super short example. You can increase this.
    max_corr = 0.95
    batch_size = 32

    start_time = time_module.time()  # Start timing the training process
    train_loss, test_loss, inter_filter, y_test, output_test = training_routine(model, criterion, optimiser, train_x, test_x, train_y, test_y, n_max_epochs=n_max_epochs, max_corr=max_corr, batch_size=batch_size)
    end_time = time_module.time()  # End timing the training process

    # Calculate and print the total training time
    total_training_time = end_time - start_time
    print(f'Total Training Time: {total_training_time:.2f} seconds')

    # Saving the losses
    train_losses.extend(train_loss)
    test_losses.extend(test_loss)

    # Save the trained model
    model_save_path = 'cnn_model.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')

    plot_r_cnn(output_test, y_test, name='R_training_performance_cnn_400_epoch')
    plot_loss_cnn(test_losses, train_losses, total_training_time, name='loss_training_performance_cnn_400_epoch')


    model, loss_values, correlation_values, num_epochs, time = training_vim(train_x, train_y)
    test_mse, test_corr, outputs_flat, targets_flat = test_vim(model, test_x, test_y)
    plot_vim(loss_values, correlation_values, num_epochs, time, name='training_performance_vim_400_epoch_0')
    plot_test_results(outputs_flat, targets_flat, test_corr, test_mse, name='test_performace_vim_400_epoch_0')

if __name__ == '__main__':
    main()

