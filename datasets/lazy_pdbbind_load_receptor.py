import binascii
import traceback
import glob
import math
import os
import pickle
import mmap
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
import random
import copy
import torch.nn.functional as F
import numpy as np
import torch
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MolFromSmiles, AddHs
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.transforms import BaseTransform
from tqdm import tqdm
from rdkit.Chem import RemoveAllHs

from datasets.process_mols import read_molecule, get_lig_graph_with_matching, generate_conformer, moad_extract_receptor_structure
from utils.diffusion_utils import modify_conformer, set_time
from utils.utils import read_strings_from_txt, crop_beyond
from utils import so3, torus
from datasets.lazy_pdbbind import LazyPDBBindSet, read_mol


class LazyPDBBindSet_LoadReceptor(LazyPDBBindSet):
    """
    This class adapts LazyPDBBindSet to load alternative receptor configurations other than the ground truth receptor.
    """
    def __init__(self, pdbbind_dir, plinder_dir, plinder_csv=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pdbbind_dir = pdbbind_dir
        self.plinder_dir = plinder_dir

        # You can add or override attributes here as needed for loading alternative receptors

    def get(self, idx):
        name = self.complex_names_all[idx]
        return self.get_by_name(name)        

    def get_complex(self, name, lm_embedding_chains):
        parts = name.split('|')
        plinder_id = parts[0]
        superpose_id = parts[1]
        pdbbind_id = parts[2]

        superpose_folder = os.path.join(self.plinder_dir, plinder_id, superpose_id)
        if not os.path.exists(superpose_folder):
            print("Superpose folder not found", superpose_folder)
            return None, None,

        pdbbind_folder = os.path.join(self.pdbbind_dir, pdbbind_id)
        if not os.path.exists(pdbbind_folder):
            print("PDBBind folder not found", pdbbind_folder)
            return None, None,

        try:
            parsed_name = pdbbind_id.split('_')
            pdb = parsed_name[0]
            if len(parsed_name) >= 3:
                lig_name = parsed_name[2]
            else:
                lig_name = None
            orig_lig_pos = None
            if lig_name and self.ligand_smiles and (pdb.upper(), lig_name.upper()) in self.ligand_smiles:
                lig_smiles = self.ligand_smiles[(pdb.upper(), lig_name.upper())]
                lig = Chem.MolFromSmiles(lig_smiles)
                generate_conformer(lig)
                Chem.SanitizeMol(lig)
                lig = Chem.RemoveHs(lig, sanitize=True)

                # The SMILE string ligand and its original PDB don't
                # necessarily load the atoms in the same order. This results in
                # unusually high RMSD results, but normal centroid distance
                # results since the centroid averages the atom positions anyway.
                # We can fix the RMSD by using the canonical SMILE ordering to
                # appropriately match up the coordinates.
                Chem.MolToSmiles(lig, canonical=True)
                lig_canon_atom_idx = list(lig.GetPropsAsDict(includePrivate=True, includeComputed=True)['_smilesAtomOutputOrder'])
                orig_lig = read_mol(self.pdbbind_dir, name, pdb, suffix=self.ligand_file, remove_hs=True)
                Chem.MolToSmiles(orig_lig, canonical=True)
                orig_lig_canon_atom_idx = list(orig_lig.GetPropsAsDict(includePrivate=True, includeComputed=True)['_smilesAtomOutputOrder'])
                orig_lig_pos = np.array(orig_lig.GetConformers()[0].GetPositions())
                orig_lig_pos[lig_canon_atom_idx] = orig_lig_pos[orig_lig_canon_atom_idx]
                if orig_lig_pos is None:
                    print('Error loading ligand original atom positions')
                    return None, None
            else:
                lig = read_mol(self.pdbbind_dir, pdbbind_id, pdb, suffix=self.ligand_file, remove_hs=False)


            if self.max_lig_size != None and lig.GetNumHeavyAtoms() > self.max_lig_size:
                print(f'Ligand with {lig.GetNumHeavyAtoms()} heavy atoms is larger than max_lig_size {self.max_lig_size}. Not including {name} in preprocessed data.')
                return None, None
            complex_graph = HeteroData()
            complex_graph['name'] = name
            get_lig_graph_with_matching(lig, complex_graph, self.popsize, self.maxiter, self.matching, self.keep_original,
                                        self.num_conformers, remove_hs=self.remove_hs, tries=self.matching_tries, skip_tor_model=self.skip_tor_model)

            moad_extract_receptor_structure(path=os.path.join(superpose_folder, 'superposed.cif'),
                                            complex_graph=complex_graph,
                                            neighbor_cutoff=self.receptor_radius,
                                            max_neighbors=self.c_alpha_max_neighbors,
                                            lm_embeddings=lm_embedding_chains,
                                            knn_only_graph=self.knn_only_graph,
                                            all_atoms=self.all_atoms,
                                            atom_cutoff=self.atom_radius,
                                            atom_max_neighbors=self.atom_max_neighbors)
            if orig_lig_pos is not None:
                complex_graph['ligand'].orig_pos = orig_lig_pos

        except Exception as e:
            print(f'Skipping {name} because of the error:')
            print(e)
            traceback.print_exc()
            return None, None
        if self.max_receptor_size is not None and complex_graph['receptor'].pos.shape[0] > self.max_receptor_size:
            print(f"Skipping {name} because receptor was too large (" + str(complex_graph['receptor'].pos.shape[0]) + ' residues)')
            return None, None
        protein_center = torch.mean(complex_graph['receptor'].pos, dim=0, keepdim=True)
        #lig_center = torch.mean(complex_graph['ligand'].pos, dim=0, keepdim=True)
        lig_center = torch.mean(torch.from_numpy(complex_graph['ligand'].orig_pos), dim=0, keepdim=True)
        #if torch.sum((protein_center - lig_center)**2) > 1000.0:
            # This usually means our simulation dissociated
            #print('Skipping ' + complex_graph['name'] + ' due to dissociation', flush=True)
            #return None, None
        complex_graph['receptor'].pos -= protein_center
        if self.all_atoms:
            complex_graph['atom'].pos -= protein_center

        if (not self.matching) or self.num_conformers == 1:
            complex_graph['ligand'].pos -= protein_center
        else:
            for p in complex_graph['ligand'].pos:
                p -= protein_center

        complex_graph.original_center = protein_center
        complex_graph['receptor_name'] = name
        return complex_graph, lig

    def preprocessing(self):
        if self.smile_file:
            with open(self.smile_file, 'r') as smile_file:
                for row in smile_file.readlines():
                    parsed_row = row.split('\t')
                    self.ligand_smiles[(parsed_row[0].upper(), parsed_row[1].upper())] = parsed_row[2].strip()
        print("Preprocessing")
        print(self.split_path)
        self.complex_names_all = read_strings_from_txt(self.split_path)
        if self.slurm_array_task_count:
            slurm_task_size = int(math.ceil(len(self.complex_names_all) / self.slurm_array_task_count))
            start_idx = self.slurm_array_idx * slurm_task_size
            end_idx = min(len(self.complex_names_all), (self.slurm_array_idx + 1) * slurm_task_size)
            self.complex_names_all = self.complex_names_all[start_idx:end_idx]
            print('Processing complexes ' + str(start_idx) + ' through ' + str(end_idx))
        else:
            if self.limit_complexes is not None and self.limit_complexes != 0:
                self.complex_names_all = self.complex_names_all[:self.limit_complexes]
        # generate embeddings for all of the complexes up front
        # load only the embeddings specific to the test set
        if self.esm_embeddings_path is not None:
            chain_embeddings_dictlist = defaultdict(list)
            chain_indices_dictlist = defaultdict(list)
            self.lm_complex_names = ["_".join(name.split('|')[:2]) for name in self.complex_names_all]
            for embedding in glob.glob(os.path.join(self.esm_embeddings_path, '*_chain_*.pt')):
                parsed_embedding = embedding.split('/')[-1].split('_chain_')
                key_name = parsed_embedding[0]
                if key_name not in self.lm_complex_names:
                    continue
                chain_idx = parsed_embedding[1].strip('.pt')
                chain_embeddings_dictlist[key_name].append(embedding)
                chain_indices_dictlist[key_name].append(int(chain_idx))
            self.lm_embeddings_chains_all = []
            for name in self.lm_complex_names:
                # key_name = ".".join(name.split('|')[:2])
                complex_chains_embeddings = chain_embeddings_dictlist[name]
                complex_chains_indices = chain_indices_dictlist[name]
                chain_reorder_idx = np.argsort(complex_chains_indices)
                reordered_chains = [complex_chains_embeddings[i] for i in chain_reorder_idx]
                self.lm_embeddings_chains_all.append(reordered_chains)
        else:
            self.lm_embeddings_chains_all = [None] * len(self.complex_names_all)
        self.complex_lm_embeddings = {}
        for i in range(0, len(self.lm_complex_names)):
            name = self.lm_complex_names[i]
            self.complex_lm_embeddings[name] = self.lm_embeddings_chains_all[i]

    def get_by_name(self, name):
        if self.cache and self.cache_idx:
            if name in self.cache_idx:
                offset = self.cache_idx[name][0]
                size = self.cache_idx[name][1]
                if size == 0:
                    # Length of 0 indicates failed preprocessing
                    return None
                else:
                    self.cache.seek(offset, 0)
                    serialized_hetgraph = self.cache.read(size)
                    ret = pickle.loads(serialized_hetgraph)
                    return ret
            else:
                return None

        lm_key_name = "_".join(name.split('|')[:2])
        if not lm_key_name or lm_key_name not in self.complex_lm_embeddings:
            return None

        lm_embedding_chains = list(map(lambda x: torch.load(x)['representations'][33], self.complex_lm_embeddings[lm_key_name]))
        complex_graph, lig = self.get_complex(name, lm_embedding_chains)
        if not complex_graph or (self.require_ligand and not lig):
            return None

        if self.require_ligand:
            complex_graph.mol = RemoveAllHs(lig)

        for a in ['random_coords', 'coords', 'seq', 'sequence', 'mask', 'rmsd_matching', 'cluster', 'orig_seq', 'to_keep', 'chain_ids']:
            if hasattr(complex_graph, a):
                delattr(complex_graph, a)
            if hasattr(complex_graph['receptor'], a):
                delattr(complex_graph['receptor'], a)

        return complex_graph
