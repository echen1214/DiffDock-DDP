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


class LazyPDBBindSet(Dataset):
    def __init__(self, root, transform=None, cache_path='data/cache', split_path='data/', limit_complexes=0, chain_cutoff=10,
                 receptor_radius=30, num_workers=1, c_alpha_max_neighbors=None, popsize=15, maxiter=15,
                 matching=True, keep_original=False, max_lig_size=None, remove_hs=False, num_conformers=1, all_atoms=False,
                 atom_radius=5, atom_max_neighbors=None, esm_embeddings_path=None, require_ligand=False,
                 include_miscellaneous_atoms=False,
                 protein_path_list=None, ligand_descriptions=None, keep_local_structures=False,
                 protein_file="protein_processed", ligand_file='ligand',
                 smile_file=None,
                 slurm_array_idx=None,
                 slurm_array_task_count=None,
                 max_receptor_size=None,
                 knn_only_graph=False, matching_tries=1, dataset='AlloSet'):

        super(LazyPDBBindSet, self).__init__(root, transform)
        self.smile_file = smile_file
        self.ligand_smiles = {}
        self.slurm_array_idx = slurm_array_idx
        self.slurm_array_task_count = slurm_array_task_count
        self.pdbbind_dir = root
        self.include_miscellaneous_atoms = include_miscellaneous_atoms
        self.max_lig_size = max_lig_size
        self.split_path = split_path
        self.limit_complexes = limit_complexes
        self.chain_cutoff = chain_cutoff
        self.receptor_radius = receptor_radius
        self.num_workers = num_workers
        self.c_alpha_max_neighbors = c_alpha_max_neighbors
        self.remove_hs = remove_hs
        self.esm_embeddings_path = esm_embeddings_path
        self.use_old_wrong_embedding_order = False
        self.require_ligand = require_ligand
        self.protein_path_list = protein_path_list
        self.ligand_descriptions = ligand_descriptions
        self.keep_local_structures = keep_local_structures
        self.protein_file = protein_file
        self.fixed_knn_radius_graph = True
        self.max_receptor_size = max_receptor_size
        self.knn_only_graph = knn_only_graph
        self.matching_tries = matching_tries
        self.ligand_file = ligand_file
        self.dataset = dataset
        assert knn_only_graph or (not all_atoms)
        self.all_atoms = all_atoms
        self.popsize, self.maxiter = popsize, maxiter
        self.matching, self.keep_original = matching, keep_original
        self.num_conformers = num_conformers

        self.atom_radius, self.atom_max_neighbors = atom_radius, atom_max_neighbors
        self.preprocessing()

        self.cache_idx = None
        self.cache = None
        self.cache_file = None
        os.makedirs(cache_path, exist_ok=True)
        cache_idx_path = os.path.join(cache_path, 'index.pkl')
        cache_path = os.path.join(cache_path, 'cache.dat')
        if os.path.exists(cache_idx_path) and os.path.exists(cache_path):
            print('Cache exists!')
            with open(cache_idx_path, 'rb') as cache_idx_file:
                self.cache_idx = pickle.load(cache_idx_file)
            self.cache_file = open(cache_path, 'rb')
            self.cache = mmap.mmap(self.cache_file.fileno(), 0, flags=mmap.MAP_PRIVATE, prot=mmap.PROT_READ)
            self.complex_names_all = [x for x in self.complex_names_all if x in self.cache_idx]

    def __del__(self):
        if self.cache:
            self.cache.close()
        if self.cache_file:
            self.cache_file.close()

    def len(self):
        return len(self.complex_names_all)

    def get(self, idx):
        name = self.complex_names_all[idx]
        return self.get_by_name(name)

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
                    return pickle.loads(serialized_hetgraph)
            else:
                return None

        if not name or name not in self.complex_lm_embeddings:
            return None

        lm_embedding_chains = list(map(lambda x: torch.load(x)['representations'][33], self.complex_lm_embeddings[name]))
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

    def preprocessing(self):
        if self.smile_file:
            with open(self.smile_file, 'r') as smile_file:
                for row in smile_file.readlines():
                    parsed_row = row.split('\t')
                    self.ligand_smiles[(parsed_row[0].upper(), parsed_row[1].upper())] = parsed_row[2].strip()

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
            for embedding in glob.glob(os.path.join(self.esm_embeddings_path, '*_chain_*.pt')):
                parsed_embedding = embedding.split('/')[-1].split('.')[0].split('_chain_')
                key_name = parsed_embedding[0]
                if key_name not in self.complex_names_all:
                    continue
                chain_idx = parsed_embedding[1]
                chain_embeddings_dictlist[key_name].append(embedding)
                chain_indices_dictlist[key_name].append(int(chain_idx))
            self.lm_embeddings_chains_all = []
            for name in self.complex_names_all:
                complex_chains_embeddings = chain_embeddings_dictlist[name]
                complex_chains_indices = chain_indices_dictlist[name]
                chain_reorder_idx = np.argsort(complex_chains_indices)
                reordered_chains = [complex_chains_embeddings[i] for i in chain_reorder_idx]
                self.lm_embeddings_chains_all.append(reordered_chains)
        else:
            self.lm_embeddings_chains_all = [None] * len(self.complex_names_all)
        self.complex_lm_embeddings = {}
        for i in range(0, len(self.complex_names_all)):
            self.complex_lm_embeddings[self.complex_names_all[i]] = self.lm_embeddings_chains_all[i]

    def get_complex(self, name, lm_embedding_chains):
        if not os.path.exists(os.path.join(self.pdbbind_dir, name)):
            print("Folder not found", name)
            return None, None,

        try:
            parsed_name = name.split('_')
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
                lig = read_mol(self.pdbbind_dir, name, pdb, suffix=self.ligand_file, remove_hs=False)


            if self.max_lig_size != None and lig.GetNumHeavyAtoms() > self.max_lig_size:
                print(f'Ligand with {lig.GetNumHeavyAtoms()} heavy atoms is larger than max_lig_size {self.max_lig_size}. Not including {name} in preprocessed data.')
                return None, None
            complex_graph = HeteroData()
            complex_graph['name'] = name
            get_lig_graph_with_matching(lig, complex_graph, self.popsize, self.maxiter, self.matching, self.keep_original,
                                        self.num_conformers, remove_hs=self.remove_hs, tries=self.matching_tries)

            moad_extract_receptor_structure(path=os.path.join(self.pdbbind_dir, name, f'{pdb}_{self.protein_file}.pdb'),
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
            print('Skipping {name} because receptor was too large (' + str(complex_graph['receptor'].pos.shape[0]) + ' residues)')
            return None, None
        protein_center = torch.mean(complex_graph['receptor'].pos, dim=0, keepdim=True)
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

def read_mol(pdbbind_dir, complex_name, pdb_name, suffix='ligand', remove_hs=False):
    try:
        lig = read_molecule(os.path.join(pdbbind_dir, complex_name, f'{pdb_name}_{suffix}.sdf'), remove_hs=remove_hs, sanitize=True)
    except:
        lig = None
    if lig is None:  # read mol2 file if sdf file cannot be sanitized
        try:
            lig = read_molecule(os.path.join(pdbbind_dir, complex_name, f'{pdb_name}_{suffix}.mol2'), remove_hs=remove_hs, sanitize=True)
        except:
            lig = None
    if lig is None:  # read pdb file if neither sdf nor mol2 can be sanitized
        lig = read_molecule(os.path.join(pdbbind_dir, complex_name, f'{pdb_name}_{suffix}.pdb'), remove_hs=remove_hs, sanitize=True)
    return lig
