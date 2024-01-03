#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/6/20 15:58
# @Author  : zhangbc0315@outlook.com
# @File    : handler.py
# @Software: PyCharm

import torch
from torch_geometric.data import Data as GData
from torch_geometric.data import Batch
from rdkit.Chem.rdchem import Atom, Bond, Mol
from rdkit.Chem import AllChem, GetPeriodicTable


class Handler:

    def __init__(self):
        self._kernels = [86, 54, 36, 18, 10, 2]

    def get_row_and_col(self, atomic_num: int):
        row = 1
        col = 1
        for i, k in enumerate(self._kernels):
            if atomic_num > k:
                row = 7 - i
                col = atomic_num - k
                break
        return row, col

    @classmethod
    def _hybrid_to_num(cls, hybrid) -> int:
        id_hybrid = int(hybrid)
        return id_hybrid

    @classmethod
    def _bool_to_num(cls, b: bool) -> int:
        return 1 if b else 0

    def _get_node_features_from_rdatom(self, atom: Atom) -> [int]:
        row, col = self.get_row_and_col(atom.GetAtomicNum())
        return [atom.GetAtomicNum(),
                atom.GetTotalNumHs(),
                atom.GetFormalCharge(),
                atom.GetNumRadicalElectrons(),
                atom.GetTotalValence(),
                self._hybrid_to_num(atom.GetHybridization()),
                self._bool_to_num(atom.IsInRing()),
                self._bool_to_num(atom.GetIsAromatic()),
                row,
                col]

    def _get_nodes_features_from_rdmol(self, mol: Mol) -> [[int]]:
        return [self._get_node_features_from_rdatom(mol.GetAtomWithIdx(i)) for i in range(mol.GetNumAtoms())]

    @classmethod
    def get_edge_features_from_rdbond(cls, bond: Bond) -> [int]:
        bond_level = int(bond.GetBondType())
        if bond_level == 12:
            bond_level = 2
        elif bond_level >= 2:
            bond_level += 1
        res = [0] * 5
        res[bond_level] = 1
        return res

    @classmethod
    def get_connection_from_rdbond(cls, bond: Bond, reverse: bool = False) -> [int]:
        if reverse:
            return [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
        else:
            return [bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]

    def _get_edges_features_and_connections_from_rdmol(self, mol: Mol) -> ([[int]], [[int]]):
        edges_features = []
        connections = []
        for bond in mol.GetBonds():
            edges_features.append(self.get_edge_features_from_rdbond(bond))
            edges_features.append(self.get_edge_features_from_rdbond(bond))
            connections.append(self.get_connection_from_rdbond(bond, False))
            connections.append(self.get_connection_from_rdbond(bond, True))
        return edges_features, connections

    def get_gnn_data_from_rdmol(self, mol: Mol) -> GData:
        edges_features, connections = self._get_edges_features_and_connections_from_rdmol(mol)
        x = torch.tensor(self._get_nodes_features_from_rdmol(mol), dtype=torch.float)
        edge_index = torch.tensor(connections, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edges_features, dtype=torch.float)
        return GData(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def get_gnn_data_from_smiles(self, smiles: str) -> GData:
        mol = AllChem.MolFromSmiles(smiles)
        return self.get_gnn_data_from_rdmol(mol)

    def initialization(self, smiles: str):
        return Batch.from_data_list([self.get_gnn_data_from_smiles(smiles)])

    def initialization_batch(self, smiles_list: [str]):
        return Batch.from_data_list([self.get_gnn_data_from_smiles(smiles) for smiles in smiles_list])


if __name__ == "__main__":
    pass
