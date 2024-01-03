# Chemistry-Informed Molecular Graph (CIMG) Descriptor

Published in [Chemistry-informed molecular graph as reaction descriptor for machine-learned retrosynthesis planning](https://www.pnas.org/doi/10.1073/pnas.2212711119)

## 1. Installation

### 1.1. Install RDKit

```bash
conda install -c rdkit rdkit
```

### 1.2. Install pytorch and torch-geometrics

If use GPU version, please install these dependencies following the instructions on the [pytorch website](https://pytorch.org/get-started/locally/) and [torch-geometric website](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

If use CPU version, you can install pytorch and torch-geometrics by the following commands:


```bash
pip install torch
pip install torch_geometric
pip install torch_scatter torch_sparse torch_cluster
```

### 1.3. Install cimg_desc

```bash
pip install cimg-desc
```

## 2. Usage

### 2.1. Predict CIMG vector for one Molecule

```python
from cimg_desc import Predictor

predictor = Predictor('cpu') # or Predictor('cuda')
smiles = 'c1ccccc1'
cimg = predictor.predict(smiles)

```

### 2.2. Predict CIMG vector for a batch of Molecules

```python
from cimg_desc import Predictor

predictor = Predictor('cpu') # or Predictor('cuda')
smiles_list = ['c1ccccc1', 'c1ccccc1O']
cimg_list = predictor.predict_batch(smiles_list)

```
