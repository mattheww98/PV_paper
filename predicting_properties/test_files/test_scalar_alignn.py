import json
import random
import torch

import numpy as np
import pandas as pd

from jarvis.core.graphs import Graph
from jarvis.core.atoms import Atoms

from jarvis.db.jsonutils import dumpjson
from jarvis.db.jsonutils import loadjson

from pymatgen.io.jarvis import JarvisAtomsAdaptor
from pymatgen.core import Structure

#from alignn.models.alignn import ALIGNN
from alignn.models.alignn import ALIGNN
from alignn.models.alignn import ALIGNNConfig
from tqdm import tqdm

import dgl


def atoms_to_graph(atoms, cutoff=8.0, max_neighbors=12,
    atom_features="cgcnn", use_canonize=True):
    """Convert structure dict to DGLGraph."""
    structure = Atoms.from_dict(atoms)
    #structure = JarvisAtomsAdaptor.get_atoms(Structure.from_dict(atoms))
    return Graph.atom_dgl_multigraph(
        structure,
        cutoff=cutoff,
        atom_features=atom_features,
        max_neighbors=max_neighbors,
        compute_line_graph=True,
        use_canonize=use_canonize,
    )

### Parameters that we need to set
data = 'test_input.json'
checkpoint_fp = 'best_model.pt'
lim = False      # Just used if you want a smaller subset for testing
n_outputs = 1 # change for spectral vs scalar properties (or diff. no. of bins for spectrum)
preds_file = 'val_preds.csv'
### No need to edit beyond here



device = "cpu"
if torch.cuda.is_available():
    print('Found GPU and CUDA')
    device = torch.device("cuda")

with open(data, "rb") as f:
    dataset = json.loads(f.read())
if lim:
    dataset = dataset[:lim]

test_data = []
for i in dataset:
    struc = i['atoms']
    test_data.append(atoms_to_graph(struc))

config = ALIGNNConfig(name="alignn")
config.output_features=n_outputs
model = ALIGNN(config)

model.to(device)
model.load_state_dict(torch.load(checkpoint_fp, map_location=torch.device(device))['model'])
val_results = []
model.eval()
for i in tqdm(range(len(test_data))):
    in_data = (test_data[i][0].to(device),test_data[i][1].to(device))
    val_id = [dataset[i]['jid']]
    if n_outputs == 1:
        pred = [model(in_data).cpu().detach().numpy().tolist()]
    else:
        pred = model(in_data).cpu().detach().numpy().tolist()
    val_results.append(val_id+pred)
df = pd.DataFrame(data = val_results)
df.to_csv(preds_file,index=False)
