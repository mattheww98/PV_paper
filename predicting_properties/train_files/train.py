import json
import glob
import os
import glob
import torch
import numpy as np

from alignn.data import get_train_val_loaders
from jarvis.db.jsonutils import loadjson
from alignn.config import TrainingConfig
from alignn.models.alignn import ALIGNN
from alignn.train import train_dgl

from tqdm import tqdm

from codecarbon import OfflineEmissionsTracker
out_dir = os.getcwd()
tracker = OfflineEmissionsTracker(
            output_dir=out_dir,
            country_iso_code="GBR"
        )

tracker.start()

## Variables that need to be set
data = 'corr_etas_input.json'  # The data to re-train on (gvrh)
model_path = False #'checkpoints/checkpoint_441.pt' # The model checkpoint to load initially
config_path = 'config-train.json' # The config file to use
## END of variables that need to be set

with open(data, "rb") as f:
    dataset = json.loads(f.read())

# ### Now set up alignn model

config = loadjson(config_path)
config = TrainingConfig(**config)

(
    train_loader,
    val_loader,
    test_loader,
    prepare_batch,
) = get_train_val_loaders(
    dataset_array=dataset,
    target=config.target,
    n_train=config.n_train,
    n_val=config.n_val,
    n_test=config.n_test,
    train_ratio=config.train_ratio,
    val_ratio=config.val_ratio,
    test_ratio=config.test_ratio,
    batch_size=config.batch_size,
    atom_features=config.atom_features,
    neighbor_strategy=config.neighbor_strategy,
    standardize=config.atom_features != "cgcnn",
    id_tag=config.id_tag,
    pin_memory=config.pin_memory,
    workers=config.num_workers,
    save_dataloader=config.save_dataloader,
    use_canonize=config.use_canonize,
    filename=config.filename,
    cutoff=config.cutoff,
    max_neighbors=config.max_neighbors,
    output_features=config.model.output_features,
    classification_threshold=config.classification_threshold,
    target_multiplication_factor=config.target_multiplication_factor,
    standard_scalar_and_pca=config.standard_scalar_and_pca,
    keep_data_order=config.keep_data_order,
    output_dir=config.output_dir,
)


## Check for GPU and CUDA
device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")
print(config.model)

## Set up the model
model = ALIGNN(config.model)
if os.path.isfile(model_path):
    print('Starting from pre-trained model')
    model.load_state_dict(torch.load(model_path,  map_location=torch.device(device))["model"])
model.to(device)

## Run the training
train_dgl(
    config,
    model,
    train_val_test_loaders=[
        train_loader,
        val_loader,
        test_loader,
        prepare_batch,
    ],
)

## Post-processing - sort through the checkpoint files, keeping only the final and best validation checkpoints
with open('checkpoints/history_val.json', 'r') as f:
  data = json.load(f)

min_val = np.argmin(data['loss'])
min_chck = './checkpoints/checkpoint_' + str(min_val) + '.pt'
last_chck = './checkpoints/checkpoint_' + str(len(data['loss'])) + '.pt'
os.rename('./checkpoints/best_model.pt','best_model.pt')
os.rename(last_chck, 'checkpoint_final.pt')

tracker.stop()
dir_path = "./checkpoints/"
# pattern for file names to be deleted
file_pattern = "*.pt"
# get a list of file paths using the glob module
file_paths = glob.glob(os.path.join(dir_path, file_pattern))
# loop over each file path and delete the file
for file_path in file_paths:
    os.remove(file_path)
