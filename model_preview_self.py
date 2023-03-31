from distutils.log import info
from data_utils.data_setup_with_labels import create_dataloaders, create_dataloaders_prediction
from data_utils.balanced_split import balanced_split
from model_utils.conv_autoencoder import AE
from torchvision import transforms, utils
import pandas as pd
import matplotlib.pyplot as plt
import torch
import json
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

def make_predictions(model: torch.nn.Module, data: list, device: torch.device = device):
    pred_gen = []
    model.eval()
    with torch.inference_mode():
        for sample in data:
            # Prepare sample
            sample = torch.unsqueeze(sample, dim=0).to(device) # Add an extra dimension and send sample to device

            # Forward pass (model outputs raw logit)
            latent, pred_gen = model(sample)
            
    # Stack the pred_probs to turn list into a tensor
    return latent, pred_gen

# Setup directories
DATA_PATH = "data/happy_hoppy"

data_transforms = transforms.Compose([
  transforms.ToTensor()
])

# Create DataLoaders with help from data_setup.py
dataloader = create_dataloaders_prediction(
    data_path = DATA_PATH,
    split=0.1,
    transforms=data_transforms,
    batch_size=1
)

MODEL_SAVE_PATH = "models/05_going_modular_script_mode_tinyvgg_model.pth"

model = AE().to(device)

model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

model = model.to(device)

json_to_save = []
"""
labels = {
    "index": [],
    "data": [],
    "columns": []
}
"""
labels = {}

exp_folder_name = "experiment33"
images_path = os.path.join(exp_folder_name, "images")
generated_path = os.path.join(exp_folder_name, "generated")
label_file_path = os.path.join(exp_folder_name, "labels.json")

for idx, item in enumerate(dataloader):
    pandas_idx = dataloader.sampler.data_source.indices[idx]
    info_row = dataloader.sampler.data_source.dataset.info.iloc[pandas_idx]
    labels[idx] = info_row.to_dict()
    labels[idx]["original_index"] = pandas_idx
    
    
    
    #img = next(it)
    img = item[0]
    label = item[1]
    filename = "{}-{}-{}.png".format(pandas_idx, info_row["source_name"], info_row["survey"])
    labels[idx]["filename"] = filename
    image_file_path = os.path.join(images_path, filename)
    generated_file_path = "{}/{}".format(generated_path, filename)

    with torch.no_grad():
        model.eval()
        latent, pred_gen = make_predictions(model=model, data=img)

    utils.save_image(img, image_file_path)
    utils.save_image(pred_gen, generated_file_path)

    #print(latent.to("cpu").numpy()[0].tolist())
    json_to_save.append(latent.to("cpu").numpy()[0].tolist())
    #labels["columns"].append(str(idx)+".png")

out_file = open("experiment3/embeddings.json", "w+")
json.dump(json_to_save, out_file)
#out_file = open("experiment3/labels.json", "w+")
labels_pd  = pd.DataFrame.from_dict(labels, orient="index")
my_cols = set(labels_pd.columns)
my_cols.remove('original_path')
my_cols.remove('target_path')
my_cols = list(my_cols)
labels_pd_to_save = labels_pd[my_cols]
labels_pd_to_save.to_json(label_file_path, orient='index')
#json.dump(labels, out_file)