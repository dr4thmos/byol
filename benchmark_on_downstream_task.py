# Qui bisogna fare classificazione sui diversi dataset
# Zorro Compact / RadioGalaxyes / Sidelobes
# Robin Background / Diffuse / Doubles / Extended / Extended-Multisland / Point-like / Triples
# Mirabest FRI / FRII
# CLoud dataset? SWIMCAT-Ext o Cirrus Cumulus Stratus Nimbus (CCSN) Database o TJNU ground-based cloud dataset (GCD)

from data_utils.zorro2 import Zorro
from data_utils.robin import Robin
#from data_utils.mirabest import Mirabest

import torch
from torchvision.transforms import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader
import torchvision
import os

from model_utils.resnet_base_network import ResNet
from model_utils.finetune_encoder import FinetuneEncoder

from sklearn.metrics import top_k_accuracy_score

ZORRO_DATA_PATH     = "./1-ZORRO/data"
ROBIN_DATA_PATH     = "./2-ROBIN"
MIRABEST_DATA_PATH  = "./Mirabest"

IMG_SIZE = 128

def plot_grid(data, writer, epoch_counter):
    grid = torchvision.utils.make_grid(data[:64])
    writer.add_image('views_paired', grid, global_step=epoch_counter)
    #print("Grid done")

def train_linear_classifier(base_model, train_dataset, eval_dataset, n_epochs=20):
    linear_classificator = torch.nn.Linear(
        base_model.repr_shape, train_dataset.num_classes).to(device)
    finetuning_model = FinetuneEncoder(base_model.encoder, linear_classificator).to(device)
    optimizer_config = {
        "lr": 0.007,
        "momentum": 0.9,
        "weight_decay": 0.0004
    }
    optimizer = torch.optim.SGD(list(finetuning_model.parameters()),
                                **optimizer_config)
    criterion = torch.nn.CrossEntropyLoss()
    
    trainer_config = {
        "batch_size": 256,
        "num_workers": 8
    }
    train_loader        = DataLoader(train_dataset, batch_size=trainer_config['batch_size'],
                                  num_workers=trainer_config['num_workers'], drop_last=True, shuffle=True)

    validation_loader   = DataLoader(eval_dataset, batch_size=trainer_config['batch_size'],
                            num_workers=trainer_config['num_workers'], drop_last=False, shuffle=True)

    min_val_loss = 100.0

    for epoch_counter in range(n_epochs):
        print(f"Epoch {epoch_counter}")
        running_loss = running_accuracy = running_val_loss = running_val_accuracy = 0.0
        grid_done = False

        finetuning_model.train()
        
        for data in train_loader:
            image_batch = data[0].to(device)
            if not(grid_done):
                grid_done = plot_grid(image_batch, writer, epoch_counter)
            #print(image_batch.shape)
            predicted = finetuning_model(image_batch)
            #print(predicted.shape)
            #print(data[1].shape)
            #print(data[1].to(device))
            #print(predicted.to(device))
            loss = criterion(predicted.to(device), data[1].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            acc = top_k_accuracy_score(data[1].to("cpu").detach().numpy(), predicted.to("cpu").detach().numpy(),k=1)
            running_accuracy += acc

        finetuning_model.eval()
        for data in validation_loader:
            image_batch = data[0].to(device)
            if not(grid_done):
                grid_done = plot_grid(image_batch, writer, epoch_counter)
            #print(image_batch.shape)
            with torch.no_grad():
                predicted = finetuning_model(image_batch)
            
            loss = criterion(predicted, data[1].to(device))
            running_val_loss += loss.item()
            
            acc = top_k_accuracy_score(data[1].to("cpu").detach().numpy(), predicted.to("cpu").detach().numpy(),k=1)
            running_val_accuracy += acc
        
        val_acc         = running_val_accuracy/len(validation_loader)
        val_loss        = running_val_loss/len(validation_loader)
        train_loss      = running_loss/len(train_loader)
        train_accuracy  = running_accuracy/len(train_loader)
        
        writer.add_scalar('loss/train_loss', train_loss, global_step=epoch_counter)
        writer.add_scalar('acc/train_accuracy', train_accuracy, global_step=epoch_counter)
        writer.add_scalar('loss/val_loss', val_loss, global_step=epoch_counter)
        writer.add_scalar('acc/val_accuracy', val_acc, global_step=epoch_counter)

        if min_val_loss > val_loss:
            min_val_loss = val_loss
            torch.save({
                'encoder': finetuning_model.encoder.state_dict(),
            }, os.path.join(writer.log_dir, "finetuned_model.pt"))
   

def test_linear_classifier(model, n_classes, dataset, iteration):
    pass

device = "cuda"


base_model_config = {
    "name": "resnet18",
    "input_shape":{
        "width": 128,
        "height": 128,
        "channels": 3
    },
    "fine_tune_from": None,
    "projection_head":{
        "mlp_hidden_size": 1024,
        "projection_size": 512
    },
    "pretrained_weights": False
}
base_model = ResNet(**base_model_config).to(device)

zorro_train_data_augmentation       = transforms.Compose([
        transforms.RandomRotation(degrees=90.0, interpolation=transforms.InterpolationMode.BILINEAR, expand=True),
        transforms.Resize([IMG_SIZE,IMG_SIZE]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) ),
        transforms.ConvertImageDtype(dtype=torch.float32)
    ])

zorro_validation_data_augmentation  = transforms.Compose([
        #transforms.RandomRotation(degrees=90.0, interpolation=transforms.InterpolationMode.BILINEAR, expand=True),
        transforms.Resize([IMG_SIZE,IMG_SIZE]),
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomVerticalFlip(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) ),
        transforms.ConvertImageDtype(dtype=torch.float32)
    ])

zorro_train_dataset       = Zorro(targ_dir = ZORRO_DATA_PATH, transform = zorro_train_data_augmentation,
                        datalist="train.json"
    )

zorro_validation_dataset  = Zorro(targ_dir = ZORRO_DATA_PATH, transform = zorro_validation_data_augmentation,
                        datalist="validation.json"
    )

zorro_eval_dataset        = Zorro(targ_dir = ZORRO_DATA_PATH, transform = zorro_validation_data_augmentation,
                        datalist="validation.json"
    )
#print(summary(pretrained_model.encoder, (3,128,128)))
#print(summary(linear_classificator, (pretrained_model.repr_shape)))
writer = SummaryWriter(log_dir=os.path.join("logs","benchmark", "zorro_scratch"))
train_linear_classifier(base_model, zorro_train_dataset, zorro_validation_dataset)


base_model_config = {
    "name": "resnet18",
    "input_shape":{
        "width": 128,
        "height": 128,
        "channels": 3
    },
    "fine_tune_from": "logs/test/test4/best_model.pt",
    "projection_head":{
        "mlp_hidden_size": 1024,
        "projection_size": 512
    },
    "pretrained_weights": True
}
base_model = ResNet(**base_model_config).to(device)

checkpoint = torch.load(base_model_config["fine_tune_from"])
base_model.load_state_dict(checkpoint['online_network_state_dict'])


zorro_train_data_augmentation       = transforms.Compose([
        transforms.RandomRotation(degrees=90.0, interpolation=transforms.InterpolationMode.BILINEAR, expand=True),
        transforms.Resize([IMG_SIZE,IMG_SIZE]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) ),
        transforms.ConvertImageDtype(dtype=torch.float32)
    ])

zorro_validation_data_augmentation  = transforms.Compose([
        #transforms.RandomRotation(degrees=90.0, interpolation=transforms.InterpolationMode.BILINEAR, expand=True),
        transforms.Resize([IMG_SIZE,IMG_SIZE]),
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomVerticalFlip(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) ),
        transforms.ConvertImageDtype(dtype=torch.float32)
    ])
zorro_train_dataset       = Zorro(targ_dir = ZORRO_DATA_PATH, transform = zorro_train_data_augmentation,
                        datalist="train.json"
    )

zorro_validation_dataset  = Zorro(targ_dir = ZORRO_DATA_PATH, transform = zorro_validation_data_augmentation,
                        datalist="validation.json"
    )

zorro_eval_dataset        = Zorro(targ_dir = ZORRO_DATA_PATH, transform = zorro_validation_data_augmentation,
                        datalist="validation.json"
    )
#print(summary(pretrained_model.encoder, (3,128,128)))
#print(summary(linear_classificator, (pretrained_model.repr_shape)))
writer = SummaryWriter(log_dir=os.path.join("logs","benchmark", "zorro_byol"))

train_linear_classifier(base_model, zorro_train_dataset, zorro_validation_dataset)

base_model_config = {
    "name": "resnet18",
    "input_shape":{
        "width": 128,
        "height": 128,
        "channels": 3
    },
    "fine_tune_from": None,
    "projection_head":{
        "mlp_hidden_size": 1024,
        "projection_size": 512
    },
    "pretrained_weights": True
}
base_model = ResNet(**base_model_config).to(device)

zorro_train_data_augmentation       = transforms.Compose([
        transforms.RandomRotation(degrees=90.0, interpolation=transforms.InterpolationMode.BILINEAR, expand=True),
        transforms.Resize([IMG_SIZE,IMG_SIZE]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) ),
        transforms.ConvertImageDtype(dtype=torch.float32)
    ])

zorro_validation_data_augmentation  = transforms.Compose([
        #transforms.RandomRotation(degrees=90.0, interpolation=transforms.InterpolationMode.BILINEAR, expand=True),
        transforms.Resize([IMG_SIZE,IMG_SIZE]),
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomVerticalFlip(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) ),
        transforms.ConvertImageDtype(dtype=torch.float32)
    ])

zorro_train_dataset       = Zorro(targ_dir = ZORRO_DATA_PATH, transform = zorro_train_data_augmentation,
                        datalist="train.json"
    )

zorro_validation_dataset  = Zorro(targ_dir = ZORRO_DATA_PATH, transform = zorro_validation_data_augmentation,
                        datalist="validation.json"
    )

zorro_eval_dataset        = Zorro(targ_dir = ZORRO_DATA_PATH, transform = zorro_validation_data_augmentation,
                        datalist="validation.json"
    )
#print(summary(pretrained_model.encoder, (3,128,128)))
#print(summary(linear_classificator, (pretrained_model.repr_shape)))
writer = SummaryWriter(log_dir=os.path.join("logs","benchmark", "zorro_imagenet"))
train_linear_classifier(base_model, zorro_train_dataset, zorro_validation_dataset)


"""
eval_loader         = DataLoader(zorro_eval_dataset, batch_size=config['trainer']['batch_size'],
                            num_workers=config['trainer']['num_workers'], drop_last=False, shuffle=True)

zorro_train
zorro_test

robin_train
robin_test

mirabest_train
mirabest_test

train_linear_classifier(model, n_classes, dataset, iteration)
test_linear_classifier(model, n_classes, dataset, iteration)
train_linear_classifier(model, n_classes, dataset, iteration)
test_linear_classifier(model, n_classes, dataset, iteration)
train_linear_classifier(model, n_classes, dataset, iteration)
test_linear_classifier(model, n_classes, dataset, iteration)
"""



