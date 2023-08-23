import warnings
import os
import torch
import torchvision
import numpy as np
import yaml
from data_utils.transforms_robin import get_data_transforms_train_robin, get_data_transforms_test_robin 
from data_utils.robin import Robin
from model_utils.mlp_head import MLPHead
from model_utils.resnet_base_network_classification import ResNet
from general_utils.parser import parse
from astropy.utils.exceptions import AstropyWarning
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report, accuracy_score
#from sklearn.model_selection import KFold

warnings.simplefilter('ignore', category=AstropyWarning)

#TORUN: python byol/byol_main_curated_classification.py -f curated -e test1 -epcs 100 -t 10 curato.yaml

def main():
    
    args = parse()
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    
    print(f"Training with: {device}")
    print(f"Number of epochs:{args.epochs}")

    src_dir             = "byol"
    config_dir          = os.path.join( src_dir, "configs")
    config_path         = os.path.join( config_dir, args.current_config )
    transforms_path     = os.path.join( src_dir, "data_utils", "transforms.py" ) # To be copied by summarywriter in logs
    main_script_path    = os.path.join( src_dir, "class_robin_supervised.py" ) # To be copied by summarywriter in logs    
    
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

    data_transform_train    = get_data_transforms_train_robin(**config['network']['input_shape'])
    data_transform_val      = get_data_transforms_test_robin(**config['network']['input_shape'])

    #kf = KFold(n_splits=5)

    data_path   = config["dataset"]

    dataset       = Robin(
        targ_dir = "2-ROBIN",
        transform = data_transform_train,
        datalist = "info_wo_meerkat.json"
    )
    random_split = False
    if random_split:
        split = 0.7

        len_train = int(len(dataset) * split)
        len_val = int(len(dataset) - len_train)

        train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [len_train, len_val])
    else:
        train_dataset       = Robin(
            targ_dir = "2-ROBIN",
            transform = data_transform_train,
            datalist = "info_wo_meerkat_train_07.json"
        )
        validation_dataset       = Robin(
            targ_dir = "2-ROBIN",
            transform = data_transform_train,
            datalist = "info_wo_meerkat_test_03.json"
        )
        

    print(f"Train dataset: {len(train_dataset)}")
    print(f"Validation dataset: {len(validation_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=config["trainer"]["batch_size"],
                                  num_workers=config["trainer"]["num_workers"], drop_last=True, shuffle=True)

    validation_loader = DataLoader(validation_dataset, batch_size=config["trainer"]["batch_size"],
                                num_workers=config["trainer"]["num_workers"], drop_last=True, shuffle=True)

    # online network
    backbone = ResNet(**config['network']).to(device)
    #checkpoint = torch.load("logs/curated/adam_0003_1001/best_model.pt")
    checkpoint = torch.load("logs/curated/adam_003_501/best_model.pt")
    backbone.load_state_dict(checkpoint["online_network_state_dict"])
    linear_classificator    = torch.nn.Linear(
        backbone.repr_shape, dataset.num_classes).to(device)
    optimizer_classificator =   torch.optim.SGD(list(linear_classificator.parameters()),
                                **config['optimizer']['params'])
    classification_loss = torch.nn.CrossEntropyLoss()

    writer = SummaryWriter(log_dir=os.path.join("logs", args.folder, args.experiment))

    max_acc = 0.
    grid_done_train = False
    grid_done_validation = False
    for epoch in range(int(args.epochs)):
        print(f"Epoch: {epoch}")
        
        for train_idx, data in enumerate(train_loader):
            batch_view      = data[0].to(device)
            gtruth          = data[1].to(device)

            if not(grid_done_train):
                grid = torchvision.utils.make_grid(batch_view[:64])
                writer.add_image('train_view', grid, global_step=epoch)
                grid_done_train = True
                print("Grid done")

            with torch.no_grad():
                features    = backbone(batch_view, repr=True)
            predicted   = linear_classificator(features)

            loss_class  = classification_loss(predicted, gtruth)
            predicted   = torch.argmax(predicted, axis = 1)
            loss_class.backward()
            optimizer_classificator.step()
            optimizer_classificator.zero_grad()
            
            pred_arr = np.round(predicted.float().cpu().detach().numpy())
            original_arr = gtruth.float().cpu().detach().numpy()
            if train_idx == 0:
                pred_arr_concatenated = pred_arr
                original_arr_concatenated = original_arr
            else:
                pred_arr_concatenated = np.concatenate((pred_arr_concatenated, pred_arr), axis=0)
                original_arr_concatenated = np.concatenate((original_arr_concatenated, original_arr), axis=0)

        #print(len(pred_arr_concatenated))
        class_rep = accuracy_score(original_arr_concatenated, pred_arr_concatenated) # TODO
        #print(class_rep)
        
        #print(original_arr_concatenated[:10])
        #print(pred_arr_concatenated[:10])
        
        accuracy = class_rep
        print(f"Train Accuracy at iteration {epoch}: {accuracy}")

        writer.add_scalars('accuracy', {'train_accuracy': accuracy}, global_step=epoch)
    
        for val_idx, data in enumerate(validation_loader):
            batch_view      = data[0].to(device)
            gtruth          = data[1].to(device)

            if not(grid_done_validation):
                grid = torchvision.utils.make_grid(batch_view[:64])
                writer.add_image('validation_view', grid, global_step=epoch)
                grid_done_validation = True
                print("Grid done")

            with torch.no_grad():
                features    = backbone(batch_view, repr=True)
                predicted   = linear_classificator(features)
            
            predicted   = torch.argmax(predicted, axis = 1)

            pred_arr = np.round(predicted.float().cpu().detach().numpy())
            original_arr = gtruth.float().cpu().detach().numpy()
            if val_idx == 0:
                pred_arr_concatenated = pred_arr
                original_arr_concatenated = original_arr
            else:
                pred_arr_concatenated = np.concatenate((pred_arr_concatenated, pred_arr), axis=0)
                original_arr_concatenated = np.concatenate((original_arr_concatenated, original_arr), axis=0)

        #print(len(pred_arr_concatenated))
        class_rep = accuracy_score(original_arr_concatenated, pred_arr_concatenated) # TODO
        #print(class_rep)
        
        #print(original_arr_concatenated[:10])
        #print(pred_arr_concatenated[:10])
        
        accuracy = class_rep
        print(f"Val Accuracy at iteration {epoch}: {accuracy}")
        writer.add_scalars('accuracy', {'val_accuracy': accuracy}, global_step=epoch)

    """
    if accuracy > max_acc:}
        max_acc = accuracy
        torch.save({
            'backbone': backbone.state_dict(),
            'epoch': train_epoch
        }, os.path.join(log_dir, "best_model.pt"))
    """
    #print("- End of epoch {} - Train Loss: {} - Train Accuracy: {}".format(train_epoch, "NaN", accuracy))    

if __name__ == '__main__':
    main()