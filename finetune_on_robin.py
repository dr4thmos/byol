import warnings
import yaml
import os
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torch
print(torch.__version__)
torch.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Training with: {device}")

from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

from general_utils.parser import create_parser
from data_utils.transforms import get_data_transforms_zorro, get_data_transforms_zorro_eval
from data_utils.robin import Robin
from model_utils.resnet_base_network import ResNet
from model_utils.finetune_encoder import FinetuneEncoder
from torch.utils.data.dataloader import DataLoader
from torchmetrics import Accuracy
#from model_utils.linear_classifier import LinearClassifier

# TORUN: python byol/finetune_on_zorro.py -f log-folder-name -e log-subfodler-name --epochs 1 config.yaml

parser = create_parser()
args = parser.parse_args()

data_dir = "."
src_dir = "byol"
config_dir = "configs"

config_path = os.path.join(src_dir, config_dir, args.current_config)
transforms_path = os.path.join(src_dir, "data_utils", "transforms.py") # To be copied by summarywriter in logs
main_script_path = os.path.join(src_dir, "finetune_on_zorro.py") # To be copied by summarywriter in logs

config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

writer = SummaryWriter(log_dir=os.path.join("logs",args.folder, args.experiment))

model_checkpoints_folder = os.path.join(writer.log_dir, 'checkpoints')

def main():
    
    data_transform      = get_data_transforms_zorro(**config['network']['input_shape'])
    data_transform_eval = get_data_transforms_zorro_eval(**config['network']['input_shape'])

    data_path           = os.path.join(data_dir, config["dataset"])

    dataset       = Robin(targ_dir = data_path, transform = data_transform)
    num_classes = dataset.num_classes
    train_split = int(0.8 * len(dataset))
    val_split  = len(dataset) - train_split
    
    train_dataset, validation_dataset  = torch.utils.data.random_split(dataset, [train_split, val_split])

    train_loader        = DataLoader(train_dataset, batch_size=config['trainer']['batch_size'],
                                  num_workers=config['trainer']['num_workers'], drop_last=True, shuffle=True)

    validation_loader   = DataLoader(validation_dataset, batch_size=config['trainer']['batch_size'],
                                num_workers=config['trainer']['num_workers'], drop_last=False, shuffle=True)

    print(len(train_dataset))
    print(len(validation_dataset))
    
    pretrained_model = ResNet(**config['network']).to(device)

    #checkpoint = torch.load(config["network"]["fine_tune_from"])
    #pretrained_model.load_state_dict(checkpoint['online_network_state_dict'])

    linear_classificator = torch.nn.Linear(
        pretrained_model.repr_shape, num_classes).to(device)

    #print(summary(pretrained_model.encoder, (3,128,128)))
    #print(summary(linear_classificator, (pretrained_model.repr_shape)))

    finetuning_model = FinetuneEncoder(pretrained_model.encoder, linear_classificator).to(device)

    #print(summary(finetuning_model, (3,128,128)))
    
    optimizer = torch.optim.SGD(list(finetuning_model.parameters()),
                                **config['optimizer']['params'])

    criterion = torch.nn.CrossEntropyLoss()
    
    accuracy = Accuracy(task="multiclass", num_classes=num_classes, top_k=1).to(device)

    min_val_loss = 100.0

    for epoch_counter in range(int(args.epochs)):
        print("Epoch")
        print(epoch_counter)
        running_loss = 0.0
        running_accuracy = 0.0
        running_val_loss = 0.0
        running_val_accuracy = 0.0
        
        grid_done = False

        finetuning_model.train()
        for idx, data in enumerate(train_loader):
            image_batch = data[0].to(device)
            if not(grid_done):
                grid_done = plot_grid(image_batch, writer, epoch_counter)
            #print(image_batch.shape)
            predicted = finetuning_model(image_batch)
            
            loss = criterion(predicted, data[1].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            acc = accuracy(predicted, data[1].to(device))
            running_accuracy += acc

        finetuning_model.eval()
        for idx, data in enumerate(validation_loader):
            image_batch = data[0].to(device)
            if not(grid_done):
                grid_done = plot_grid(image_batch, writer, epoch_counter)
            #print(image_batch.shape)
            with torch.no_grad():
                predicted = finetuning_model(image_batch)
            
            loss = criterion(predicted, data[1].to(device))
            running_val_loss += loss.item()
            
            acc = accuracy(predicted, data[1].to(device))
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
            }, os.path.join(writer.log_dir, "best_model.pt"))
        
def plot_grid(data, writer, epoch_counter):
    grid = torchvision.utils.make_grid(data[:64])
    writer.add_image('views_paired', grid, global_step=epoch_counter)
    print("Grid done")
    return True

if __name__ == '__main__':
    main()