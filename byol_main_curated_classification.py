import warnings
import os
import torch
import yaml
from astropy.io import fits
from data_utils.transforms import get_data_transforms_curated, get_data_transforms_eval_curated, get_data_transforms_test_robin
from data_utils.multi_view_data_injector import MultiViewDataInjector
from data_utils.curated import Curated
from data_utils.robin import Robin
from model_utils.mlp_head import MLPHead
from model_utils.resnet_base_network import ResNet
from engine_utils.byol_engine_single_class import BYOLTrainer
from general_utils.parser import parse
from astropy.utils.exceptions import AstropyWarning

warnings.simplefilter('ignore', category=AstropyWarning)

#TORUN: python byol/byol_main_curated_classification.py -f curated -e test1 -epcs 100 -t 10 curato.yaml

def main():
    #torch.manual_seed(0)

    args = parse()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Training with: {device}")
    print(f"Number of epochs:{args.epochs}")

    src_dir             = "byol"
    config_dir          = os.path.join( src_dir, "configs")
    config_path         = os.path.join( config_dir, args.current_config )
    transforms_path     = os.path.join( src_dir, "data_utils", "transforms.py" ) # To be copied by summarywriter in logs
    main_script_path    = os.path.join( src_dir, "byol_main_curated_classification.py" ) # To be copied by summarywriter in logs    
    
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

    data_transform_train    = get_data_transforms_curated(**config['network']['input_shape'])
    data_transform_val      = get_data_transforms_eval_curated(**config['network'])
    data_transform_test     = get_data_transforms_test_robin(**config['network']['input_shape']) # for Robin

    data_path   = config["dataset"]

    train_dataset       = Curated(
        targ_dir = data_path,
        transform = MultiViewDataInjector([data_transform_train, data_transform_train]),
        datalist = "train.json"
    )
    validation_dataset  = Curated(
        targ_dir = data_path,
        transform = MultiViewDataInjector([data_transform_val, data_transform_val]),
        datalist = "validation.json"
    )
    test_dataset  = Robin(
        targ_dir = "2-ROBIN",
        transform = data_transform_test,
        datalist = "info_wo_meerkat.json"
    )
    
    #train_dataset = torch.utils.data.Subset(train_dataset, range(1,200000))

    print(f"Train dataset: {len(train_dataset)}")
    print(f"Validation dataset: {len(validation_dataset)}")
    print(f"Test dataset: {len(test_dataset)}")
    
    # online network
    online_network = ResNet(**config['network']).to(device)
    target_network = ResNet(**config['network']).to(device)

    predictor = MLPHead(in_channels=online_network.projetion.net[-1].out_features,
                        **config['network']['projection_head']).to(device)
    """
    optimizer = torch.optim.SGD(list(online_network.parameters()) + list(predictor.parameters()),
                                **config['optimizer']['params'])
    """
    optimizer_params = {"lr": 0.0003}
    optimizer = torch.optim.Adam(list(online_network.parameters()) + list(predictor.parameters()),
                                **optimizer_params)
    
    config["trainer"]["max_epochs"] = int(args.epochs)
    linear_classificator    = torch.nn.Linear(
        online_network.repr_shape, test_dataset.num_classes).to(device)
    optimizer_classificator =   torch.optim.SGD(list(linear_classificator.parameters()),
                                **config['optimizer']['params'])
        
    if args.from_checkpoint:
        recover_from_checkpoint = True
    else:
        recover_from_checkpoint = False

    trainer = BYOLTrainer(
        online_network=online_network,
        target_network=target_network,
        optimizer=optimizer,
        predictor=predictor,
        linear_classifier=linear_classificator,
        optimizer_classificator=optimizer_classificator,
        classes = test_dataset.num_classes,
        multilabel_linear_classificator = None,
        device=device,
        logs_folder=args.folder,
        exp_name=args.experiment,
        config_path=config_path,
        transforms_path=transforms_path,
        main_script_path = main_script_path,
        pretrained=config["network"]["pretrained_weights"],
        img_size = config["network"]["input_shape"]["width"],
        recover_from_checkpoint = recover_from_checkpoint,
        preview_shape = config["network"]["preview_shape"],
        
        optimizer_params = config['optimizer']['params'],
        test_every_n_epochs=int(args.test_every_n_epochs),
        **config["trainer"]
    )

    trainer.train(train_dataset, validation_dataset, test_dataset)

if __name__ == '__main__':
    main()