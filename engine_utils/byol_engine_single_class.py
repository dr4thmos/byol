import os
import psutil
import gc
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import transforms
from sklearn.metrics import classification_report, accuracy_score

import numpy as np

from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from general_utils.byol_utils import _create_model_training_folder

class BYOLTrainer:
    def __init__(self, online_network, target_network, linear_classifier, multilabel_linear_classificator, predictor,
                 optimizer, optimizer_classificator, device, logs_folder, exp_name,
                 config_path, transforms_path, main_script_path, pretrained, img_size,
                 recover_from_checkpoint, preview_shape, classes, test_every_n_epochs, **params):
        self.preview_shape = preview_shape
        self.online_network = online_network
        self.target_network = target_network
        self.optimizer = optimizer
        self.device = device
        self.predictor = predictor
        self.max_epochs = params['max_epochs']
        self.linear_classifier = linear_classifier
        self.multilabel_linear_classificator = multilabel_linear_classificator
        self.classification_loss = nn.CrossEntropyLoss()
        self.multilabel_classification_loss = nn.BCEWithLogitsLoss()
        self.optimizer_classificator = optimizer_classificator
        self.log_dir = os.path.join("logs", logs_folder, exp_name)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.m = params['m'] #momentum
        self.batch_size = params['batch_size']
        self.num_workers = params['num_workers']
        self.checkpoint_interval = params['checkpoint_interval']
        self.epoch = 0
        self.start_epoch = 0
        self.classes = classes
        self.class_train_epochs = 10
        self.test_every_n_epochs = test_every_n_epochs
        _create_model_training_folder(self.writer, files_to_same=[config_path, transforms_path, main_script_path])
        
        print("---log folder---")
        print(os.path.join(logs_folder, exp_name))

        if recover_from_checkpoint:
            PATH = os.path.join(self.log_dir, "best_model.pt")
            checkpoint = torch.load(PATH)
            self.online_network.load_state_dict(checkpoint['online_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            #self.start_loss = checkpoint['loss']

    @torch.no_grad()
    def _update_target_network_parameters(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @staticmethod
    def regression_loss(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)

    def _reshape_zipped(self, bw1, bw2):
        for aug_idx, e in enumerate(zip(bw1, bw2)):
            if aug_idx == 0:
                c = torch.stack((e[0],e[1]))
                d = c
            else:
                c = torch.stack((e[0],e[1]))
                d = torch.cat((d, c))
        return d

    def initializes_target_network(self):
        # init momentum network as encoder net
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    def train(self, train_dataset, validation_dataset, eval_dataset):
        min_loss = 1000.0
        max_acc = 0.0
        ### metterlo nell'init
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  num_workers=self.num_workers, drop_last=True, shuffle=True)

        validation_loader = DataLoader(validation_dataset, batch_size=self.batch_size,
                                  num_workers=self.num_workers, drop_last=False, shuffle=True)

        test_loader = DataLoader(eval_dataset, batch_size=self.batch_size,
                                  num_workers=self.num_workers, drop_last=False, shuffle=True)

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        self.initializes_target_network()
        
        for epoch_counter in range(self.start_epoch, self.start_epoch + self.max_epochs):
            self.epoch = epoch_counter
            with tqdm(train_loader, unit="batch") as tepoch:
                tepoch.set_description(f"Epoch {epoch_counter}")
                running_loss = 0.0
                
                grid_done = False # to make only one time plot_grid
                for idx, data in enumerate(tepoch, start=1):
                    (batch_view_1, batch_view_2) = data[0]
                    
                    if not(grid_done):
                        #pair the two views into a grid
                        reshaped = self._reshape_zipped(batch_view_1[:64], batch_view_2[:64])
                        grid = torchvision.utils.make_grid(reshaped)
                        self.writer.add_image('views_paired_train', grid, global_step=epoch_counter)
                        grid_done = True
                        print("Grid done")
                    
                    batch_view_1 = batch_view_1.to(self.device)
                    batch_view_2 = batch_view_2.to(self.device)
                    
                    loss = self.update(batch_view_1, batch_view_2)                

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    self._update_target_network_parameters()  # update the key encoder
                    running_loss += loss.item()
                
                train_loss = running_loss/idx
                
                tepoch.set_postfix(loss=train_loss)
                self.writer.add_scalar('train_loss', train_loss, global_step=epoch_counter)
                running_loss        = 0.0
                running_accuracy    = 0.0
            
            if self.epoch % self.test_every_n_epochs == 0 or self.epoch == 0 or self.epoch == self.max_epochs:
                # Classificator training
                grid_done = False
                for test_epoch in range(self.class_train_epochs):
                    print(f"Test Iteration: {test_epoch}")
                    
                    for idx, data in enumerate(test_loader):
                        batch_views     = data[0].to(self.device)
                        gtruth          = data[1].to(self.device)

                        if not(grid_done):
                            grid = torchvision.utils.make_grid(batch_views[:64])
                            self.writer.add_image('views_paired_downstream', grid, global_step=self.epoch)

                            grid_done = True
                            print("Grid done")

                        with torch.no_grad():
                            features    = self.online_network(batch_views, repr=True)
                    
                        predicted       = self.linear_classifier(features)
                        ## DEBUG HERE
                        loss_class      = self.classification_loss(predicted, gtruth)
                        predicted       = torch.argmax(predicted, axis = 1)
                        #print(loss_class.item())
                        loss_class.backward()
                        #running_loss += loss_class.item()
                        self.optimizer_classificator.step()
                        self.optimizer_classificator.zero_grad()
                        
                        pred_arr = np.round(predicted.float().cpu().detach().numpy())
                        original_arr = gtruth.float().cpu().detach().numpy()
                        if idx == 0:
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
                    print(f"Accuracy at iteration {test_epoch}: {accuracy}")

                self.writer.add_scalars('val_accuracy', {'val_accuracy': accuracy}, global_step=epoch_counter)
                self.linear_classifier.reset_parameters()

                if accuracy > max_acc:
                    max_acc = accuracy
                    self.save_model(os.path.join(self.log_dir, "best_model.pt"))
                
                print("- End of epoch {} - Train Loss: {} - Validation Accuracy: {}".format(epoch_counter, train_loss, max_acc))    
            print("- End of epoch {} - Train Loss: {}".format(epoch_counter, train_loss))

        self.writer.close()
        print("Closed")

    def update(self, batch_view_1, batch_view_2):
        # compute query feature
        predictions_from_view_1 = self.predictor(self.online_network(batch_view_1))
        predictions_from_view_2 = self.predictor(self.online_network(batch_view_2))

        # compute key features
        with torch.no_grad():
            targets_to_view_1 = self.target_network(batch_view_2)
            targets_to_view_2 = self.target_network(batch_view_1)

        loss = self.regression_loss(predictions_from_view_1, targets_to_view_1)
        loss += self.regression_loss(predictions_from_view_2, targets_to_view_2)
        return loss.mean()
    
    def eval_update(self, batch_view_1, batch_view_2):
            
        with torch.no_grad():
            # compute query feature
            predictions_from_view_1 = self.predictor(self.online_network(batch_view_1))
            predictions_from_view_2 = self.predictor(self.online_network(batch_view_2))

            # compute key features
            targets_to_view_1 = self.target_network(batch_view_2)
            targets_to_view_2 = self.target_network(batch_view_1)

        loss = self.regression_loss(predictions_from_view_1, targets_to_view_1)
        loss += self.regression_loss(predictions_from_view_2, targets_to_view_2)
        return loss.mean()
    
    def get_accuracy(self, pred_arr, original_arr):
        pred_arr = pred_arr.cpu().detach().numpy()
        original_arr = original_arr.cpu().detach().numpy()
        final_pred= []

        for i in range(len(pred_arr)):
            final_pred.append(np.argmax(pred_arr[i]))
        final_pred = np.array(final_pred)
        count = 0

        for i in range(len(original_arr)):
            if final_pred[i] == original_arr[i]:
                count+=1
        return count/len(final_pred)*100

    def get_multilabel_accuracy(self, pred_arr, original_arr):  
        pred_arr = np.round(pred_arr.cpu().detach().numpy())
        original_arr = original_arr.cpu().detach().numpy()
        class_rep = classification_report(pred_arr, original_arr, target_names=self.classes, output_dict=True, zero_division=1) # TODO
        """
        final_pred= []

        for i in range(len(pred_arr)):
            final_pred.append(np.argmax(pred_arr[i]))
        final_pred = np.array(final_pred)
        count = 0

        for i in range(len(original_arr)):
            if final_pred[i] == original_arr[i]:
                count+=1
        
        return count/len(final_pred)*100
        """
        return class_rep["weighted avg"]["f1-score"]*100


    def eval_classify_multilabel(self, batch_view, gtruth):
        with torch.no_grad():
            features = self.online_network(batch_view)
        
        predicted = self.multilabel_linear_classificator(features)
        ## DEBUG HERE
        loss = self.multilabel_classification_loss(predicted, gtruth)
        #acc = self.get_multilabel_accuracy(torch.max(predicted,1)[1].float(), gtruth.float())
        acc = self.get_multilabel_accuracy(predicted.float(), gtruth.float())
        return loss, acc

    def eval_classify(self, batch_view, gtruth):
       
        with torch.no_grad():
            features = self.online_network(batch_view)
        
        predicted = self.linear_classifier(features)
        ## DEBUG HERE
        loss = self.classification_loss(predicted, gtruth)
        acc = self.get_accuracy(torch.max(predicted,1)[1].float(), gtruth.float())
        return loss, acc

    def get_features(self, batch_view):
        # compute query feature
        self.online_network.eval()
        with torch.no_grad():
            predictions = self.online_network(batch_view)
            
        return predictions

    def save_model(self, PATH):

        torch.save({
            'online_network_state_dict': self.online_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.epoch
        }, PATH)