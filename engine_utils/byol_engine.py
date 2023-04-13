import os
import psutil
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import transforms
from sklearn.metrics import classification_report

import numpy as np

from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from general_utils.byol_utils import _create_model_training_folder

class BYOLTrainer:
    def __init__(self, online_network, target_network, linear_classifier, multilabel_linear_classificator, predictor,
                 optimizer, optimizer_classificator, device, logs_folder, exp_name,
                 config_path, transforms_path, main_script_path, pretrained, img_size,
                 recover_from_checkpoint, preview_shape, classes, **params):
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
        self.multilabel_classification_loss = nn.BCELoss()
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

        eval_loader = DataLoader(eval_dataset, batch_size=self.batch_size,
                                  num_workers=self.num_workers, drop_last=True, shuffle=True)

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        self.initializes_target_network()
        
        for epoch_counter in range(self.start_epoch, self.start_epoch + self.max_epochs):
            self.epoch = epoch_counter
            print("Epoch")
            print(epoch_counter)
            running_loss = 0.0
            
            done = False # to make only one time plot_grid

            for idx, data in enumerate(train_loader):
                print("Train Iteration")
                print(idx)
                (batch_view_1, batch_view_2) = data[0]
                
                if not(done):
                    #pair the two views into a grid
                    for idx, e in enumerate(zip(batch_view_1[:64], batch_view_2[:64])):
                        if idx == 0:
                            c = torch.stack((e[0],e[1]))
                            d = c
                        else:
                            c = torch.stack((e[0],e[1]))
                            d = torch.cat((d, c))
                    
                    grid = torchvision.utils.make_grid(d[:64])
                    self.writer.add_image('views_paired', grid, global_step=epoch_counter)

                    done = True
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
            self.writer.add_scalar('train_loss', train_loss, global_step=epoch_counter)
            running_loss = 0.0
            running_accuracy = 0.0
            
            for epoch in range(10):
                print("Val-Classification Iteration")
                print(epoch)
                for idx, data in enumerate(eval_loader):
                    batch_view = data[0].to(self.device)
                    gtruth = data[1].to(self.device)
                    loss_class, accuracy = self.eval_classify_multilabel(batch_view, gtruth) # calcolare fuori dal loop l'accuracy, per avere tutte le classi
                    loss_class.backward()
                    running_loss += loss_class.item()
                    running_accuracy += accuracy
                    self.optimizer_classificator.zero_grad()
                    self.optimizer_classificator.step()
                
                accuracy = running_accuracy/idx
                print(accuracy)
                val_loss = running_loss/idx
                running_loss = 0.0
                running_accuracy = 0.0
            
            self.writer.add_scalars('val_accuracy', {'val_accuracy': accuracy}, global_step=epoch_counter)
            self.linear_classifier.reset_parameters()
            
            """
            # Sostituire con accuracy
            for idx, data in enumerate(validation_loader):
                print("Val Iteration")
                print(idx)
                (batch_view_1, batch_view_2) = data[0]
                batch_view_1 = batch_view_1.to(self.device)
                batch_view_2 = batch_view_2.to(self.device)
                
                loss = self.eval_update(batch_view_1, batch_view_2)
                running_loss += loss.item()

            val_loss = running_loss/idx
            #self.writer.add_scalar('val_loss', val_loss, global_step=epoch_counter)
            
            self.writer.add_scalars('loss', {'train_loss': train_loss,
                                    'val_loss': val_loss
            }, global_step=epoch_counter)
            """
            """
            if loss < min_loss:
                min_loss = loss
                self.save_model(os.path.join(self.log_dir, "best_model.pt"))
            """
            if accuracy > max_acc:
                max_acc = accuracy
                self.save_model(os.path.join(self.log_dir, "best_model.pt"))
            
            print("- End of epoch {} - Train Loss: {} - Validation Accuracy: {}, Validation classification Loss: {}".format(epoch_counter, train_loss, accuracy, val_loss))

        #print(torch.cuda.memory_allocated())
        """
        torch.cuda.empty_cache()
        del self.target_network
        del batch_view_1
        del batch_view_2
        del train_dataset
        del validation_dataset
        del train_loader
        del validation_loader
        del grid
        gc.collect()

        limit = 8192 # projector value
        n_images = int(np.floor(limit/ self.preview_shape))
        n_images *= n_images

        print(n_images)

        self.online_network.to("cuda")
        resize = transforms.Resize(self.preview_shape)
        for idx, data in enumerate(eval_loader):
            #torch.cuda.empty_cache()
            print("embedding creations")
            print(idx)
            batch_view = data[0].to("cuda")
            
            with torch.no_grad():
                features_batch = self.online_network(batch_view)

            if idx == 0:                
                features = features_batch.to("cpu")
                batch_view = resize(batch_view)
                imgs = batch_view.to("cpu")
                print("done")
                #metadata = data
            else:
                if len(features) > n_images:
                    self.writer.add_embedding(
                        features[:n_images],
                        #metadata=all_labels[:2000],
                        #metadata=list(zip(metadata[:4000].tolist(), list(np.zeros(4000, dtype=int)))),
                        metadata=list(zip(list(np.zeros(n_images, dtype=int)), list(np.zeros(n_images, dtype=int)))),
                        label_img=imgs[:n_images],
                        global_step=epoch_counter,
                        metadata_header=['source_type', 'new_labels'],
                        tag="to_idx_"+str(idx*len(batch_view))
                    )
                    print("-- Embedding salvati --")
                    print("-- Nuova trance --")
                    del features
                    del imgs
                    gc.collect()
                    features = features_batch.to("cpu")
                    batch_view = resize(batch_view)
                    imgs = batch_view.to("cpu")
                else:
                    features = torch.cat((features, features_batch.to("cpu")), 0)
                    batch_view = resize(batch_view)
                    imgs = torch.cat((imgs, batch_view.to("cpu")), 0)
            
            del features_batch
            del batch_view
            gc.collect()
        """
        """
        # get the class labels for each image
        class_labels = [classes[lab] for lab in labels]
        """
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