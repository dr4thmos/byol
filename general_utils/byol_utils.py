import os
from shutil import copyfile

def _create_model_training_folder(writer, files_to_same):
    model_checkpoints_folder = os.path.join(writer.log_dir, 'checkpoints')
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        for file in files_to_same:
            os.makedirs(os.path.join(model_checkpoints_folder, os.path.split(file)[0]))
            if not(os.path.exists(os.path.join(model_checkpoints_folder, os.path.split(file)[0]))):
                copyfile(file, os.path.join(model_checkpoints_folder, os.path.split(file)[0]))