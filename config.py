import os

# DIRECTORIES
model_dir = '../models'
data_dir = '../data'
    
# PATHS
train_path = os.path.join(data_dir, 'train.csv')
#test_path = os.path.join(data_dir, 'test_new_nodress.csv')
#val_path = os.path.join(data_dir, 'val_new_nodress.csv')
    
log_path = os.path.join(model_dir, 'train.log')
model_path = os.path.join(model_dir, 'ep{epoch:02d}-val_loss{val_loss:.2f}')
    
# HYPERPARAMETERS
target_size = (320, 320)
    
# OTHERS
categories = ['all', 'dress', 'pants', 'skirt', 'tops']

