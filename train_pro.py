import os
import numpy as np
from models import Noise2Same, Noise2SamePro
import tensorflow as tf
import random

os.environ['PYTHONHASHSEED'] = '1'
random.seed(666)
np.random.seed(666)
tf.set_random_seed(666)
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # Adjust to choose GPU you want to use

data_dir = 'Denoising_data/'
X = np.load(data_dir+'train/DCNN400_train_gaussian25.npy')
X_val = np.load(data_dir+'val/DCNN400_validation_gaussian25.npy')
X = np.array([(x - x.mean())/x.std() for x in X])
X_val = np.array([(x - x.mean())/x.std() for x in X_val]).astype('float32')

model_dir = 'N2S_PRO-8000-2' # Set model checkpoints save path
steps = 8000 # Set training steps

sgm_loss = 1 # the default sigma is 1
model = Noise2Same('trained_models/', model_dir, dim=2, in_channels=1, lmbd=2*sgm_loss) 
model.train(X[..., None], patch_size=[64, 64], validation=X_val[..., None], batch_size=64, steps=steps-500)
model = Noise2SamePro('trained_models/', model_dir, dim=2, in_channels=1)
model.train(X[..., None], patch_size=[64, 64], validation=X_val[..., None], batch_size=64, steps=steps)
