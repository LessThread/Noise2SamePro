import os
import numpy as np
from models import Noise2Same

os.environ['CUDA_VISIBLE_DEVICES'] = '0' # Adjust to choose GPU you want to use

def PSNR(gt, img):
    mse = np.mean(np.square(gt - img))
    return 20 * np.log10(255) - 10 * np.log10(mse)

model_dir = 'N2S-3000' # Adjust your model path
data_dir = 'Denoising_data/test/'
model = Noise2Same('trained_models/', model_dir, dim=2, in_channels=1)
groundtruth_data = np.load(data_dir+'bsd68_groundtruth.npy', allow_pickle=True)
test_data = np.load(data_dir+'bsd68_gaussian25.npy', allow_pickle=True)
preds = [model.predict(d.astype('float32')) for d in test_data]
psnrs = [PSNR(preds[idx], groundtruth_data[idx]) for idx in range(len(test_data))]
print(np.array(psnrs).mean())
