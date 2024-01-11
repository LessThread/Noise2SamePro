import os 
import numpy as np
import matplotlib.pyplot as plt
from models import Noise2Same
from PIL import Image

os.environ['CUDA_VISIBLE_DEVICES'] = '2' # Adjust to choose GPU you want to use

def test_single(png_file_path, model_dir, save_path):
    image = Image.open(png_file_path).convert('L')
    image_array = np.array(image)
    model = Noise2Same('trained_models/', model_dir, dim=2, in_channels=1)
    denoised_image = model.predict(image_array.astype('float32'))
    denoised_image_pil = Image.fromarray(np.uint8(denoised_image))
    denoised_image_pil.save(save_path)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image_array, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(denoised_image, cmap='gray')
    plt.title('Denoised Image')
    plt.axis('off')
    plt.show()

picture = 'man/' # Adjust path of the picture you want to test
model_dir = 'N2S_PRO' # Adjust your model path
test_single('test_single/' + picture + 'original_image.png', model_dir, 'test_single/' + picture + 'denoised_image.png')
