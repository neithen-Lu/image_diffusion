import numpy as np
data = np.load('/home/qindafei/KX/image_diffusion/baseline/samples_10000x32x32x3.npz')
data = data['arr_0']
from PIL import Image

w, h = 32,32
# data = np.zeros((h, w, 3), dtype=np.uint8)
# data[0:256, 0:256] = [255, 0, 0] # red patch in upper left
for i in range(10000):
    img = Image.fromarray(data[i,:,:,:], 'RGB')
    img.save(f'visualize/{i}.png')