import numpy as np
image_size = 32
data_sum = np.zeros((40000,image_size,image_size,3))
for i in range(4):
    data = np.load(f'/home/qindafei/KX/image_diffusion/result/baseline/samples_{i}_10000x32x32x3.npz')
    data_sum[i*10000:(i+1)*10000,:,:,:] = data['arr_0']
shape_str = "x".join([str(x) for x in data_sum.shape])
np.savez(f'/home/qindafei/KX/image_diffusion/result/baseline/samples_{shape_str}.npz', data_sum)

data = np.load('/home/qindafei/KX/image_diffusion/tmp/samples_0_10000x32x32x3.npz')
data = data['arr_0']
from PIL import Image

w, h = 32,32
# data = np.zeros((h, w, 3), dtype=np.uint8)
# data[0:256, 0:256] = [255, 0, 0] # red patch in upper left
for i in range(10000):
    img = Image.fromarray(data[i,:,:,:], 'RGB')
    img.save(f'visualize/{i}.png')