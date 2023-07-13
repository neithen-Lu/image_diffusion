import subprocess
from subprocess import STDOUT
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

subprocess.run([
    'python', 'fid_kid.py',
    '/home/qindafei/KX/data/cifar',
    '/home/qindafei/KX/image_diffusion/baseline/samples_10000x32x32x3.npz',
    '--num_samples', '10000',
    '--batch_size', '100'
], stderr=STDOUT)