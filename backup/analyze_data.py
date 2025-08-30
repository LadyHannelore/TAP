import numpy as np
import os

# Analyze the data files
files = ['sg_t16_train.npy', 'sg_t16_test.npy', 'sg_t16_validation.npy']

for file in files:
    data = np.load(f'data/{file}')
    file_size = os.path.getsize(f'data/{file}')
    print(f'{file}:')
    print(f'  Shape: {data.shape}')
    print(f'  Data type: {data.dtype}')
    print(f'  File size: {file_size} bytes')
    print(f'  Min value: {data.min()}')
    print(f'  Max value: {data.max()}')
    print(f'  First 10 values: {data[:10]}')
    print()
