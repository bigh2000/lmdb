from __future__ import print_function
import time
import numpy as np
import pyxis as px

np.random.seed(1234)
nb_samples = 100

X = np.zeros((nb_samples, 254, 254, 3), dtype=np.uint8)
y = np.arange(nb_samples, dtype=np.uint8)
X[10, :, :, 0] = 255

db = px.Writer(dirpath='pyxis_1', map_size_limit=4000, ram_gb_limit=2)
start = time.time()
db.put_samples('X', X, 'y', y)
print('Average time per image = {:.4f}s'.format((time.time() - start)/nb_samples))
db.close()


### Reading Data
try:
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError('Could not import the matplotlib library required to '
                      'plot images. Please refer to http://matplotlib.org/ '
                      'for installation instructions.')

db = px.Reader('pyxis_1')
db.get_data_keys()

for i in range(9, 12):
    sample = db.get_sample(i)
    print('X: ', sample['X'].shape, sample['X'].dtype)
    print('y: ', sample['y'].shape, sample['y'].dtype)

    plt.figure()
    plt.imshow(sample['X'])
    plt.axis('off')
    plt.show()

print(db.get_data_value(9, 'y'))
db.close()