from __future__ import print_function
import numpy as np
import pyxis as px

rng = np.random.RandomState(1234)
nb_samples = 10

X = rng.rand(nb_samples, 254, 254, 3)
y = np.arange(nb_samples, dtype=np.uint8)

db = px.Writer(dirpath='pyxis_2', map_size_limit=100, ram_gb_limit=1)
db.put_samples('X', X, 'y', y)
db.close()

db = px.Reader('pyxis_2')
gen = px.SimpleBatch(db, keys=('X', 'y'), batch_size=5, shuffle=False)

for i in range(4):
    xs, ys = next(gen)
    print()
    print('Iteration:', i, '\tTargets:', ys)
    if gen.end_of_dataset:
        print('We have reached the end of the dataset')

gen = px.SimpleBatch(db, keys=('X', 'y'), batch_size=3, shuffle=False)

for i in range(6):
    xs, ys = next(gen)
    print()
    print('Iteration:', i, '\tTargets:', ys)
    if gen.end_of_dataset:
        print('We have reached the end of the dataset')

gen = px.SimpleBatch(db, keys=('y'), batch_size=5, shuffle=True, rng=rng)

for i in range(6):
    ys = next(gen)
    print()
    print('Iteration:', i, '\tTargets:', ys)
    if gen.end_of_dataset:
        print('We have reached the end of the dataset')

gen = px.StochasticBatch(db, keys=('y'), batch_size=5, rng=rng)

for i in range(10):
    ys = next(gen)
    print('Iteration:', i, '\tTargets:', ys)

gen = px.SequentialBatch(db, keys=('y'), batch_size=3)

for i in range(10):
    ys = next(gen)
    print('Iteration:', i, '\tTargets:', ys)

class SquareTargets(px.SimpleBatchThreadSafe):
    def __init__(self, db, keys, batch_size):
        super(SquareTargets, self).__init__(db, keys, batch_size, shuffle=False, endless=False)

    def __next__(self):
        with self.lock:
            X, y = next(self.gen)
        y = y ** 2
        return X, y

gen = SquareTargets(db, keys=('X', 'y'), batch_size=2)

print('Squared targets:')
for _, y in gen:
    print(y)

db.close()