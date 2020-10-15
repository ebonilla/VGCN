import numpy as np


class BatchHandler:

    def __init__(self, num_examples, shuffle=True):
        self._num_examples = num_examples
        self.perm = np.arange(self._num_examples)
        if shuffle:
            np.random.shuffle(self.perm)
        self._epochs_completed = 0
        self._index_in_epoch = 0

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if (
            self._index_in_epoch > self._num_examples
            ) and (
                start != self._num_examples):
            self._index_in_epoch = self._num_examples
        if self._index_in_epoch > self._num_examples:   # Finished epoch
            self._epochs_completed += 1
            self.perm = np.arange(self._num_examples)
            np.random.shuffle(self.perm)                  # Shuffle the data
            start = 0                               # Start next epoch
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        # print("start=" + str(start) + " end=" + str(end))
        return self.perm[np.arange(start, end)]

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed


if __name__ == "__main__":
    N = 100
    X = np.random.normal(size=(N, 2))
    bh = BatchHandler(num_examples=N)
    b_size = 10

    epochs = 3
    i = 0
    last_epoch = bh.epochs_completed
    print(bh.epochs_completed)
    idx = bh.next_batch(batch_size=b_size)
    while bh.epochs_completed < epochs:
        if bh.epochs_completed != last_epoch:
            last_epoch = bh.epochs_completed
            print(bh.epochs_completed)
        idx = bh.next_batch(batch_size=b_size)


