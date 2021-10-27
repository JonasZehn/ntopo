
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


class SimulationMonitor:
    def __init__(self, n_iterations):
        self.n_iterations = n_iterations

        self.iter = None
        self.values = None

    def __iter__(self):
        self.iter = -1
        self.values = []
        return self

    def __next__(self):
        self.iter += 1
        if self.iter >= self.n_iterations:
            raise StopIteration()
        return self.iter

    def monitor(self, loss):
        assert len(loss) == 1
        loss = tf.reshape(loss, (-1, )).numpy()[0]

        self.values.append(loss)

    def save_plot(self, save_path, prefix, postfix):
        start = 0
        end = len(self.values)
        if end > 100:
            start = 50
        x = np.arange(start, end)
        y = self.values[start:end]
        fig, _ = plt.subplots()
        plt.plot(x, y)
        plt.axis([min(x), max(x), min(y), max(y)])
        fig.savefig(os.path.join(
            save_path, prefix + 'loss' + postfix + '.png'))
        plt.close(fig)
