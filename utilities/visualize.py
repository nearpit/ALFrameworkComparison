import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle

import numpy as np
import torch 
from sklearn.preprocessing import MinMaxScaler

class Visualize:
    max_dot_size = 100
    marker = MarkerStyle('o', fillstyle = 'none')
    def __init__(self, pool, clf, steps=100):
        self.pool = pool
        self.clf = clf
        x = np.concatenate([split.x for split in pool.data.values()])
        x1_min, x2_min = np.amin(x, axis=0) - np.std(x)
        x1_max, x2_max = np.amax(x, axis=0) + np.std(x)

        x1_span = np.linspace(x1_min, x1_max, steps)
        x2_span = np.linspace(x2_min, x2_max, steps)
        self.x1, self.x2 = np.meshgrid(x1_span, x2_span)
        self.clf_inputs = torch.from_numpy(np.column_stack([self.x1.ravel(), self.x2.ravel()])).float()
    
    def plot_chosen(self, scores, chosen_idx):
        
        rescaled_scores = MinMaxScaler((0, self.max_dot_size)).fit_transform(scores.reshape(-1, 1))
        ax = self.plot_decision_boundary(solo=False)
        x, y = self.pool.get("unlabeled")
        chosen_x, chosen_y = x[chosen_idx], y[chosen_idx]
        x, y, rescaled_scores = np.delete(x, chosen_idx, axis=0), np.delete(y, chosen_idx, axis=0), np.delete(rescaled_scores, chosen_idx, axis=0)
        plt.scatter(x[:, 0], x[:, 1], marker='2', c=y[:, 0], s=rescaled_scores, cmap=plt.cm.coolwarm)
        plt.scatter(chosen_x[0], chosen_x[1], marker='o', facecolor='none', color=plt.cm.coolwarm(chosen_y[0]*255), s=2*self.max_dot_size)
        

        plt.show()

    def plot_decision_boundary(self, solo=True):
        Z = (self.clf(self.clf_inputs)[:, 0]).reshape(self.x1.shape)
        fig, ax = plt.subplots(1, 1)
        ax.contourf(self.x1, self.x2, Z, alpha=0.3, cmap=plt.cm.coolwarm)
        x, y = self.pool.get("labeled")
        ax.scatter(x[:, 0], x[:, 1], marker='^', c=y[:, 0], s=self.max_dot_size, cmap=plt.cm.coolwarm)
        if solo:
            plt.show()
        else:
            return ax

