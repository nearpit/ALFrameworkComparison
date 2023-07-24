import numpy as np
import torch 

import matplotlib 

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D 
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patches as mpatches


import utilities


class Visualize:
    markers = {
            "test": "*",
            "new labeled": "x",
            "unviolated labeled": "^",
            "unlabeled": "2",
            "next added": "o"
        }
    max_dot_size = 150
    fontsize='medium'
    contour_levels = 10
    test_perf = np.array([])
    markersize = 14
    plt.rcParams.update({'font.size': 18})




    def __init__(self, pool, clf, acq, total_budget, steps=100):
        self.pool = pool
        self.clf = clf
        self.acq = acq
        self.total_budget = total_budget

        x = np.concatenate([split.x for split in pool.data.values()])
        x1_min, x2_min = np.amin(x, axis=0) - np.std(x)
        x1_max, x2_max = np.amax(x, axis=0) + np.std(x)

        x1_span = np.linspace(x1_min, x1_max, steps)
        x2_span = np.linspace(x2_min, x2_max, steps)
        self.x1, self.x2 = np.meshgrid(x1_span, x2_span)
        self.clf_inputs = torch.from_numpy(np.column_stack([self.x1.ravel(), self.x2.ravel()])).float()
       

    
    def compute_clf_grad(self):
        self.clf_grad = (self.clf(self.clf_inputs)[:, 1]).cpu().reshape(self.x1.shape)

    def clf_train(self, ax, train_perf, val_perf):
        ax.contourf(self.x1, self.x2, self.clf_grad, levels=self.contour_levels, alpha=0.3, cmap=plt.cm.coolwarm, antialiased=True)
        x, y = self.pool.get("unlabeled")
        ax.scatter(x[:, 0], x[:, 1], marker='2', c='grey', alpha=0.4, s=self.max_dot_size)

        x, y = self.pool.get("unviolated")
        ax.scatter(x[:, 0], x[:, 1], marker=self.markers["unviolated labeled"], c=y.argmax(axis=-1), s=self.max_dot_size, cmap=plt.cm.coolwarm)
        x, y = self.pool.get("new_labeled")

        color = plt.cm.coolwarm(y.argmax(axis=-1)*255)
        ax.scatter(x[:, 0], x[:, 1], marker=self.markers["new labeled"], color=color, s=self.max_dot_size)

        ax.set_title("Classifier")
        performance_string = f"Train Acc {train_perf[1]['MulticlassAccuracy']:.1%}\nVal Acc {val_perf[1]['MulticlassAccuracy']:.1%}"
        ax.annotate(performance_string, xy=(0.03, 0.97), xycoords='axes fraction',
                    ha='left', va='top',
                    bbox=dict(boxstyle='round', fc='w'))

    

    def clf_eval(self, ax, perf, split="test"):

        ax.contourf(self.x1, self.x2, self.clf_grad, levels=self.contour_levels, alpha=0.3, cmap=plt.cm.coolwarm, antialiased=True)

        x, y = self.pool.get(split)
        ax.scatter(x[:, 0], x[:, 1], marker=self.markers[split], alpha=0.5, c=y.argmax(axis=-1), s=self.max_dot_size, cmap=plt.cm.coolwarm)
        
        ax.set_title(f"Classifier {split.capitalize()}")
        performance_string = f"Test Acc {perf[1]['MulticlassAccuracy']:.1%}"
        ax.annotate(performance_string, xy=(0.03, 0.97), xycoords='axes fraction',
                    ha='left', va='top',
                    bbox=dict(boxstyle='round', fc='w'))          
        
    def acq_boundary(self, ax, chosen_idx):
        
        if chosen_idx:
            Z = (self.acq.get_scores(self.clf_inputs)).reshape(self.x1.shape)
            ax.contourf(self.x1, self.x2, Z, levels=self.contour_levels, cmap=plt.cm.binary, alpha=0.3, antialiased=True)
            chosen_x, chosen_y = self.pool[chosen_idx]
            ax.scatter(chosen_x[0], chosen_x[1], marker=self.markers["next added"], linewidths=2, facecolor=plt.cm.coolwarm(chosen_y[1]*255), color="black", s=2*self.max_dot_size)

        x, y = self.pool.get("unlabeled")
        ax.scatter(x[:, 0], x[:, 1], marker=self.markers["unlabeled"], c=y.argmax(axis=-1), s=self.max_dot_size, cmap=plt.cm.coolwarm)
        ax.set_title("Acquisition")
       

    def plot_test_curve(self, ax, keep_every=2):
        current_len = len(self.test_perf)
        ax.plot(np.arange(0, current_len, 1), self.test_perf, c='black')
        ax.set_xticks(np.arange(0, self.total_budget+1, 10))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

        [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % keep_every != 0]
        ax.set_title("Test Performance")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("CE Loss")
        ax.grid(alpha=0.7)


    def acq_colorbar(self, ax):

        cb = plt.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0, 1), 
                                                       cmap=plt.cm.binary), 
                          cax=ax,
                          orientation='horizontal')
        cb.set_ticks([0, 1])
        cb.ax.tick_params(size=0)
        cb.set_ticklabels(["Min", "Max"], fontsize=self.fontsize)
        cb.set_label('Score', rotation=0, fontsize=self.fontsize, labelpad=-70)  

    
    def clf_colorbar(self, ax):

        cb_boundary = plt.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0, 1), 
                                                                cmap=plt.cm.coolwarm), 
                                   cax=ax,
                                   orientation='horizontal')
        cb_boundary.set_ticks([0, 1])
        cb_boundary.ax.tick_params(size=0)
        cb_boundary.set_ticklabels(["Class A", "Class B"], fontsize=self.fontsize)


    def make_plots(self, args, iteration, train_perf, val_perf, test_perf, path_to_store, chosen_idx=None):
        self.test_perf = np.append(self.test_perf, test_perf[0]).astype(float)
        fig, ax = plt.subplots(3, 2, figsize=(20, 20), gridspec_kw={'height_ratios': [1, 20, 20]})
        
        self.compute_clf_grad()
        self.acq_colorbar(ax[0, 0])
        self.clf_colorbar(ax[0, 1])
        self.acq_boundary(ax[1, 0], chosen_idx)
        self.clf_train(ax[1, 1], train_perf, val_perf)
        self.plot_test_curve(ax[2, 0])
        self.clf_eval(ax[2, 1], test_perf, split="test")

        title = plt.suptitle(f"{str(args.algorithm).capitalize()} Iter:{iteration} Random Seed:{args.random_seed}", fontweight="semibold", y=0.925)
        unviolated_labeled = Line2D([0], [0], label='Unviolated Labeled', marker=self.markers['unviolated labeled'], markersize=self.markersize, color='black', linestyle='')
        new_labeled = Line2D([0], [0], label='New Labeled', marker=self.markers['new labeled'], markersize=self.markersize, color='black', linestyle='')

        unlabeled_point = Line2D([0], [0], label='Unlabeled', marker=self.markers['unlabeled'], markersize=self.markersize*1.5, color='black', linestyle='')
        test_points = Line2D([0], [0], label='Test', marker=self.markers['test'], markersize=self.markersize*1.25, color='black', linestyle='')

        chosen_point = Line2D([0], [0], label='Next Added', marker=self.markers['next added'],  markeredgewidth=2, markersize=self.markersize, markerfacecolor='grey', markeredgecolor='black', linestyle='')
        class_0 = mpatches.Patch(color=plt.cm.coolwarm(0), label='Class A')  
        class_1 = mpatches.Patch(color=plt.cm.coolwarm(255), label='Class B')  

        legend_elements = [chosen_point, unlabeled_point, unviolated_labeled, new_labeled, test_points, class_0, class_1]
        lgd = plt.figlegend(handles=legend_elements, 
                      loc='lower center', 
                      handletextpad=0.3,
                      bbox_to_anchor=(0.5, 0.025), 
                      ncol=len(legend_elements), 
                      fancybox=True, 
                      shadow=True)

        utilities.makedir(path_to_store + f"{args.random_seed}/")
        plt.savefig(path_to_store + f"{args.random_seed}/" + str(iteration).zfill(4),  bbox_extra_artists=(lgd, title), bbox_inches='tight')
        plt.close()
