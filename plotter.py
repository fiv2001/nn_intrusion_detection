import matplotlib.pyplot as plt
import matplotlib
import copy

from seaborn import heatmap
from matplotlib.colors import LogNorm

def save_df_as_image(df, path):
    fig, ax = plt.subplots(figsize=(20, 20))
    my_cmap = copy.copy(matplotlib.cm.get_cmap('viridis'))
    my_cmap.set_bad((0,0,0))
    plot = heatmap(df, annot=True, fmt='.5g', xticklabels=True, yticklabels=True, robust=True, norm=LogNorm(), cmap=my_cmap)
    plt.xlabel('True class')
    plt.ylabel('Predicted class')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
