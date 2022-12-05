import matplotlib.pyplot as plt
import numpy as np

FIG_SIZE = (5, 5)

def get_figure(title, subtitle='', figsize=FIG_SIZE, xlabel=None, ylabel=None, ylim=None):
    """Returns a figure with the given title, subtitle, figsize, xlabel, ylabel.
    Args:
        title (str): the title of the figure.
        subtitle (str, optional): the subtitle of the figure. Defaults to ''.
        figsize (tuple, optional): the size of the figure. Defaults to FIG_SIZE.
        xlabel (str, optional): the label of the x axis. Defaults to None.
        ylabel (str, optional): the label of the y axis. Defaults to None.
        ylim (tuple, optional): the limits of the y axis. Defaults to None.
    Returns:
        matplotlib.figure.Figure: the figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(title)
    ax.set_title(subtitle)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    if ylim is not None:
        ax.set_ylim(ylim)
    return fig, ax

def heatmap_from_numpy(numpy_array, title, subtitle='', figsize=FIG_SIZE, xlabel=None, ylabel=None, cmap='Blues'):
    """Returns a heatmap from the given numpy array.
    Args:
        numpy_array (np.array): the numpy array to plot.
        title (str): the title of the figure.
        subtitle (str, optional): the subtitle of the figure. Defaults to ''.
        figsize (tuple, optional): the size of the figure. Defaults to FIG_SIZE.
        xlabel (str, optional): the label of the x axis. Defaults to None.
        ylabel (str, optional): the label of the y axis. Defaults to None.
        cmap (str, optional): the color map to use. Defaults to 'Blues'.
    
    Returns:
        matplotlib.figure.Figure: the figure.
    """
    fig, ax = get_figure(title, subtitle, figsize, xlabel, ylabel)

    ax.imshow(numpy_array, cmap=cmap)
    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(numpy_array.shape[0]), labels=np.arange(numpy_array.shape[0]))
    ax.set_yticks(np.arange(numpy_array.shape[1]), labels=np.arange(numpy_array.shape[1]))
    
    # Loop over data dimensions and create text annotations.
    for k in range(numpy_array.shape[0]):
        for l in range(numpy_array.shape[1]):
            ax.text(k, l, numpy_array[k, l], ha="center", va="center", color="orange")

    return fig

    