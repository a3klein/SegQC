import numpy as np
import matplotlib.pyplot as plt

# Plotting a table
# Taken from https://medium.com/data-science/simple-little-tables-with-matplotlib-9780ef5d0bc4
def plot_table(
        data:np.ndarray,
        rowLabels:list,
        colLabels:list,
        title:str,
        save_fig:bool=False,
        image_path:str=None,
        rcolors:list=None,
        ccolors:list=None,
        footer_text:str="",
        fig_background_color:str='skyblue',
        fig_border:str='steelblue',
        figsize:tuple=(8, 4),
        ):
    """
    Plot a table with the given data, row labels, and column labels.

    Parameters
    ----------
    data : np.ndarray
        The data to be plotted in the table.
    rowLabels : list
        The labels for the rows of the table.
    colLabels : list
        The labels for the columns of the table.
    title : str
        The title of the plot.
    save_fig : bool, optional
        Whether to save the figure, by default False
    image_path : str, optional
        The path to save the figure, by default None
    rcolors : list, optional
        color for the rows, by default None
    ccolors : list, optional
        color for the cols, by default None
    footer_text : str, optional
        The text to be displayed in the footer, by default ""
    fig_background_color : str, optional
        The background color of the figure, by default 'skyblue'
    fig_border : str, optional
        The border color of the figure, by default 'steelblue'
    figsize : tuple, optional
        The size of the figure, by default (8, 4)
    """
    # Make sure that the table is of a non-numeric str type
    cell_text = []
    for row in data:
        cell_text.append([str(x) for x in row])
    
    
    # # Get some lists of color specs for row and column headers
    # rcolors = plt.cm.BuPu(np.full(len(row_headers), 0.1))
    # ccolors = plt.cm.BuPu(np.full(len(column_headers), 0.1))


    # Create the figure. Setting a small pad on tight_layout
    # seems to better regulate white space. Sometimes experimenting
    # with an explicit figsize here can produce better outcome.
    plt.figure(linewidth=2,
            edgecolor=fig_border,
            facecolor=fig_background_color,
            tight_layout={'pad':1},
            figsize=figsize
            )

    # Add a table at the bottom of the axes
    the_table = plt.table(cellText=cell_text,
                        rowLabels=rowLabels,
                        rowColours=rcolors,
                        rowLoc='right',
                        colColours=ccolors,
                        colLabels=colLabels,
                        loc='center')
    # Scaling is the only influence we have over top and bottom cell padding.
    # Make the rows taller (i.e., make cell y scale larger).
    the_table.scale(1, 1.5)
    # Hide axes
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Hide axes border
    plt.box(on=None)
    # Add title
    plt.suptitle(title)
    # Add footer
    plt.figtext(0.95, 0.05, footer_text, horizontalalignment='right', size=6, weight='light')
    # Force the figure to update, so backends center objects correctly within the figure.
    # Without plt.draw() here, the title will center on the axes and not the figure.
    plt.draw()
    # Create image. plt.savefig ignores figure edge and face colors, so map them.
    fig = plt.gcf()
    if save_fig: 
        plt.savefig(image_path,
                    #bbox='tight',
                    edgecolor=fig.get_edgecolor(),
                    facecolor=fig.get_facecolor(),
                    dpi=150
                    )
    else: 
        return (fig, ax) 
    


def plot_table_on_ax(
        data:np.ndarray,
        rowLabels:list,
        colLabels:list,
        title:str="",
        rcolors:list=None,
        ccolors:list=None,
        ax=None,
        ):
    """
    Plot a table with the given data, row labels, and column labels.

    Parameters
    ----------
    data : np.ndarray
        The data to be plotted in the table.
    rowLabels : list
        The labels for the rows of the table.
    colLabels : list
        The labels for the columns of the table.
    title : str
        The title of the axes.
    rcolors : list, optional
        color for the rows, by default None
    ccolors : list, optional
        color for the cols, by default None
    ax : 
        the ax object to plot this on
    """
    # Make sure that the table is of a non-numeric str type
    if ax == None: 
        fig, ax = plt.subplots()

    # Add a table at the bottom of the axes
    the_table = ax.table(cellText=data,
                        rowLabels=rowLabels,
                        rowColours=rcolors,
                        rowLoc='right',
                        colColours=ccolors,
                        colLabels=colLabels,
                        loc='top')
    # Scaling is the only influence we have over top and bottom cell padding.
    # Make the rows taller (i.e., make cell y scale larger).
    the_table.scale(1, 1.5)
    # Hide axes
    ax = plt.gca()
    ax.axis("off")
    ax.axis('tight')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Add title
    ax.set_title(title)
    
    fig = plt.gcf()
    return (fig, ax) 