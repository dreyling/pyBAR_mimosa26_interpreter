import numpy as np
from matplotlib.figure import Figure
from matplotlib.artist import setp
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors, cm
from matplotlib.backends.backend_pdf import PdfPages


def plot_fancy_occupancy(hist, title, z_max=None, filename=None):
    if z_max == 'median':
        z_max = 2 * np.ma.median(hist)
    elif z_max == 'maximum' or z_max is None:
        z_max = np.ma.max(hist)
    if z_max < 1 or hist.all() is np.ma.masked:
        z_max = 1.0

    fig = Figure()
    FigureCanvas(fig)
    ax = fig.add_subplot(111)

    ax.set_title(title, size=6)

    extent = [0.5, 1152.5, 576.5, 0.5]
    bounds = np.linspace(start=0, stop=z_max, num=255, endpoint=True)
    if z_max == 'median':
        cmap = cm.get_cmap('coolwarm')
    else:
        cmap = cm.get_cmap('cool')
    cmap.set_bad('w', 1.0)
    norm = colors.BoundaryNorm(bounds, cmap.N)

    im = ax.imshow(hist, interpolation='nearest', aspect='auto', cmap=cmap, norm=norm, extent=extent)  # TODO: use pcolor or pcolormesh
    ax.set_ylim((576.5, 0.5))
    ax.set_xlim((0.5, 1152.5))
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')

    # Create new axes on the right and on the top of the current axes
    # The first argument of the new_vertical(new_horizontal) method is
    # the height (width) of the axes to be created in inches.
    divider = make_axes_locatable(ax)
    axHistx = divider.append_axes("top", 1.2, pad=0.2, sharex=ax)
    axHisty = divider.append_axes("right", 1.2, pad=0.2, sharey=ax)

    cax = divider.append_axes("right", size="5%", pad=0.1)
    cb = fig.colorbar(im, cax=cax, ticks=np.linspace(start=0, stop=z_max, num=9, endpoint=True))
    cb.set_label("#")
    # make some labels invisible
    setp(axHistx.get_xticklabels() + axHisty.get_yticklabels(), visible=False)
    hight = np.ma.sum(hist, axis=0)

    axHistx.bar(left=range(1, 1153), height=hight, align='center', linewidth=0)
    axHistx.set_xlim((0.5, 1152.5))
    if hist.all() is np.ma.masked:
        axHistx.set_ylim((0, 1))
    axHistx.locator_params(axis='y', nbins=3)
    axHistx.ticklabel_format(style='sci', scilimits=(0, 4), axis='y')
    axHistx.set_ylabel('#')
    width = np.ma.sum(hist, axis=1)

    axHisty.barh(bottom=range(1, 577), width=width, align='center', linewidth=0)
    axHisty.set_ylim((576.5, 0.5))
    if hist.all() is np.ma.masked:
        axHisty.set_xlim((0, 1))
    axHisty.locator_params(axis='x', nbins=3)
    axHisty.ticklabel_format(style='sci', scilimits=(0, 4), axis='x')
    axHisty.set_xlabel('#')

    if not filename:
        fig.show()
    elif isinstance(filename, PdfPages):
        filename.savefig(fig)
    else:
        fig.savefig(filename)