from  .objects import Tile, Grid, RegularParquetFactory, Histogram, TileImage, ParquetFactory
__version__="1.0.0"

def crazybin(x,y,C=None,tile='reptile', gridsize=7, cmap='viridis', vmin=None, vmax=None, edgecolor=None, areadensity=True, ax=None):
    """Make a 2D binning plot with arbitrary tilings.

    Parameters
    ----------
    x,y : array-like
        The data positions. x and y must be of the same length.
    C : array-like, optional
        If given, these values are accumulated in the bins. Otherwise, every point has a value of 1. Must be of the same length as x and y., by default None
    tile : str or Path, optional
        Keyword or path to file with tiling. See documentation for details. By default 'reptile'
    gridsize : int or float, optional
        The resolution of the parquet. The higher, the more tiles are generated. For regular parquets, this is roughly equivalent to the number of tiles in the x direction. For penrose parquets, this is the number of generations in the deflation algorithm and should usually be below 10., by default 7.
    cmap : str or Colormap instance, optional
        The Colormap instance or registered colormap name used to map scalar data to colors., by default 'viridis'
    vmin, vmax : float, optional
        When using scalar data and no explicit norm, vmin and vmax define the data range that the colormap covers. By default, the colormap covers the complete value range of the supplied data. By default None
    edgecolor : color, optional
        The color of the tile edges. If None, draw outlines in the default color. By default None
    areadensity : bool, optional
        If true, divide the value of each tile by its area. By default True
    ax : matplotlib.axes instance, optional
        If given, draw into this axes instance. Otherwise, a figure with one axes is created. By default None
    """
    parquet_factory=ParquetFactory.from_keyword(tile)
    hist=Histogram(x,y,parquet_factory=parquet_factory, weights=C, gridsize=gridsize, areadensity=areadensity)
    hist.plot(cmap=cmap, vmin=vmin, vmax=vmax, edgecolor=edgecolor, ax=ax)

def imshow(X, tile='reptile', gridsize=10, cmap='viridis', vmin=None, vmax=None, edgecolor=None, extent=None, ax=None):
    """Display data as an image on a 2D parquet out of arbitrary tiles.

    Parameters
    ----------
    X : array-like
        The image data. Supported array shapes are:

        (M, N): an image with scalar data. The values are mapped to colors using normalization and a colormap. See parameters cmap, vmin, vmax.

        (M, N, 3): an image with RGB values (0-1 float or 0-255 int).

    The first two dimensions (M, N) define the rows and columns of the image.

    Out-of-range RGB values are clipped.

    tile : str or Path, optional
        Keyword or path to file with tiling. See documentation for details. By default 'reptile'
    gridsize : int or float, optional
        The resolution of the parquet. The higher, the more tiles are generated. For regular parquets, this is roughly equivalent to the number of tiles in the x direction. For penrose parquets, this is the number of generations in the deflation algorithm and should usually be below 10., by default 7.
    cmap : str or Colormap instance, optional
        The Colormap instance or registered colormap name used to map scalar data to colors., by default 'viridis'
    vmin, vmax : float, optional
        When using scalar data and no explicit norm, vmin and vmax define the data range that the colormap covers. By default, the colormap covers the complete value range of the supplied data. By default None
    edgecolor : color, optional
        The color of the tile edges. If None, draw outlines in the default color. By default None
    extent : 

    ax : matplotlib.axes instance, optional
        If given, draw into this axes instance. Otherwise, a figure with one axes is created. By default None
    """
    parquet_factory=ParquetFactory.from_keyword(tile)
    im=TileImage(X, parquet_factory=parquet_factory, gridsize=gridsize, extent=extent)
    im.plot(cmap=cmap, vmin=vmin, vmax=vmax, edgecolor=edgecolor, ax=ax)