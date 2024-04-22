from  .objects import Tile, Histogram, TileImage
__version__="1.0.0"

def crazybin(x,y,C,tile='reptile', gridsize=10, cmap='viridis', vmin=None, vmax=None, edgecolor=None, areadensity=True, ax=None):
    hist=Histogram(x,y,weights=C, tile=tile, gridsize=gridsize, areadensity=areadensity)
    hist.plot(cmap=cmap, vmin=vmin, vmax=vmax, edgecolor=edgecolor, ax=ax)

def imshow(X, tile='reptile', gridsize=10, cmap='viridis', vmin=None, vmax=None, edgecolor=None, extent=None, ax=None):
    im=TileImage(X, tile=tile, gridsize=gridsize, extent=extent)
    im.plot(cmap=cmap, vmin=vmin, vmax=vmax, edgecolor=edgecolor, ax=ax)