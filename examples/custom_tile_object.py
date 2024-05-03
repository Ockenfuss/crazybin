import numpy as np
from shapely import Polygon
import matplotlib.pyplot as plt
from crazybin import Tile, Grid, TileImage, RegularParquetFactory
grid=Grid([0,1],[1,0])
tile=Tile([Polygon([(0,0), (1,0), (1,1), (0,1)])])
fac=RegularParquetFactory(tile, grid)
image=np.random.rand(10,10)
im=TileImage(image, fac, gridsize=10)
im.plot()
plt.show()