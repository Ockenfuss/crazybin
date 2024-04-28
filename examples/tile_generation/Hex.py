#%%
from shapely import Point, Polygon
from shapely import affinity
from shapely.ops import unary_union
from matplotlib import pyplot as plt
import numpy as np
import json
#%%
# Define Hexagon
radius1=1

O=Point(0,0)
A=affinity.rotate(Point(0, radius1), 30, origin=O)
hexpoints=[affinity.rotate(A, 60*i, origin=O) for i in range(6)]
hex=Polygon(hexpoints)
fig, ax=plt.subplots(figsize=(5,5))
ax.scatter(*O.coords[0], color='red')
ax.scatter(*A.coords[0], color='green')
ax.plot(*hex.exterior.xy)
v1=[hexpoints[4].x-hexpoints[2].x, hexpoints[4].y-hexpoints[2].y]
v2=[hexpoints[5].x-hexpoints[3].x, hexpoints[5].y-hexpoints[3].y]
ax.fill(*affinity.translate(hex, *v1).exterior.xy, alpha=0.5)
ax.fill(*affinity.translate(hex, *v2).exterior.xy, alpha=0.5)
#%%
tile={'v1':v1, 'v2':v2, 'atoms':[list(hex.exterior.coords)]}
with open('../../src/crazybin/tiles/hex.json', 'w') as f:
    json.dump(tile, f, indent=4)

# %%
