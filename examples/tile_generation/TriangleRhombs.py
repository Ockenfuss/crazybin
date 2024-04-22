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
radius2=np.tan(75/180*np.pi)*radius1/2+np.sin(60/180*np.pi)*radius1 #this way, the rhombs are acutally squares

O=Point(0,0)
A=affinity.rotate(Point(0, radius1), 30, origin=O)
B=Point(0, radius2)
hexpoints=[affinity.rotate(A, 60*i, origin=O) for i in range(6)]
hex=Polygon(hexpoints)
outer_points=[affinity.rotate(B, 60*i, origin=O) for i in range(6)]
triangles=[Polygon([hexpoints[i-1], hexpoints[i], outer_points[i]]) for i in range(6)]
vv=[Point(outer_points[i].x-outer_points[i-2].x, outer_points[i].y-outer_points[i-2].y) for i in range(6)]
# v2=[outer_points[0].x-outer_points[2].x, outer_points[0].y-outer_points[2].y]
# v3=[outer_points[1].x-outer_points[3].x, outer_points[1].y-outer_points[3].y]
rhomb_points=[affinity.translate(hexpoints[i-3], *(vv[i].coords[0])) for i in range(6)]
rhombs=[Polygon([hexpoints[i], outer_points[i+1], rhomb_points[i], outer_points[i]]) for i in range(-1,5)]

fig, ax=plt.subplots(figsize=(5,5))
ax.scatter(*O.coords[0], color='red')
ax.scatter(*A.coords[0], color='green')
ax.plot(*hex.exterior.xy)
for t in triangles:
    ax.plot(*t.exterior.xy)
for v in vv:
    ax.plot([0, v.x], [0, v.y], color='red')
for sq in rhomb_points:
    ax.scatter(*sq.coords[0], color='blue')
for s in rhombs:
    ax.plot(*s.exterior.xy)
ax.set_xlim(-4,4)
ax.set_ylim(-4,4)
#%%
primitive=[hex]+triangles+rhombs[3:]
fig, ax=plt.subplots()
for p in primitive:
    x,y=p.exterior.xy
    ax.plot(x,y)
union=unary_union(primitive)
v1=[vv[4].x, vv[4].y]
v2=[vv[5].x, vv[5].y]
ax.fill(*union.exterior.xy, alpha=0.5)
ax.fill(*affinity.translate(union, *v1).exterior.xy, alpha=0.5)
ax.fill(*affinity.translate(union, *v2).exterior.xy, alpha=0.5)
#%%
tile={'v1':v1, 'v2':v2, 'atoms':[list(p.exterior.coords) for p in primitive]}
with open('/project/meteo/work/Paul.Ockenfuss/PhD/Miscellaneous/EscherPlots/Tiles/hex_rhombs.json', 'w') as f:
    json.dump(tile, f, indent=4)

# %%
