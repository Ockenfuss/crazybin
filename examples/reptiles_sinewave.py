import numpy as np
import matplotlib.pyplot as plt
from crazybin import crazybin

x=np.linspace(0,10,100)
y=np.linspace(0,10,100)
x,y=np.meshgrid(x,y)
x=x.ravel()
y=y.ravel()
weights=np.sin(x)*np.cos(y)

fig, ax=plt.subplots(figsize=(5,5))
crazybin(x,y,weights, tile='reptile', cmap='jet', gridsize=4, edgecolor='black', ax=ax)
ax.get_yaxis().set_visible(False)
ax.get_xaxis().set_visible(False)

savepath='images/reptiles_sinewave.jpg'
fig.savefig(savepath, dpi=300, bbox_inches='tight')