#%%
import numpy as np
import matplotlib.pyplot as plt
from crazybin import imshow

center=8
x=np.linspace(0,2*center,100)
y=np.linspace(0,2*center,100)
x,y=np.meshgrid(x,y)
weights=np.sin(np.sqrt((x-center)**2+(y-center)**2))#*np.cos(y)
#%%
fig, ax=plt.subplots(figsize=(7,7))
#nice: tab20, plasma, Set2
imshow(weights, tile='hex_rhomb', ax=ax, gridsize=4.0, edgecolor='black', cmap='plasma')
ax.set_title('Hexagonal rhomb tiles')
ax.get_yaxis().set_visible(False)
ax.get_xaxis().set_visible(False)
#%%
savepath='images/hex_rhomb.jpg'
fig.savefig(savepath, dpi=300, bbox_inches='tight')
# %%
