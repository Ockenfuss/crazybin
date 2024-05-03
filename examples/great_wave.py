#%%
import numpy as np
import matplotlib.pyplot as plt
from crazybin import imshow

imagepath='images/originals/Great_Wave_Kanagawa.jpg'
image=plt.imread(imagepath)/255
#%%
fig, ax=plt.subplots(figsize=(5,5))
imshow(image, tile='pen_rhomb', ax=ax, gridsize=8)
ax.set_title('The Great Wave of Kanagawa, Penrose Rhombs')
ax.get_yaxis().set_visible(False)
ax.get_xaxis().set_visible(False)
#%%
savepath='images/great_wave.jpg'
fig.savefig(savepath, dpi=300, bbox_inches='tight')
# %%
