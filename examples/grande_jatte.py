#%%
import numpy as np
import matplotlib.pyplot as plt
from crazybin import imshow

imagepath='images/originals/Grande_Jatte.jpeg'
image=plt.imread(imagepath)/255
#%%
fig, ax=plt.subplots(figsize=(7,7))
imshow(image, tile='hex', ax=ax, gridsize=150)
ax.get_yaxis().set_visible(False)
ax.get_xaxis().set_visible(False)
#%%
savepath='images/grande_jatte_seurat.jpg'
fig.savefig(savepath, dpi=300, bbox_inches='tight')
# %%
