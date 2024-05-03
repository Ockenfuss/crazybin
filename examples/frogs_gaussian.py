import numpy as np
import matplotlib.pyplot as plt
from crazybin import crazybin

x=np.linspace(0,10,100)
y=np.linspace(0,10,100)
x,y=np.meshgrid(x,y)
x=x.ravel()
y=y.ravel()
#Gauss peak
weights=np.exp(-((x-5)**2+(y-5)**2)/10)

fig, ax=plt.subplots(figsize=(5,5))
crazybin(x,y,weights, tile='frog', cmap='inferno', gridsize=6, edgecolor='black', ax=ax)
ax.set_title('Gaussian peak, Frog tiles')
ax.get_yaxis().set_visible(False)
ax.get_xaxis().set_visible(False)

savepath='images/frogs_gaussian.jpg'
fig.savefig(savepath, dpi=300, bbox_inches='tight')