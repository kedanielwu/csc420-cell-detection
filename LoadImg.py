import os
import imageio
import numpy as np
path = ('../cell/')
allImgs = os.listdir(path)
arrs = [imageio.imread('../cell/' + allImgs[i]) for i in range(1, len(allImgs)+1)]
res = np.concatenate([arr[np.newaxis] for arr in arrs])
print res.shape

