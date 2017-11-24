import os
import imageio
import numpy as np
# path = ('../cell/')
# allImgs = os.listdir(path)
# arrs = [imageio.imread('../cell/' + allImgs[i]) for i in range(1, len(allImgs)+1)]
# res = np.concatenate([arr[np.newaxis] for arr in arrs])
# print res.shape

positive_path = '../cell/'
negative_path = '../negative'
positive_example = os.listdir(positive_path)
negative_example = os.listdir(negative_path)

print('reading positive examples\n')
positive_arr = [imageio.imread('../cell/' + positive_example[i]) for i in range(1, 1000)]
print('reading negative examples\n')
negative_arr = [imageio.imread('../negative/' + negative_example[i]) for i in range(1, 1000)]

result = positive_arr + negative_arr

res = np.concatenate([arr[np.newaxis] for arr in result])

print(result)

