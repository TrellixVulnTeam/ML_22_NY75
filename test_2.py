import sys
sys.path.append('./Libs') 
import functions as cf
#-----------------------------------------------------------------------------------------#
import rasterio
import matplotlib.pyplot as plt

# pip install scikit-image

# NOTE import near infrared band (B8)
file_name = '../datasets/sentinel_2/2020/20200107/IMG_DATA/47PQS_20200107_B04.jp2'
B4 = rasterio.open(file_name)
B4 = B4.read(); B4 = B4[0, :, :]
print(B4.shape)
# plt.imshow(B4, cmap='terrain')
# plt.show()

file_name = '../datasets/sentinel_2/2020/20200107/IMG_DATA/47PQS_20200107_B02.jp2'
B2 = rasterio.open(file_name)
B2 = B2.read(); B2 = B2[0, :, :]

'''
step 2: flatten
'''

flat_B4 = B4.flatten()
flat_B2 = B2.flatten()
# print(flat_B4)
# print('number of pixels: ', len(flat_B4))
# number_of_pixels = B4.shape[0]*B4.shape[1]
# print('number of pixels: ', number_of_pixels)

'''
step 3: scattering plot
'''
import numpy as np

flat_B4 = flat_B4[flat_B4 >= 100]
flat_B4 = flat_B4[::10000]

flat_B2 = flat_B2[flat_B2 >= 100]
flat_B2 = flat_B2[::10000]

y = np.linspace(0, len(flat_B4), len(flat_B4))

plt.figure(figsize=(10, 10))
plt.scatter(flat_B4, y, s=100, color='orange', edgecolor='black')
plt.scatter(flat_B2, y, s=100, color='green', edgecolor='black', marker='X')
plt.show()

