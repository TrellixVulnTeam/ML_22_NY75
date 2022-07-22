import sys
sys.path.append('./Libs') 
import customized_functions as cf
#-----------------------------------------------------------------------------------------#
import pandas as pd
import rasterio
from rasterio.features import rasterize
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd

'''
Part 3: The final part wraps up how B2, B3, B4, B8, B8A, and NDVI (resolution 10 m) transformed into tabular data. After this step, we will have data that are ready to feed in ML algorithm. 
'''