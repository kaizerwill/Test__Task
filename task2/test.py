from train import match_images1
from train import match_images2
import rasterio
from rasterio.plot import reshape_as_image

RASTER_PATH = 'Data/T36UYA_20160212T084052_B09.jp2'

with rasterio.open(RASTER_PATH, "r", driver='JP2OpenJPEG') as src:
    raster_image = src.read()
    raster_meta = src.meta
raster_img1 = reshape_as_image(raster_image)

RASTER_PATH = 'Data/T36UYA_20160212T084052_B10.jp2'

with rasterio.open(RASTER_PATH, "r", driver='JP2OpenJPEG') as src:
    raster_image = src.read()
    raster_meta = src.meta
raster_img2 = reshape_as_image(raster_image)

match_images1(raster_img1, raster_img2)
match_images2(raster_img1, raster_img2)