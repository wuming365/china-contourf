from tqdm import tqdm
import numpy as np
from osgeo import gdal, gdalconst


def reviseTif(infile, input_shape, outfile):

    # 大于100的数设置为100
    inputraster = gdal.Open(infile, gdal.GA_ReadOnly)

    ds = gdal.Warp(
        outfile,
        inputraster,
        format='GTiff',
        cutlineDSName=input_shape,
        cutlineWhere="OBJECTID = 1",
        dstNodata=-9999,
    )
    del inputraster

    outraster = gdal.Open(outfile, gdal.GA_Update)
    band = outraster.GetRasterBand(1)
    img_data = band.ReadAsArray()
    img_data[img_data > 100] = 100
    band.WriteArray(img_data)
    del outraster


if __name__ == "__main__":
    infile = r"E:\RHU-13003_19900501.tif"
    input_shape = r"E:\b.shp"
    outfile = r"E:\RHU-13003_19900501-1.tif"
    reviseTif(infile, input_shape, outfile)
