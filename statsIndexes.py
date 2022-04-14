import os
import numpy as np
from osgeo import gdal
from numpy import ma
import pandas as pd
from tqdm import tqdm

dir_path = r"E:\研究生毕业论文\数据和处理（原始资料）\热浪指标计算数据\危险性\危险性"
filenames = [
    os.path.join(dir_path, i) for i in os.listdir(dir_path)
    if i.endswith("tif")
]
df = pd.DataFrame()
with tqdm(filenames) as t:
    for filename in t:
        dataset = gdal.Open(filename)
        ndv = dataset.GetRasterBand(1).GetNoDataValue()
        im_data = dataset.ReadAsArray(0, 0, dataset.RasterXSize,
                                      dataset.RasterYSize)
        del dataset
        nan_im_data = ma.masked_where(im_data == ndv, im_data).filled(np.nan)
        name=filename.split("\\")[-1].split(".")[0]
        year=name.split("_")[0]
        if len(name.split("_"))==2:
            type_index = name.split("_")[1]
        else:
            type_index = name.split("_")[1:]
        nonzeronum=np.count_nonzero(nan_im_data) - len(nan_im_data[np.isnan(nan_im_data)])
        is0 = "包含0"

        stats = [year, type_index, is0]

        stats.append(nonzeronum)
        stats.append(np.nanmin(nan_im_data))
        stats.extend(np.nanpercentile(nan_im_data, (25, 50, 75)))
        stats.append(np.nanmax(nan_im_data))
        stats.append(np.nanmean(nan_im_data))

        df_1 = pd.DataFrame(
            [stats],
            columns=['年份','类型','是否包含0','非0格网数量', 'min', 'Q1', 'median', 'Q3', 'max', 'mean'])
        df = pd.concat([df, df_1], axis=0)

        nan_im_data = ma.masked_where(nan_im_data == 0,
                                      nan_im_data).filled(np.nan)
        is0 = "不包含0"

        stats = [year,type_index,is0]
        stats.append(nonzeronum)
        stats.append(np.nanmin(nan_im_data))
        stats.extend(np.nanpercentile(nan_im_data, (25, 50, 75)))
        stats.append(np.nanmax(nan_im_data))
        stats.append(np.nanmean(nan_im_data))

        df_1 = pd.DataFrame(
            [stats],
            columns=['年份','类型','是否包含0','非0格网数量', 'min', 'Q1', 'median', 'Q3', 'max', 'mean'])
        df = pd.concat([df, df_1], axis=0)

df.to_csv(r"E:\研究生毕业论文\实验部分\统计\一级指标_危险性.csv",
          index=False,
          header=True,
          encoding='gb2312')
