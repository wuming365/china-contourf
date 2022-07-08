import os
from numpy.core.numeric import full
from osgeo.gdalconst import GA_Update
import pandas as pd
# from icecream import ic
from scipy.ndimage.measurements import label, mean
from tqdm import tqdm
import numpy as np
from osgeo import gdal
import numpy.ma as ma
import time
# import numba as nb
# import cupy as cp
from scipy import stats
from multiprocessing import Pool


def openSingleImage(filename: str):
    """打开影像

    Args:
        filename (str): 文件路径

    Returns:
        im_data: array,数据数组
        im_proj: str,坐标系
        im_geotrans: tuple,仿射矩阵
        im_height,im_width: float,图像高和宽
        ndv: float,NoDataValue
    """

    dataset = gdal.Open(filename)
    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    im_bands = dataset.RasterCount
    im_band = dataset.GetRasterBand(1)
    ndv = im_band.GetNoDataValue()
    im_geotrans = dataset.GetGeoTransform()
    im_proj = dataset.GetProjection()
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)
    del dataset
    return im_data, im_proj, im_geotrans, im_height, im_width, ndv


def openImages(filenames):
    """
    打开工作文件夹中的所有影像
    dirpath:数据读取文件夹
    return:Igs,im_geotrans.im_proj,ndv
    """
    Igs = []
    idate = 0
    with tqdm(filenames) as t:
        for filename in t:
            if filename[-4:] == ".tif":
                Image, im_proj, im_geotrans, _, _, ndv = openSingleImage(
                    filename)
                Igs.append(Image)
                idate = idate + 1
                t.set_description(filename + " is already open……")
    return np.array(Igs), im_geotrans, im_proj, ndv


def write_img(im_data, filename, im_proj, im_geotrans, dirpath, ndv):
    """写影像"""
    # im_data被写的影像
    # im_proj, im_geotrans均为被写影像参数
    # filename创建新影像的名字，dirpath影像写入文件夹
    # 判断栅格数据类型
    datatype = gdal.GDT_Float32
    fullpath = dirpath + "\\" + filename
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_bands = 1  # 均为单波段影像
        im_height, im_width = im_data.shape
        im_data = [im_data]
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(fullpath, im_width, im_height, im_bands, datatype)
    if dataset != None:
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).SetNoDataValue(ndv)  # 设置nodata值
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    else:
        os.remove(fullpath)
        with open(r"I:/dem_chazhi/log/error.txt", "a") as o:
            o.write(fullpath + "\r")
    del dataset
    del im_data


def calc_heat_index(RHU, TEM):

    minRHU = ma.masked_where(RHU > 60, RHU).filled(0)
    minTEM = ma.masked_where(RHU > 60, TEM).filled(0)
    maxRHU = ma.masked_where(RHU <= 60, RHU).filled(0)
    maxTEM = ma.masked_where(RHU <= 60, TEM).filled(0)
    minTI = 1.8 * minTEM - 0.55 * (1.8 * minTEM - 26) * (1 - 60 * 0.01) + 32
    maxTI = 1.8 * maxTEM - 0.55 * (1.8 * maxTEM - 26) * (1 -
                                                         maxRHU * 0.01) + 32
    TI = maxTI + minTI
    return TI


def calc_the_heat_threshold(T_threshold=33):
    global tems_path
    global rhus_path
    global tem_files
    turn_path = r"I:\dem_chazhi\result\TI'"
    with tqdm(tem_files) as t:
        for tem_file in t:
            t.set_description_str("Done:")
            TI_threshold_file = f"TI_threshold_{tem_file[-12:]}"
            if not os.path.exists(os.path.join(turn_path, TI_threshold_file)):
                rhu_file = f"{rhus_path}/RHU-13003_{tem_file.split('/')[-1][10:]}"
                tem_img, im_proj, im_geotrans, im_height, im_width, ndv = openSingleImage(
                    tem_file)
                rhu_img = openSingleImage(rhu_file)[0]
                mask = (tem_img == ndv) + (tem_img <= T_threshold)
                tem_mask = ma.masked_where(mask, tem_img)
                rhu_mask = ma.masked_where(mask, rhu_img)
                TI_threshold = calc_heat_index(rhu_mask, tem_mask)
                TI_threshold = ma.masked_where(mask, TI_threshold).filled(ndv)
                write_img(TI_threshold, TI_threshold_file, im_proj,
                          im_geotrans, turn_path, ndv)


def calc_TIs():
    global tems_path
    global rhus_path
    global tem_files
    turn_path = r"I:\dem_chazhi\result\TI"
    with tqdm(tem_files) as t:
        for tem_file in t:
            t.set_description_str("Done:")
            TI_file = f"TI_{tem_file[-12:]}"
            if not os.path.exists(os.path.join(turn_path, TI_file)):
                rhu_file = f"{rhus_path}/RHU-13003_{tem_file.split('/')[-1][10:]}"
                tem_img, im_proj, im_geotrans, im_height, im_width, ndv = openSingleImage(
                    tem_file)
                rhu_img = openSingleImage(rhu_file)[0]
                mask = (tem_img == ndv)
                tem_mask = ma.masked_where(mask, tem_img)
                rhu_mask = ma.masked_where(mask, rhu_img)
                TI = calc_heat_index(rhu_mask, tem_mask)
                TI = ma.masked_where(mask, TI).filled(ndv)
                write_img(TI, TI_file, im_proj, im_geotrans, turn_path, ndv)


def calc_TIpie():
    path = r"I:\dem_chazhi\result\TI'"
    split_path = r"J:/splitTI'"
    files = [i for i in os.listdir(path) if i.endswith("tif")]

    with tqdm(range(8)) as t1:
        for i in t1:
            t1.set_description_str("Done")
            first = False
            Igs = []
            with tqdm(files) as t2:
                for file in t2:
                    t2.set_description_str("Open")
                    split_file = os.path.join(split_path,
                                              file[:-4] + str(i) + ".tif")
                    if os.path.exists(split_file):
                        im_data, im_proj, im_geotrans, im_height, im_width, ndv = openSingleImage(
                            split_file)
                        im_data = ma.masked_where(im_data == ndv,
                                                  im_data).filled(np.nan)
                        # im_data = cp.asarray(
                        # cp.reshape(im_data, (1, im_height, im_width)))
                        Igs.append(im_data)
                        # if first == False:
                        # Igs = im_data
                        # first = True
                        # else:
                        # Igs = cp.append(Igs, im_data, axis=0)
            # cp.putmask(Igs, Igs == ndv, cp.nan)
            Igs = np.array(Igs)
            out_img = np.nanmedian(Igs, axis=0)
            write_img(out_img, "TI'_" + str(i) + ".tif", im_proj, im_geotrans,
                      "I:/dem_chazhi/result", ndv)


def calc_TIpie_2():
    TIpie_path = r"I:\dem_chazhi\result\TI'.tif"
    TIpie_copy = r"I:\dem_chazhi\result\TI'_copy.tif"
    TIs_path = r"I:\dem_chazhi\result\TI"
    # split_path = r"M:/splitTI"
    max_path = r"I:\dem_chazhi\result\Max.tif"
    TIs_name = [i for i in os.listdir(TIs_path) if i.endswith(".tif")]
    TI_mask = os.path.join(TIs_path, TIs_name[0])
    # os.system("copy %s %s" % (TIpie_path, TIpie_copy))
    im_data, ndv = openSingleImage(TI_mask)[0], openSingleImage(TI_mask)[-1]
    mask = im_data != ndv
    del im_data
    # first = True
    # Max = None
    # with tqdm(range(8)) as t1:
    #     for i in t1:
    #         t1.set_description_str("Done")
    #         # is_first = True
    #         Igs = []
    #         with tqdm(TIs_name) as t2:
    #             for file in t2:
    #                 t2.set_description_str("Open")
    #                 split_file = os.path.join(split_path,
    #                                           file[:-4] + str(i) + ".tif")
    #                 if os.path.exists(split_file):
    #                     data, im_proj, im_geotrans, _, _, ndv = openSingleImage(
    #                         split_file)
    #                     Igs.append(data)
    #         print("Calculating Max:")
    #         Igs = np.array(Igs)
    #         max = np.nanmax(Igs, axis=0)
    #         write_img(max, "Max_" + str(i) + ".tif", im_proj, im_geotrans,
    #                   "I:/dem_chazhi/result", ndv)
    #         del Igs
    #         del max
    #         if first:
    #             Max = max
    #             first = False
    #         else:
    #             Max = np.append(Max, max, axis=1)
    dataset = gdal.Open(max_path)
    im_band = dataset.GetRasterBand(1)
    Max = im_band.ReadAsArray()
    del dataset

    dataset = gdal.Open(TIpie_copy, GA_Update)
    im_band = dataset.GetRasterBand(1)
    im_data = im_band.ReadAsArray()

    (height, width) = im_data.shape
    for i in tqdm(range(height)):
        for j in tqdm(range(width)):
            if mask[i][j] and im_data[i][j] == ndv and Max[i][j] != ndv:
                im_data[i][j] = Max[i][j]
    im_band.WriteArray(im_data)
    del dataset


# def getHeatWaveFreq(b):
#     if np.max(b) != 0:
#         b = b.astype(np.int32)  #关键，把浮点型转换为整数型,否则结果会很小"00.10"过分割
#         c = ''.join(str(i) for i in b)
#         d = np.array([len(i) for i in c.split('0')])
#         return d
#     else:
#         return []


def calc_HI():
    years = np.arange(1990, 2000)
    TI_path = r"G:\dem_chazhi\result\TI"
    TIpie_path = r"G:\dem_chazhi\result\TI'_copy.tif"
    output_path = r"G:\dem_chazhi\result\HI"
    TIpie, im_proj, im_geotrans, im_height, im_width, ndv = openSingleImage(
        TIpie_path)
    mask = TIpie == ndv
    TIpie = ma.masked_where(mask, TIpie)
    for year in years:
        if not os.path.exists(os.path.join(output_path,
                                           f"{str(year)}_HIs.tif")):
            TIs = [
                os.path.join(TI_path, i) for i in os.listdir(TI_path)
                if str(year) in i
            ]
            TIs = openImages(TIs)[0]
            hotimg = np.zeros_like(TIs, dtype=np.bool8)

            hotimg[TIs > TIpie] = True
            hotimg = np.transpose(hotimg)
            # """热浪日数"""
            # freimg_1 = np.zeros_like(TIs[0])
            # freimg_2 = np.zeros_like(TIs[0])
            # freimg_3 = np.zeros_like(TIs[0])
            # """最长热浪持续日数"""
            # longest_duration = np.zeros_like(TIs[0])
            # """热浪开始日期"""
            # start_date=np.zeros_like(TIs[0])
            # """热浪结束日期"""
            # end_date=np.zeros_like(TIs[0])
            # """计算HI"""
            # min_HI = np.zeros_like(TIs[0])
            # max_HI = np.zeros_like(TIs[0])
            # total_HI = np.zeros_like(TIs[0])

            HIs = np.zeros_like(TIs)
            with tqdm(range(im_width)) as t:
                for i in t:
                    for j in range(im_height):

                        if np.max(hotimg[i][j]) == 0:
                            continue
                        # fre = getHeatWaveFreq(hotimg[i][j])

                        date = 0  #热浪到目前为止持续的天数-1
                        for k, is_hot in enumerate(hotimg[i][j]):
                            if is_hot:
                                sum = 0
                                HI = 1.2 * (TIs[k][j][i] - TIpie[j][i])
                                if date >= 1:
                                    for l in range(date):
                                        ndi = l + 1
                                        sum += 0.35 * (
                                            1 / ndi *
                                            (TIs[k - ndi][j][i] - TIpie[j][i])
                                        ) + 0.15 * (1 / ndi)  #1/2=0.5
                                HI += sum + 1
                                HIs[k][j][i] = HI
                                date += 1
                            else:
                                date = 0
            for HI in HIs:
                HI = ma.masked_where(mask, HI).filled(ndv)
            # HIs = ma.masked_where(mask, HIs).filled(ndv)

            write_img(HIs, f"{str(year)}_HIs.tif", im_proj, im_geotrans,
                      output_path, ndv)
            del HIs
            del TIs
            del hotimg
        # for n in fre:
        #     if n == 0:
        #         date += 1
        #     elif n == 1:
        #         date += 1
        #         HI = 1.2 * (TIs[date][j][i] - TIpie[j][i])
        #     else:
        #         date += n
        #         for k in range(n - 1):
        #             ndi = k + 1
        #             sum += 0.35 * (1 / TIs[date - ndi][j][i] *
        #                            (TIs[date - ndi][j][i] -
        #                             TIpie[j][i])) + 0.15 * (
        #                                 1 / ndi)
        #         HI = 1.2 * (TIs[date][j][i] -
        #                     TIpie[j][i]) + 1 + sum
        #         sum = 0
        #         if HI >= 2.8:
        #             HIs.append(HI)
        #             if 2.8 <= HI < 6.5:
        #                 freimg_1[j][i] += 1
        #             elif 6.5 <= HI < 10.5:
        #                 freimg_2[j][i] += 1
        #             elif 10.5 <= HI:
        #                 freimg_3[j][i] += 1
        # HIs = np.array(HIs)

        # if len(HIs) != 0:
        #     min_HI[j][i] = np.min(HIs)
        #     max_HI[j][i] = np.max(HIs)
        #     # total_HI[j][i] = np.sum(HIs)
        # del HIs
        # del HI

        # freimg_1 = ma.masked_where(mask, freimg_1).filled(ndv)
        # freimg_2 = ma.masked_where(mask, freimg_2).filled(ndv)
        # freimg_3 = ma.masked_where(mask, freimg_3).filled(ndv)
        # min_HI = ma.masked_where(mask, min_HI).filled(ndv)
        # max_HI = ma.masked_where(mask, max_HI).filled(ndv)
        # total_HI = ma.masked_where(mask, total_HI).filled(ndv)

        # write_img(freimg_1, f"{str(year)}_frequent_1.tif", im_proj,
        #           im_geotrans, output_path, ndv)
        # write_img(freimg_2, f"{str(year)}_frequent_2.tif", im_proj,
        #           im_geotrans, output_path, ndv)
        # write_img(freimg_3, f"{str(year)}_frequent_3.tif", im_proj,
        #           im_geotrans, output_path, ndv)
        # write_img(min_HI, f"{str(year)}_min_HI.tif", im_proj, im_geotrans,
        #           output_path, ndv)
        # write_img(max_HI, f"{str(year)}_max_HI.tif", im_proj, im_geotrans,
        #           output_path, ndv)
        # write_img(total_HI, f"{str(year)}_total_HI.tif", im_proj, im_geotrans,
        #           output_path, ndv)


def getHeatWaveFreq(b):
    if np.max(b) != 0:
        b = b.astype(np.int32)  #关键，把浮点型转换为整数型,否则结果会很小"00.10"过分割
        c = ''.join(str(i) for i in b)
        d = np.array([len(i) for i in c.split('0')])
        return len(d[d >= 3])
    else:
        return 0


def get_hw(HIs_time):
    b = ma.masked_where(HIs_time < 2.8, HIs_time).filled(0)
    b = ma.masked_where(b != 0, b).filled(1)
    d_hw_3 = np.zeros_like(HIs_time, dtype=np.int16)  #3天
    c_hw_mode = []  #对热浪次数
    c_hw_max = []  #对热浪次数
    c_hw_mean = []  #对热浪次数
    c_sum_hw_hi = []  #对热浪次数hi3
    c_mean_hw_hi = []  #对热浪次数hi3
    c_median_hw_hi = []  #对热浪次数hi3
    c_std_hw_hi = []  #对热浪次数hi3
    c_range_hw_hi = []  #对热浪次数hi3
    #将不到3天且不达到重度的变为0,留下的按热浪等级标记
    danci_hw = []
    for k in range(len(b)):
        if b[k]:
            if HIs_time[k] < 6.5:
                danci_hw.append(1)
            elif 6.5 <= HIs_time[k] < 10.5:
                danci_hw.append(2)
            elif HIs_time[k] >= 10.5:
                danci_hw.append(3)
        else:
            hw_duration = len(danci_hw)
            if hw_duration >= 3:
                d_hw_3[k - hw_duration:k] = danci_hw
                c_hw_mode.append(stats.mode(danci_hw)[0][0])
                c_hw_max.append(np.max(danci_hw))
                m = np.mean(HIs_time[k - hw_duration:k])
                if m < 6.5:  #大于等于2.8
                    c_hw_mean.append(1)
                elif 6.5 <= m < 10.5:
                    c_hw_mean.append(2)
                elif m >= 10.5:
                    c_hw_mean.append(3)
                c_sum_hw_hi.append(np.sum(HIs_time[k - hw_duration:k]))
                c_mean_hw_hi.append(m)  #换成m
                c_median_hw_hi.append(
                    np.median(np.array(HIs_time[k - hw_duration:k])))
                c_std_hw_hi.append(np.std(HIs_time[k - hw_duration:k]))
                c_range_hw_hi.append(
                    np.max(HIs_time[k - hw_duration:k]) -
                    np.min(HIs_time[k - hw_duration:k]))
            danci_hw = []
    #检查最后一天是否为热浪
    hw_duration = len(danci_hw)
    if hw_duration >= 3:
        d_hw_3[-hw_duration:] = danci_hw
        c_hw_mode.append(stats.mode(danci_hw)[0][0])
        c_hw_max.append(np.max(danci_hw))
        m = np.mean(HIs_time[-hw_duration:])
        if m < 6.5:  #大于等于2.8
            c_hw_mean.append(1)
        elif 6.5 <= m < 10.5:
            c_hw_mean.append(2)
        elif m >= 10.5:
            c_hw_mean.append(3)
        c_sum_hw_hi.append(np.sum(HIs_time[-hw_duration:]))
        c_mean_hw_hi.append(m)  #换成m
        c_median_hw_hi.append(np.median(np.array(HIs_time[-hw_duration:])))
        c_std_hw_hi.append(np.std(HIs_time[-hw_duration:]))
        c_range_hw_hi.append(
            np.max(HIs_time[-hw_duration:]) - np.min(HIs_time[-hw_duration:]))

    return d_hw_3, np.array(c_hw_mode), np.array(c_hw_mean), np.array(
        c_hw_max), np.array(c_sum_hw_hi), np.array(c_mean_hw_hi), np.array(
            c_median_hw_hi), np.array(c_std_hw_hi), np.array(c_range_hw_hi)
    # return d_hw_3, np.array(c_hw_mode)


def summary(HIs_file, out_path, masks, ndv_int):
    HIs, im_proj, im_geotrans, im_height, im_width, ndv = openSingleImage(
        HIs_file)
    # mask = HIs[0]==ndv
    HIs = ma.masked_where(masks, HIs)
    # #热浪日数（3天）
    hw_days_1 = np.zeros_like(HIs[0], dtype=np.int16)
    hw_days_2 = np.zeros_like(HIs[0], dtype=np.int16)
    hw_days_3 = np.zeros_like(HIs[0], dtype=np.int16)
    #最长热浪持续日数
    hw_longgestdays = np.zeros_like(HIs[0], dtype=np.int16)
    # #热浪次数（3天，等级众数）
    # hw_fre_mode_1 = np.zeros_like(HIs[0], dtype=np.int16)
    # hw_fre_mode_2 = np.zeros_like(HIs[0], dtype=np.int16)
    # hw_fre_mode_3 = np.zeros_like(HIs[0], dtype=np.int16)
    # #热浪次数（3天，平均值）
    hw_fre_mean_1 = np.zeros_like(HIs[0], dtype=np.int16)
    hw_fre_mean_2 = np.zeros_like(HIs[0], dtype=np.int16)
    hw_fre_mean_3 = np.zeros_like(HIs[0], dtype=np.int16)
    # # #热浪次数（3天，最大值）
    # hw_fre_max_1 = np.zeros_like(HIs[0], dtype=np.int16)
    # hw_fre_max_2 = np.zeros_like(HIs[0], dtype=np.int16)
    # hw_fre_max_3 = np.zeros_like(HIs[0], dtype=np.int16)
    #最大热浪单日指数（3天）
    d_hw_max = np.zeros_like(HIs[0])
    # #最小热浪单日指数（3天）
    # d_hw_min = np.zeros_like(HIs[0])
    # #平均热浪单日指数（3天）
    # d_hw_mean = np.zeros_like(HIs[0])
    # #热浪单日指数中位数（3天）
    # d_hw_median = np.zeros_like(HIs[0])
    # #热浪单日指数标准差（3天）
    # d_hw_std = np.zeros_like(HIs[0])
    # #热浪单日指数之和（3天）
    # d_hw_sum = np.zeros_like(HIs[0])
    #最大热浪次HI之和（3天）
    c_hw_max_sum = np.zeros_like(HIs[0])
    # #最大热浪次HI平均值（3天）
    # c_hw_max_mean = np.zeros_like(HIs[0])
    # #最大热浪次HI中位数（3天）
    # c_hw_max_median = np.zeros_like(HIs[0])
    # #最大热浪次HI标准差（3天）
    # c_hw_max_std = np.zeros_like(HIs[0])
    #最大热浪次HI极差（3天）
    c_hw_max_range = np.zeros_like(HIs[0])
    # #最小热浪次HI之和（3天）
    # c_hw_min_sum = np.zeros_like(HIs[0])
    # #最小热浪次HI平均值（3天）
    # c_hw_min_mean = np.zeros_like(HIs[0])
    # #最小热浪次HI中位数（3天）
    # c_hw_min_median = np.zeros_like(HIs[0])
    # #最小热浪次HI标准差（3天）
    # c_hw_min_std = np.zeros_like(HIs[0])
    # #最小热浪次HI极差（3天）
    # c_hw_min_range = np.zeros_like(HIs[0])
    #最早热浪开始日期（3天）
    hw_start_date = np.zeros_like(HIs[0], dtype=np.int16)
    hw_start_date = ma.masked_where(hw_start_date == 0,
                                    hw_start_date).filled(ndv_int)
    #最晚热浪结束日期（3天）
    hw_end_date = np.zeros_like(HIs[0], dtype=np.int16)
    hw_end_date = ma.masked_where(hw_end_date == 0,
                                  hw_end_date).filled(ndv_int)
    #热旬期持续天数（3天）
    hw_max_duration = np.zeros_like(HIs[0], dtype=np.int16)

    HIs = np.transpose(HIs)
    with tqdm(range(im_width)) as t:
        for i in t:
            for j in range(im_height):
                if not HIs[i][j][0] is ma.masked:
                    max_HI = np.max(HIs[i][j])
                    b = ma.masked_where(HIs[i][j] < 2.8, HIs[i][j]).filled(0)
                    b = ma.masked_where(b != 0, b).filled(1)
                    fre = getHeatWaveFreq(b)
                    if max_HI == 0 or fre == 0:  #全是0 or #没有连续超过3天的热浪
                        continue
                    else:
                        d_hw_3, c_hw_mode_3, c_hw_mean_3, c_hw_max_3, c_sum_hw, c_mean_hw, c_median_hw, c_std_hw, c_range_hw = get_hw(
                            HIs[i][j])
                        # d_hw_3, c_hw_mode_3=get_hw(HIs[i][j])
                        # 热浪日数
                        hw_days_1[j][i] = len(d_hw_3[d_hw_3 == 1])
                        hw_days_2[j][i] = len(d_hw_3[d_hw_3 == 2])
                        hw_days_3[j][i] = len(d_hw_3[d_hw_3 == 3])
                        c = ''.join(str(i) for i in d_hw_3)
                        d = np.array([len(i) for i in c.split('0')])
                        hw_longgestdays[j][i] = np.max(d)

                        #热浪次数
                        # hw_fre_mode_1[j][i] = len(
                        #     c_hw_mode_3[c_hw_mode_3 == 1])
                        # hw_fre_mode_2[j][i] = len(
                        #     c_hw_mode_3[c_hw_mode_3 == 2])
                        # hw_fre_mode_3[j][i] = len(
                        #     c_hw_mode_3[c_hw_mode_3 == 3])
                        hw_fre_mean_1[j][i] = len(
                            c_hw_mean_3[c_hw_mean_3 == 1])
                        hw_fre_mean_2[j][i] = len(
                            c_hw_mean_3[c_hw_mean_3 == 2])
                        hw_fre_mean_3[j][i] = len(
                            c_hw_mean_3[c_hw_mean_3 == 3])
                        # hw_fre_max_1[j][i] = len(
                        #     c_hw_max_3[c_hw_max_3 == 1])
                        # hw_fre_max_2[j][i] = len(
                        #     c_hw_max_3[c_hw_max_3 == 2])
                        # hw_fre_max_3[j][i] = len(
                        #     c_hw_max_3[c_hw_max_3 == 3])

                        #热浪单日指数
                        mask_hw_3 = ma.masked_where(d_hw_3 != 0,
                                                    d_hw_3).filled(1)
                        hw_3_HIs = HIs[i][j] * mask_hw_3
                        hw_3_HIs = np.array(hw_3_HIs[hw_3_HIs != 0])
                        #if len(hw_3_HIs)!=0: #一定连续超过3天的热浪
                        d_hw_max[j][i] = np.max(hw_3_HIs)
                        # d_hw_min[j][i] = np.min(hw_3_HIs)
                        # d_hw_mean[j][i] = np.mean(hw_3_HIs)
                        # d_hw_median[j][i] = np.median(hw_3_HIs)
                        # d_hw_std[j][i] = np.std(hw_3_HIs)
                        # d_hw_sum[j][i] = np.sum(hw_3_HIs)

                        #热浪次内指数比较大小
                        c_hw_max_sum[j][i] = np.max(c_sum_hw)
                        # c_hw_min_sum[j][i] = np.min(c_sum_hw)
                        # c_hw_max_mean[j][i] = np.max(c_mean_hw)
                        # c_hw_min_mean[j][i] = np.min(c_mean_hw)
                        # c_hw_max_median[j][i] = np.max(c_median_hw)
                        # c_hw_min_median[j][i] = np.min(c_median_hw)
                        # c_hw_max_std[j][i] = np.max(c_std_hw)
                        # c_hw_min_std[j][i] = np.min(c_std_hw)
                        c_hw_max_range[j][i] = np.max(c_range_hw)
                        # c_hw_min_range[j][i] = np.min(c_range_hw)

                        #热浪起止日期和持续天数
                        for k in range(len(d_hw_3)):
                            if d_hw_3[k]:
                                hw_start_date[j][i] = k
                                break
                        for k in range(len(d_hw_3) - 1, -1, -1):
                            if d_hw_3[k]:
                                hw_end_date[j][i] = k + 1
                                break
                        hw_max_duration[j][
                            i] = hw_end_date[j][i] - hw_start_date[j][i]
                        del d_hw_3, c_hw_mode_3, c_hw_mean_3, c_hw_max_3, c_sum_hw, c_mean_hw, c_median_hw, c_std_hw, c_range_hw, max_HI, b, fre
    mask = masks[0]
    hw_days_1 = ma.masked_where(mask, hw_days_1).filled(ndv_int)
    hw_days_2 = ma.masked_where(mask, hw_days_2).filled(ndv_int)
    hw_days_3 = ma.masked_where(mask, hw_days_3).filled(ndv_int)
    #最长热浪持续日数
    hw_longgestdays = ma.masked_where(mask, hw_longgestdays).filled(ndv_int)
    # #热浪次数（3天，等级众数）
    # hw_fre_mode_1 = ma.masked_where(mask, hw_fre_mode_1).filled(ndv_int)
    # hw_fre_mode_2 = ma.masked_where(mask, hw_fre_mode_2).filled(ndv_int)
    # hw_fre_mode_3 = ma.masked_where(mask, hw_fre_mode_3).filled(ndv_int)
    #热浪次数（3天，平均值）
    hw_fre_mean_1 = ma.masked_where(mask, hw_fre_mean_1).filled(ndv_int)
    hw_fre_mean_2 = ma.masked_where(mask, hw_fre_mean_2).filled(ndv_int)
    hw_fre_mean_3 = ma.masked_where(mask, hw_fre_mean_3).filled(ndv_int)
    # #热浪次数（3天，最大值）
    # hw_fre_max_1 = ma.masked_where(mask, hw_fre_max_1).filled(ndv_int)
    # hw_fre_max_2 = ma.masked_where(mask, hw_fre_max_2).filled(ndv_int)
    # hw_fre_max_3 = ma.masked_where(mask, hw_fre_max_3).filled(ndv_int)
    #最大热浪单日指数（3天）
    d_hw_max = ma.masked_where(mask, d_hw_max).filled(ndv)
    # #最小热浪单日指数（3天）
    # d_hw_min = ma.masked_where(mask, d_hw_min).filled(ndv)

    # #平均热浪单日指数（3天）
    # d_hw_mean = ma.masked_where(mask, d_hw_mean).filled(ndv)
    # #热浪单日指数中位数（3天）
    # d_hw_median = ma.masked_where(mask, d_hw_median).filled(ndv)
    # #热浪单日指数标准差（3天）
    # d_hw_std = ma.masked_where(mask, d_hw_std).filled(ndv)
    # #热浪单日指数之和（3天）
    # d_hw_sum = ma.masked_where(mask, d_hw_sum).filled(ndv)

    #最大热浪次HI之和（3天）
    c_hw_max_sum = ma.masked_where(mask, c_hw_max_sum).filled(ndv)
    # #最大热浪次HI平均值（3天）
    # c_hw_max_mean = ma.masked_where(mask, c_hw_max_mean).filled(ndv)
    # #最大热浪次HI中位数（3天）
    # c_hw_max_median = ma.masked_where(mask, c_hw_max_median).filled(ndv)
    # #最大热浪次HI标准差（3天）
    # c_hw_max_std = ma.masked_where(mask, c_hw_max_std).filled(ndv)
    #最大热浪次HI极差（3天）
    c_hw_max_range = ma.masked_where(mask, c_hw_max_range).filled(ndv)
    # #最小热浪次HI之和（3天）
    # c_hw_min_sum = ma.masked_where(mask, c_hw_min_sum).filled(ndv)
    # #最小热浪次HI平均值（3天）
    # c_hw_min_mean = ma.masked_where(mask, c_hw_min_mean).filled(ndv)
    # #最小热浪次HI中位数（3天）
    # c_hw_min_median = ma.masked_where(mask, c_hw_min_median).filled(ndv)
    # #最小热浪次HI标准差（3天）
    # c_hw_min_std = ma.masked_where(mask, c_hw_min_std).filled(ndv)
    # #最小热浪次HI极差（3天）
    # c_hw_min_range = ma.masked_where(mask, c_hw_min_range).filled(ndv)

    #最早热浪开始日期（3天）
    hw_start_date = ma.masked_where(mask, hw_start_date).filled(ndv_int)
    #最晚热浪结束日期（3天）
    hw_end_date = ma.masked_where(mask, hw_end_date).filled(ndv_int)
    #热旬期持续天数（3天）
    hw_max_duration = ma.masked_where(mask, hw_max_duration).filled(ndv_int)

    year = HIs_file.split("_")[1][-4:]
    write_img(hw_days_1, f"{year}_热浪日数(3)_1.tif", im_proj, im_geotrans,
              out_path, ndv_int)
    write_img(hw_days_2, f"{year}_热浪日数(3)_2.tif", im_proj, im_geotrans,
              out_path, ndv_int)
    write_img(hw_days_3, f"{year}_热浪日数(3)_3.tif", im_proj, im_geotrans,
              out_path, ndv_int)
    write_img(hw_longgestdays, f"{year}_最长热浪持续日数.tif", im_proj, im_geotrans,
              out_path, ndv_int)
    # write_img(hw_fre_mode_1, f"{year}_热浪次数(mode)_1.tif", im_proj,
    #           im_geotrans, out_path, ndv_int)
    # write_img(hw_fre_mode_2, f"{year}_热浪次数(mode)_2.tif", im_proj,
    #           im_geotrans, out_path, ndv_int)
    # write_img(hw_fre_mode_3, f"{year}_热浪次数(mode)_3.tif", im_proj,
    #           im_geotrans, out_path, ndv_int)
    write_img(hw_fre_mean_1, f"{year}_热浪次数(mean)_1.tif", im_proj, im_geotrans,
              out_path, ndv_int)
    write_img(hw_fre_mean_2, f"{year}_热浪次数(mean)_2.tif", im_proj, im_geotrans,
              out_path, ndv_int)
    write_img(hw_fre_mean_3, f"{year}_热浪次数(mean)_3.tif", im_proj, im_geotrans,
              out_path, ndv_int)
    # write_img(hw_fre_max_1, f"{year}_热浪次数(max)_1.tif", im_proj,
    #           im_geotrans, out_path, ndv_int)
    # write_img(hw_fre_max_2, f"{year}_热浪次数(max)_2.tif", im_proj,
    #           im_geotrans, out_path, ndv_int)
    # write_img(hw_fre_max_3, f"{year}_热浪次数(max)_3.tif", im_proj,
    #           im_geotrans, out_path, ndv_int)
    write_img(d_hw_max, f"{year}_最大单日热浪指数(3).tif", im_proj, im_geotrans,
              out_path, ndv)
    # write_img(d_hw_min, f"{year}_最小单日热浪指数(3).tif", im_proj, im_geotrans,
    #           out_path, ndv)

    # write_img(d_hw_mean, f"{year}_平均单日热浪指数(3).tif", im_proj, im_geotrans,
    #           out_path, ndv)
    # write_img(d_hw_median, f"{year}_单日热浪指数中位数(3).tif", im_proj,
    #           im_geotrans, out_path, ndv)
    # write_img(d_hw_std, f"{year}_单日热浪指数标准差(3).tif", im_proj, im_geotrans,
    #           out_path, ndv)

    # write_img(c_hw_max_std, f"{year}_最大热浪HI指数标准差(3).tif", im_proj,
    #           im_geotrans, out_path, ndv)
    # write_img(c_hw_max_mean, f"{year}_最大热浪HI指数平均值(3).tif", im_proj,
    #           im_geotrans, out_path, ndv)
    # write_img(c_hw_max_median, f"{year}_最大热浪HI指数中位数(3).tif", im_proj,
    #           im_geotrans, out_path, ndv)
    write_img(c_hw_max_sum, f"{year}_最大热浪HI指数之和(3).tif", im_proj, im_geotrans,
              out_path, ndv)
    write_img(c_hw_max_range, f"{year}_最大热浪HI指数极差(3).tif", im_proj,
              im_geotrans, out_path, ndv)
    # write_img(c_hw_min_std, f"{year}_最小热浪HI指数标准差(3).tif", im_proj,
    #           im_geotrans, out_path, ndv)
    # write_img(c_hw_min_mean, f"{year}_最小热浪HI指数平均值(3).tif", im_proj,
    #           im_geotrans, out_path, ndv)
    # write_img(c_hw_min_median, f"{year}_最小热浪HI指数中位数(3).tif", im_proj,
    #           im_geotrans, out_path, ndv)
    # write_img(c_hw_min_sum, f"{year}_最小热浪HI指数之和(3).tif", im_proj,
    #           im_geotrans, out_path, ndv)
    # write_img(c_hw_min_range, f"{year}_最小热浪HI指数极差(3).tif", im_proj,
    #           im_geotrans, out_path, ndv)

    write_img(hw_start_date, f"{year}_热浪开始日期(3).tif", im_proj, im_geotrans,
              out_path, ndv_int)
    write_img(hw_end_date, f"{year}_热浪结束日期(3).tif", im_proj, im_geotrans,
              out_path, ndv_int)
    write_img(hw_max_duration, f"{year}_热浪旬期长(3).tif", im_proj, im_geotrans,
              out_path, ndv_int)

    del hw_days_1, hw_days_2, hw_days_3, hw_longgestdays, hw_fre_mean_1, hw_fre_mean_2, hw_fre_mean_3, d_hw_max, c_hw_max_sum, c_hw_max_range, hw_start_date, hw_end_date, hw_max_duration,
    # hw_fre_mode_1, hw_fre_mode_2, hw_fre_mode_3, d_hw_min, d_hw_mean, d_hw_median, d_hw_std, c_hw_max_std, c_hw_max_mean, c_hw_max_median, c_hw_min_std, c_hw_min_mean, c_hw_min_median, c_hw_min_sum, c_hw_min_range, hw_fre_max_1, hw_fre_max_3, hw_fre_max_2


#计算月TI变化
def calc_monthTI():
    TIs_path = r"G:\dem_chazhi\result\TI"
    out_path = r"G:\dem_chazhi\result\时间序列"
    TIs = [os.path.join(TIs_path, i) for i in os.listdir(TIs_path)]
    months = ["05", "06", "07", "08", "09"]
    ymonths = sorted(list(set([i[-12:-6] for i in TIs])))
    for ymonth in ymonths:
        if not os.path.exists(os.path.join(out_path, ymonth + ".tif")):
            TIs_cal = [i for i in TIs if i[-12:-6] == ymonth]
            Igs, im_geotrans, im_proj, ndv = openImages(TIs_cal)
            Igs = ma.masked_where(Igs == ndv, Igs)
            avg_data = np.average(Igs, axis=0)
            avg_data = ma.masked_where(np.isnan(avg_data),
                                       avg_data).filled(ndv)

            write_img(avg_data, ymonth + ".tif", im_proj, im_geotrans,
                      out_path, ndv)
            del Igs
    for month in months:
        if not os.path.exists(os.path.join(out_path, month + ".tif")):
            TIs_cal = sorted([
                os.path.join(out_path, i) for i in os.listdir(out_path)
                if len(i) == 10 and i[-6:-4] == month
            ])
            Igs, im_geotrans, im_proj, ndv = openImages(TIs_cal)
            Igs = ma.masked_where(Igs == ndv, Igs)
            avg_data = np.sum(Igs, axis=0) / 30
            avg_data = ma.masked_where(np.isnan(avg_data),
                                       avg_data).filled(ndv)

            write_img(avg_data, month + ".tif", im_proj, im_geotrans, out_path,
                      ndv)
            del Igs
    for month in months:
        if not os.path.exists(os.path.join(out_path, f"slope_{month}.tif")):
            TIs_cal = sorted([
                os.path.join(out_path, i) for i in os.listdir(out_path)
                if len(i) == 10 and i[-6:-4] == month
            ])
            Igs, im_geotrans, im_proj, ndv = openImages(TIs_cal)
            slope_data = np.full_like(Igs[0], ndv, dtype=float)
            import scipy.stats as st
            height = slope_data.shape[0]
            width = slope_data.shape[1]
            Igs = ma.masked_where(Igs == ndv, Igs)
            Igs = np.transpose(Igs)
            x = list(range(len(TIs_cal)))
            from tqdm import trange
            for i in trange(width):
                for j in range(height):
                    if Igs[i][j][0] is not ma.masked:
                        y = Igs[i][j]
                        slope, intercept, r_value, p_value, std_err = st.linregress(
                            x, y)
                        if p_value < 0.05:
                            slope_data[j][i] = slope
            write_img(slope_data, f"slope_{month}.tif", im_proj, im_geotrans,
                      out_path, ndv)
            del Igs
    if not os.path.exists(os.path.join(out_path, f"slope_whole.tif")):
        TIs_cal = np.sort([
            os.path.join(out_path, i) for i in os.listdir(out_path)
            if len(i) == 10 and i.endswith("tif")
        ])
        Igs, im_geotrans, im_proj, ndv = openImages(TIs_cal)
        slope_data = np.full_like(Igs[0], ndv, dtype=float)
        import scipy.stats as st
        height = slope_data.shape[0]
        width = slope_data.shape[1]
        Igs = ma.masked_where(Igs == ndv, Igs)
        Igs = np.transpose(Igs)
        x = list(range(len(TIs_cal)))
        from tqdm import trange
        for i in trange(width):
            for j in range(height):
                if Igs[i][j][0] is not ma.masked:
                    y = Igs[i][j]
                    slope, intercept, r_value, p_value, std_err = st.linregress(
                        x, y)
                    if p_value < 0.05:
                        slope_data[j][i] = slope
        write_img(slope_data, f"slope_whole.tif", im_proj, im_geotrans,
                  out_path, ndv)
        del Igs


#计算热浪日数与天数和并拟合
def combine(start_year=1990, end_year=2020):
    dir_path = r"G:\dem_chazhi\result\热浪不同指标结果"
    out_path = r"G:\dem_chazhi\result\时间序列"
    # for year in range(1990, 2000):
    #     HWC_name = os.path.join(dir_path, f"{year}_热浪次数(mean).tif")
    #     if not os.path.exists(HWC_name):
    #         Igs, im_geotrans, im_proj, ndv = openImages([
    #             fr"{dir_path}\{year}_热浪次数(mean)_1.tif",
    #             fr"{dir_path}\{year}_热浪次数(mean)_2.tif",
    #             fr"{dir_path}\{year}_热浪次数(mean)_3.tif"
    #         ])
    #         Ig_out = np.sum(Igs, axis=0)
    #         Ig_out = ma.masked_where(Igs[0] == ndv, Ig_out).filled(ndv)
    #         write_img(Ig_out, f"{year}_热浪次数(mean).tif", im_proj, im_geotrans,
    #                   dir_path, ndv)

    #     HWR_name = os.path.join(dir_path, f"{year}_热浪日数(3).tif")
    #     if not os.path.exists(HWR_name):
    #         Igs, im_geotrans, im_proj, ndv = openImages([
    #             fr"{dir_path}\{year}_热浪日数(3)_1.tif",
    #             fr"{dir_path}\{year}_热浪日数(3)_2.tif",
    #             fr"{dir_path}\{year}_热浪日数(3)_3.tif"
    #         ])
    #         Ig_out = np.sum(Igs, axis=0)
    #         Ig_out = ma.masked_where(Igs[0] == ndv, Ig_out).filled(ndv)
    #         write_img(Ig_out, f"{year}_热浪日数(3).tif", im_proj, im_geotrans,
    #                   dir_path, ndv)

    HWCs_1 = [
        fr"{dir_path}\{year}_热浪次数(mean)_1.tif"
        for year in range(start_year, end_year)
    ]
    HWCs_2 = [
        fr"{dir_path}\{year}_热浪次数(mean)_2.tif"
        for year in range(start_year, end_year)
    ]
    HWCs_3 = [
        fr"{dir_path}\{year}_热浪次数(mean)_3.tif"
        for year in range(start_year, end_year)
    ]
    HWCs = [
        fr"{dir_path}\{year}_热浪次数(mean).tif"
        for year in range(start_year, end_year)
    ]
    HWRs_1 = [
        fr"{dir_path}\{year}_热浪日数(3)_1.tif"
        for year in range(start_year, end_year)
    ]
    HWRs_2 = [
        fr"{dir_path}\{year}_热浪日数(3)_2.tif"
        for year in range(start_year, end_year)
    ]
    HWRs_3 = [
        fr"{dir_path}\{year}_热浪日数(3)_3.tif"
        for year in range(start_year, end_year)
    ]
    HWRs = [
        fr"{dir_path}\{year}_热浪日数(3).tif"
        for year in range(start_year, end_year)
    ]
    HWMDs = [
        fr"{dir_path}\{year}_最长热浪持续日数.tif"
        for year in range(start_year, end_year)
    ]
    HWMHIs = [
        fr"{dir_path}\{year}_最大单日热浪指数(3).tif"
        for year in range(start_year, end_year)
    ]
    HWEDs = [
        fr"{dir_path}\{year}_热浪开始日期(3).tif"
        for year in range(start_year, end_year)
    ]
    HWSDs = [
        fr"{dir_path}\{year}_热浪结束日期(3).tif"
        for year in range(start_year, end_year)
    ]
    HWMRHIs = [
        fr"{dir_path}\{year}_最大热浪HI指数极差(3).tif"
        for year in range(start_year, end_year)
    ]
    indexs = [
        HWCs_1, HWCs_2, HWCs_3, HWCs, HWRs_1, HWRs_2, HWRs_3, HWRs, HWMDs,
        HWMHIs, HWEDs, HWSDs, HWMRHIs
    ]
    for index in indexs:
        avg_name = "avg_" + "_".join(
            index[0].split("\\")[-1].split("_")[1:])[:-4] + "_" + str(
                start_year) + "_" + str(end_year) + ".tif"
        filename = "slope_" + "_".join(
            index[0].split("\\")[-1].split("_")[1:])[:-4] + "_" + str(
                start_year) + "_" + str(end_year) + ".tif"
        if not os.path.exists(os.path.join(out_path, avg_name)):
            Igs, im_geotrans, im_proj, ndv = openImages(index)
            Igs = ma.masked_equal(Igs, ndv)

            avg_data = np.mean(Igs, axis=0).filled(ndv)
            write_img(avg_data, avg_name, im_proj, im_geotrans, out_path, ndv)

        # if not os.path.exists(os.path.join(out_path, filename)):
        #     Igs, im_geotrans, im_proj, ndv = openImages(index)
        #     slope_data = np.full_like(Igs[0], ndv, dtype=float)
        #     height = slope_data.shape[0]
        #     width = slope_data.shape[1]
        #     Igs = ma.masked_where(Igs == ndv, Igs)
        #     Igs = np.transpose(Igs)

        #     import scipy.stats as st
        #     from tqdm import trange
        #     for i in trange(width):
        #         for j in range(height):
        #             if Igs[i][j][0] is not ma.masked:
        #                 y = Igs[i][j]
        #                 if ma.masked in y:
        #                     y = np.array([i for i in y if i is not ma.masked])
        #                 x = list(range(len(y)))
        #                 if len(y) < 10:
        #                     continue
        #                 slope, intercept, r_value, p_value, std_err = st.linregress(
        #                     x, y)
        #                 if p_value < 0.05:
        #                     slope_data[j][i] = slope
        #     write_img(slope_data, filename, im_proj, im_geotrans, out_path,
        #               ndv)
        #     del Igs


if __name__ == "__main__":
    # data_path = r"I:/dem_chazhi/result"
    # tems_path = f"{data_path}/TEM"
    # rhus_path = f"{data_path}/RHU"
    # tem_files = [
    #     f"{tems_path}/{i}" for i in os.listdir(tems_path) if i.endswith("tif")

    # ]
    # HIs_path = r"G:\dem_chazhi\result\HI"
    # out_path = r"G:\dem_chazhi\result\热浪不同指标结果"
    # HIs_files = [
    #     f"{HIs_path}/{i}" for i in os.listdir(HIs_path) if i.endswith("tif")
    # ]
    # HIs_files = HIs_files[:10]
    # ndv_int = -32768
    # mm, _, _, _, _, ndv = openSingleImage(r"G:\dem_chazhi\result\TI'_copy.tif")
    # mask = mm == ndv
    # masks = []
    # for i in range(153):
    #     masks.append(mask)
    # process_num = 1
    # pool = Pool(process_num)
    # tmp = []
    # for i in range(len(HIs_files)):
    #     res = pool.apply_async(func=summary,
    #                            args=(HIs_files[i], out_path, masks, ndv_int),
    #                            callback=None)
    # pool.close()
    # pool.join()
    # calc_the_heat_threshold()
    # calc_TIs()
    # calc_TIpie()
    # calc_TIpie_2()
    # calc_HI()
    # calc_monthTI()
    combine()
