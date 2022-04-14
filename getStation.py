import os
from tqdm import tqdm
from tqdm import trange
from osgeo import osr
stations = []


def proj(
    latitude,
    longitude,
):
    """将地理坐标系投影为投影坐标系

    Args:
        latitude ([type]): 纬度
        longitude ([type]): 经度

    Returns:
        list: [x,y](经、纬)
    """
    geosrs = osr.SpatialReference()
    prosrs = osr.SpatialReference()
    prosrs.ImportFromWkt(
        'PROJCS["Asia_North_Albers_Equal_Area_Conic",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Albers_Conic_Equal_Area"],PARAMETER["latitude_of_center",0],PARAMETER["longitude_of_center",105],PARAMETER["standard_parallel_1",25],PARAMETER["standard_parallel_2",47],PARAMETER["false_easting",4000000],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'
    )
    #改为你需要的投影信息
    geosrs.SetWellKnownGeogCS("WGS84")
    ct = osr.CoordinateTransformation(geosrs, prosrs)

    coords = ct.TransformPoint(latitude, longitude, 0)
    return coords[:2]


with open(r"E:\研究生毕业论文\实验部分\气象站点数据.txt", "w") as oo:  # 存放全部气象站点信息的文件带后缀
    for dirpath, _, filenames in os.walk(
            r"E:\研究生毕业论文\数据和处理（原始资料）\中国气象站点数据_from国家气象数据中心\温度和相对湿度_待处理"
    ):  # 存放所有要处理文件的文件夹（无子目录）
        with tqdm(filenames) as t:
            for filename in t:
                with open(os.path.join(dirpath, filename), "r") as oi:
                    for line in oi.readlines():
                        lineStr = line.split()
                        station = lineStr[0]
                        if station not in stations:
                            latitude = float(
                                lineStr[1][:-2]) + float(lineStr[1][-2:]) / 60
                            longitude = float(
                                lineStr[2][:-2]) + float(lineStr[2][-2:]) / 60
                            longitude, latitude = proj(latitude, longitude)
                            if int(lineStr[3]) > 100000:
                                elevation = (float(lineStr[3]) -
                                             100000) * 0.1  #高程
                            else:
                                elevation = float(lineStr[3]) * 0.1  #高程
                            oo.write(
                                f"{station}\t{longitude}\t{latitude}\t{elevation}\n"
                            )
                            stations.append(station)
