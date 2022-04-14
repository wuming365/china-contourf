import os
import pandas as pd
from icecream import ic
from tqdm import tqdm
file = r"../气象站点数据/气象站点分层随机抽样.xlsx"
data = pd.read_excel(
    file,
    sheet_name=['时间点', '站点号'],
    converters={
        "dlfq": str,
        "qhfq": str,
        "jieti": str,
        "dxfz": str,
    },
    usecols=[0, 1, 2, 3],
)
timePoint = data['时间点'].values.transpose()
stationId = data['站点号'].values.transpose()
info = {}
for i, name in enumerate(data['时间点'].keys()):
    info[name] = {"timePoint": timePoint[i], "stationId": stationId[i]}

pathNeedProcess = "../文本_验证点提取/处理后_插值文本/"
pathVeri = "../文本_验证点提取/验证点/"
RHU = "RHU-13003_"
TEM = "TEM-12001_"
fileLog = "TIMEPOINT\tRHU\tTEM\tMETHOD\n"
for i in info.keys():  #不同分区方法
    for j in tqdm(info[i]["timePoint"], desc=i):
        fileRhu = ""
        fileTem = ""
        pathRhu = os.path.join(pathNeedProcess, RHU + j + ".txt")
        pathTem = os.path.join(pathNeedProcess, TEM + j + ".txt")
        pathRhuVeri = os.path.join(pathVeri, RHU + i + "_" + j + ".txt")
        pathTemVeri = os.path.join(pathVeri, TEM + i + "_" + j + ".txt")
        numRhu = 0
        numTem = 0
        try:
            with open(pathRhu, "r") as rhu, open(pathTem, "r") as tem:
                with open(pathRhuVeri, "w") as rhuVeri, open(pathTemVeri,
                                                             "w") as temVeri:
                    for line in rhu:
                        if line.split()[0] not in info[i]["stationId"]:
                            fileRhu += line
                        else:
                            numRhu += 1
                            rhuVeri.write(line)
                    for line in tem:
                        if line.split()[0] not in info[i]["stationId"]:
                            fileTem += line
                        else:
                            numTem += 1
                            temVeri.write(line)
        except:
            continue
        fileLog += f"{j}\t{str(numRhu)}\t{str(numTem)}\t{i}\n"
        with open(pathRhu, "w") as rhu, open(pathTem, "w") as tem:
            rhu.write(fileRhu)
            tem.write(fileTem)

with open("../文本_验证点提取/log.txt", "w") as log:
    log.write(fileLog)