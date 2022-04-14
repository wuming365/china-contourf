import pandas as pd
import os
txt_path = r"E:\研究生毕业论文\实验部分\气象站点数据\气象站点数据_经纬度_使用.txt"
df = pd.read_table(txt_path, sep=",")
dongbei = list(df["station_id"])

tems = [i for i in os.listdir(r"E:\研究生毕业论文\实验部分\1990-2020文本数据") if "TEM" in i]
dic_stationid = {}
dic_times = {}
for i in dongbei:
    dic_times[str(i)] = 0
for tem in tems:
    with open(r"E:\研究生毕业论文\实验部分\1990-2020文本数据\\" + tem) as o:
        for line in o.readlines():
            if float(line.split()[4]) > 33 and int(line.split()[0]) in dongbei:
                if line.split()[0] in dic_stationid.keys():
                    dic_stationid[line.split()[0]].append(
                        tem.split(".")[0].split("_")[-1])
                    dic_times[line.split()[0]] += 1
                else:
                    dic_stationid[line.split()[0]] = [
                        tem.split(".")[0].split("_")[-1]
                    ]

df1 = pd.DataFrame(dic_stationid)
df2 = pd.DataFrame(dic_times)

print(df1)
