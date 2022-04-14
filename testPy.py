import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import os
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def statistics(tValue,vValue,dValue):
    """[summary]

    Args:
        tValue ([type]): real value
        vValue ([type]): predictive value
        dValue ([type]): abs difference of two values

    Returns:
        slope,r_square,pt,mae,rmse
    """
    n=len(tValue)
    rmse=np.sqrt(np.sum(dValue*dValue)/n)
    mae=np.mean(dValue)
    xx,xy=0,0
    for i in range(len(tValue)):
        xx+=np.power(vValue[i],2)
        xy+=vValue[i]*tValue[i]
    slope=xy/xx
    y_prd= slope * vValue
    regression=np.sum((y_prd-np.mean(tValue))**2)
    residual=np.sum((tValue-y_prd)**2)
    # print(regression,residual)
    r_square=regression/(regression+residual)
    
    t=slope * np.sqrt(n*np.var(vValue))/np.sqrt(residual/(n-2))
    pt=stats.t.sf(t,n-2)
    return slope,r_square,pt,mae,rmse

def mapping(path_data,path_zuotu,lan="ch"):
    df=pd.read_csv(path_data,usecols=['tValue','date','series','vValue','month','year'])
    df=df.iloc[:81805,:]
    df["date"]=df["date"].astype("int").astype("str")
    df['month']=df['month'].astype("int").astype("str")
    df['year']=df['year'].astype("int").astype("str")
    df1=pd.read_csv(path_zuotu,usecols=['date','series','slope','r_square'])

    df2=pd.DataFrame()
    name=""
    for series in df1["series"].unique():
        i=0
        letter = list(map(chr, range(ord('a'), ord('z') + 1)))
        fig=plt.figure(figsize=(15,15))
        fig.subplots_adjust(right=0.95,left=0.05,top=0.95,bottom=0.05,hspace=0.15,wspace=0.15)
        if lan=="ch":
            if "RHU" in series:
                fig.suptitle('相对湿度（%）',fontsize=20)
                name="相对湿度"
            else:
                fig.suptitle('温度（℃）',fontsize=20)
                name="温度"
        elif lan=="en":
            if "RHU" in series:
                fig.suptitle('Relative Humidity(%)',fontsize=20)
            else:
                fig.suptitle('Temperature(℃)',fontsize=20)
                
        df3=df1[df1["series"]==series]
        for row in df3.itertuples():
            series=getattr(row,'series')
            date=str(getattr(row,'date'))
            if len(date)==8:
                df2=df[(df['series']==series) & (df['date']==date)]
            elif len(date)==6:
                df2=df[(df['series']==series) & (df['month']==date)]
            elif len(date)==4:
                df2=df[(df['series']==series) & (df['year']==date)]
            tValue=df2["tValue"]
            vValue=df2["vValue"]
            slope=getattr(row,"slope")
            r_square=getattr(row,'r_square')
            ax=fig.add_subplot(3,3,i+1)
            ax.text(0.05,0.95,horizontalalignment='left',verticalalignment='top',s=letter[i],transform=ax.transAxes,fontsize=20,fontweight='black')
            i+=1
            ax.scatter(vValue,tValue,s=10)
            x=np.array([0,np.max(vValue)+1])
            y=slope*x
            ax.plot(x,y,linestyle='--')
            # ax.set_title(date)
            if lan=="ch":
                ax.set_xlabel("预测值")
                ax.set_ylabel("真实值")
            elif lan=="en":
                ax.set_xlabel("Predictive value")
                ax.set_ylabel("Real value")    
            ax.text(0.95,0.05,horizontalalignment='right',verticalalignment='bottom',s=f"$y={slope:.4f}x$\n$R^2={r_square:.4f}$",transform=ax.transAxes,fontsize=10)
        plt.savefig(fr'E:\高温热浪危险性论文\图片\{name}.png')#保存图片

def main(path,path_zuotu):
    df=pd.read_csv(path,usecols=['tValue','date','series','vValue','dValue'])
    df=df.iloc[:81805,:]
    df["date"]=pd.to_datetime(df["date"].astype("int"),format="%Y%m%d")
    df["year"]=df["date"].dt.year
    df["month"]=df["date"].dt.strftime("%Y%m")
    df["date"]=df["date"].dt.strftime("%Y%m%d")
    df2=pd.DataFrame(columns=['date','series','slope','r_square','pt','mae','rmse'])
    df4=pd.DataFrame()
    for series in df["series"].unique():
        #按日
        df1=df[df["series"]==series]
        for date in df["date"].unique():
            tValue=np.array(df1[df1["date"]==date]["tValue"])
            vValue=np.array(df1[df1["date"]==date]["vValue"])
            dValue=np.abs(np.array(df1[df1["date"]==date]["dValue"],dtype=np.float))
            slope,r_square,pt,mae,rmse=statistics(tValue,vValue,dValue)
            df3=pd.DataFrame({'date':date,'series':series,'slope':slope,'r_square':r_square,'pt':pt,'mae':mae,'rmse':rmse},index=[0])
            df2=df2.append(df3)
        for month in df["month"].unique():
            tValue=np.array(df1[df1["month"]==month]["tValue"])
            vValue=np.array(df1[df1["month"]==month]["vValue"])
            dValue=np.abs(np.array(df1[df1["month"]==month]["dValue"],dtype=np.float))
            slope,r_square,pt,mae,rmse=statistics(tValue,vValue,dValue)
            df3=pd.DataFrame({'date':month,'series':series,'slope':slope,'r_square':r_square,'pt':pt,'mae':mae,'rmse':rmse},index=[0])
            df2=df2.append(df3)
        for year in df["year"].unique():
            tValue=np.array(df1[df1["year"]==year]["tValue"])
            vValue=np.array(df1[df1["year"]==year]["vValue"])
            dValue=np.abs(np.array(df1[df1["year"]==year]["dValue"],dtype=np.float))
            slope,r_square,pt,mae,rmse=statistics(tValue,vValue,dValue)
            df3=pd.DataFrame({'date':year,'series':series,'slope':slope,'r_square':r_square,'pt':pt,'mae':mae,'rmse':rmse},index=[0])
            df2=df2.append(df3)
        df3=df2[df2["series"]==series].sort_values(by="r_square",ascending=False)
        day=[i for i in np.array(df3["date"],dtype=str) if len(i)==8][:4]
        month=[i for i in np.array(df3["date"],dtype=str) if len(i)==6][:3]
        year=[int(i) for i in np.array(df3["date"],dtype=str) if len(i)==4][:2]
        date2pic=day+month+year
        for date in date2pic:
            df4=df4.append(df3[df3["date"]==date])
    df4.to_csv(path_zuotu,index=False)


        
# df2.to_csv(r"E:\研究生毕业论文\实验部分\插值精度\结果.csv")
if __name__ == "__main__":
    path_zuotu=r"E:\研究生毕业论文\实验部分\插值精度\作图.csv"
    path_data=r"E:\研究生毕业论文\实验部分\插值精度\updatePoint.csv"
    # main(path_data,path_zuotu)
    mapping(path_data,path_zuotu)