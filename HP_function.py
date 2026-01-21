import pandas as pd
import datetime as dt
import numpy as np

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

#滤波函数 data:data数据 index:columns name列名  start_date开始时间   end_date结束时间   lamb: filter parameter HP滤波的参数
def HP_filter(data, index, start_date='2024-12-31', end_date='2005-07-31', lamb=14400, seasonal_adjusted=False):
    # 去除空值
    df = data.loc[:, index].to_frame()
    df.dropna(axis=0, inplace=True)
    df = df.loc[start_date:end_date]

    # 是否进行季节性调整
    if seasonal_adjusted == True:
        df.reset_index(inplace=True)
        df['Month'] = df['Date'].dt.month
        Mnth_average = df.groupby('Month')[index].mean()  #当月的均值
        Mnth_average.name = 'month_average'
        # print(Mnth_average)
        df = df.merge(Mnth_average, on=['Month'], how='inner')
        all_average = df[index].mean()  #数据整体的均值
        df.set_index('Date', inplace=True)
        df[index + '_adjusted'] = df[index] * all_average / df['month_average']  #调整公式： 值*总体均值/当月均值
        df.sort_index(ascending=False, inplace=True)
        df.drop([index, 'Month', 'month_average'], axis=1, inplace=True)
        index += '_adjusted'

    # HP滤波
    cycle, trend = sm.tsa.filters.hpfilter(df, lamb)
    df['trend'] = trend
    df['cycle'] = cycle

    # 绘图
    fig = px.line(df[index], title='Original Data')  #
    fig.update_traces(line_color='red')
    fig.show()
    fig = px.line(df['trend'], title='HP filter for ' + index)
    fig.update_layout(yaxis2=dict(title='Residual', overlaying='y', side='right'))
    fig.add_trace(go.Bar(x=df.index, y=df["cycle"], name='Residual', yaxis='y2', marker_color='green'))
    fig.update_layout(title='HP filter for ' + index, xaxis_title='Date', yaxis_title='Trend')
    fig.show()

    # 计算目标函数值
    df['fit'] = (df['cycle']) ** 2
    smooth = np.sum(np.diff(df['trend'], 2) ** 2)
    objective_value = df['fit'].sum() + lamb * smooth
    print('目标函数值：', objective_value)

    # 差分
    df['diff'] = df['trend'] - df['trend'].shift(-1)

    # 导入xlsx
    index += str(start_date)[:10]
    df.to_excel('Output/Data/' + index + '.xlsx', index=True) # Save data under Output filefolder
    return df

#与十年期国债进行比较——判断拐点 Filter_data: 滤波后的数据 YTM: 对比的数据  index: 分析的列（用于图片命名）  Shift:偏移量 +向前偏移；-向后偏移
def compare_plot(Filter_data,YTM,index,Shift = 0):
    #数据和并
    Merge_df = YTM.merge(Filter_data[['trend','diff']],on = 'Date',how = 'outer')  #merge dataframe for plot 合并数据
    Merge_df = Merge_df.sort_index(ascending = False) 
    non_null_indices = Merge_df['trend'].notna() #locate the NaN in trend columns, where lacking the original data  
    #确定空值位置（因为滤波数据和收益率数据长度不一致，通过空值将两者对其）
    non_null_index_positions = Merge_df.loc[non_null_indices == True].index  #find index of NaN
    Merge_df = Merge_df.loc[:non_null_index_positions[-1]] #slice the dataframe from the earlist non_NaN time to the latest 从最晚的空值开始切割
    #对于低频率的数据进行线性差值，不会改变拐点的判断和图形走势，仅仅为了绘图方便以及美观
    Merge_df['trend'] = Merge_df['trend'].interpolate(method='linear').shift(Shift) #interpolate for better plot result, since Original Data -> Weekly/Monthly
    Merge_df['diff'] = Merge_df['diff'].interpolate(method='linear').shift(Shift)

    #绘图1：滤波后数据和收益率比对
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(Merge_df.index, Merge_df['YTM'], label='10 year Treasury Bond YTM', color='#44a2f6', marker='') #绘图
    ax.set_xlabel('Date') #设置轴名称
    ax.set_ylabel('YTM(%)', color='#000000') 
    ax.tick_params(axis='y', labelcolor='#000000')
    ax.spines[['top','bottom']].set_visible(False) #将上下边框去除
    ax.xaxis.set_ticks_position('none')
    ax1 = ax.twinx()  # 创建一个共享 X 轴的副轴
    ax1.plot(Merge_df.index, Merge_df['trend'], label='Trend', color='#FF3300', marker='') #绘图
    ax1.set_ylabel('Trend', color='#000000')
    ax1.tick_params(axis='y', labelcolor='#000000')
    ax1.spines[['top','bottom']].set_visible(False) #将上下边框去除
    ax1.xaxis.set_ticks_position('none')
    plt.title('Trend vs 10 year Treasury Bond') #命名
    ax.legend(loc='upper left')
    ax1.legend(loc='upper right')
    plt.savefig('Output/Image/'+index+'_Trend.pdf', dpi = 600, bbox_inches='tight') #保存
    plt.show()

    #绘图2：滤波后数据的差分和收益率比对 过程与上面绘图1相同
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(Merge_df.index, Merge_df['YTM'], label='10 year Treasury Bond YTM', color='#44a2f6', marker='')
    ax.set_xlabel('Date')
    ax.set_ylabel('YTM(%)', color='#000000')
    ax.tick_params(axis='y', labelcolor='#000000')
    ax.spines[['top','bottom']].set_visible(False)
    ax.xaxis.set_ticks_position('none')
    ax1 = ax.twinx()  # 创建一个共享 X 轴的副轴
    ax1.plot(Merge_df.index, Merge_df['diff'], label='Trend', color='#FF3300', marker='')
    ax1.set_ylabel('Trend diff', color='#000000')
    ax1.tick_params(axis='y', labelcolor='#000000')
    ax1.spines[['top','bottom']].set_visible(False)
    ax1.xaxis.set_ticks_position('none')
    plt.title('Trend diff vs 10 year Treasury Bond')
    ax.legend(loc='upper left')
    ax1.legend(loc='upper right')
    plt.savefig('Output/Image/'+index+'_Trend diff.pdf', dpi = 600, bbox_inches='tight')
    plt.show()

    return None