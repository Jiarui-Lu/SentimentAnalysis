"""
第五步:
历史数据回测
采取沪深300指数的收盘价
通过调取测算出的情绪表来进行多空信号交易
情绪为积极时，看多市场；情绪为消极时，看空市场
对持仓信号进行差分形成交易信号
考虑到时滞性，对市场情绪滞后三天后再进行交易
最终回测与传统MACD策略和基准指数对比查看结果

"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置字体 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']

# 用来正常显示负号
plt.rcParams['axes.unicode_minus'] = False
# 图表主题
plt.style.use('ggplot')


def macd(signals, ma1, ma2, dif):
    signals['ma1'] = signals['close'].rolling(window=ma1, min_periods=1, center=False).mean()
    signals['ma2'] = signals['close'].rolling(window=ma2, min_periods=1, center=False).mean()
    signals['dif'] = (signals['ma1'] - signals['ma2']).rolling(window=dif, min_periods=1, center=False).mean()

    return signals


def signal_generation(df):
    sentiment_df = pd.read_excel(r'result\sentiment_data.xls', index_col=0)
    # print(df)
    # print(sentiment_df)
    df['sentiment'] = 0
    df.loc[df.index[:len(sentiment_df.index)], 'sentiment'] = sentiment_df['sentiment']
    df['positions'] = np.where(df['sentiment'] == 1, 1, 0)
    df['signals'] = df['positions'].diff()
    # 收益
    df['ret'] = df['close'].pct_change()
    # 基准净值
    benchmark = (1 + df['ret']).cumprod()
    df['signals'] = df['signals'].shift(3)
    # 策略净值
    CUM = (1 + df['signals'] * df['ret']).cumprod()
    ma1 = 12
    ma2 = 26
    dif = 9
    df = macd(df, ma1=ma1, ma2=ma2, dif=dif)
    df['macd_positions'] = 0
    df['macd_positions'][ma1:] = np.where(df['dif'][ma1:] > 0, 1, 0)
    df['macd_signals'] = df['macd_positions'].diff()
    macd_CUM = (1 + df['macd_signals'] * df['ret']).cumprod()
    # 画图
    plt.figure()
    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(1, 1, 1)

    ax1.plot(df.index, CUM, label='情感分析策略')
    ax1.plot(df.index, benchmark, label='沪深300')
    ax1.plot(df.index, macd_CUM, label='传统MACD策略')

    plt.legend(loc='best')
    plt.xlabel('时间')
    plt.ylabel('净值')
    plt.title('各策略净值曲线对比')
    plt.savefig(r'result\return backtest.jpg')
    plt.show()
    return df


def LongShortPlot(new, ticker):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    new['close'].plot(label=ticker)
    ax.plot(new.loc[new['signals'] == 1].index, new['close'][new['signals'] == 1], label='LONG', lw=0, marker='^',
            c='g')
    ax.plot(new.loc[new['signals'] == -1].index, new['close'][new['signals'] == -1], label='SHORT', lw=0, marker='v',
            c='r')

    plt.legend(loc='best')
    plt.grid(True)
    plt.title('Positions')
    plt.savefig(r'result\positions_strategy.jpg')
    plt.show()


# 风险报告
def summary(df):
    summary_dic = {}
    index_name = '年化收益率,累计收益率,夏普比率,最大回撤,持仓总天数,交易次数,' \
                 '平均持仓天数,获利天数,亏损天数,胜率(按天),平均盈利率(按天),平均亏损率(按天),' \
                 '平均盈亏比(按天),盈利次数,亏损次数,单次最大盈利,单次最大亏损,' \
                 '胜率(按此),平均盈利率(按次),平均亏损率(按次),平均盈亏比(按次)'.split(
        ',')
    signal_name = ['signals', 'macd_signals']
    col_name = ['情感分析策略', '传统MACD策略']

    def format_x(x):
        return '{:.2%}'.format(x)

    for signal in signal_name:
        RET = df['ret'] * df[signal]
        CUM_RET = (1 + RET).cumprod()

        # 计算年华收益率
        annual_ret = CUM_RET[-1] ** (250 / len(RET)) - 1

        # 计算累计收益率
        cum_ret_rate = CUM_RET[-1] - 1

        # 最大回撤
        max_nv = np.maximum.accumulate(np.nan_to_num(CUM_RET))
        mdd = -np.min(CUM_RET / max_nv - 1)

        # 夏普
        sharpe_ratio = np.mean(RET) / np.nanstd(RET, ddof=1) * np.sqrt(250)

        # 标记买入卖出时点
        mark = df[signal]
        pre_mark = np.nan_to_num(df[signal].shift(-1))
        # 买入时点
        trade = (mark == 1) & (pre_mark < mark)

        # 交易次数
        trade_count = np.nansum(trade)

        # 持仓总天数
        total = np.sum(mark)
        # 平均持仓天数
        mean_hold = total / trade_count
        # 获利天数
        win = np.sum(np.where(RET > 0, 1, 0))
        # 亏损天数
        lose = np.sum(np.where(RET < 0, 1, 0))
        # 胜率
        win_ratio = win / total
        # 平均盈利率（天）
        mean_win_ratio = np.sum(np.where(RET > 0, RET, 0)) / win
        # 平均亏损率（天）
        mean_lose_ratio = np.sum(np.where(RET < 0, RET, 0)) / lose
        # 盈亏比(天)
        win_lose = win / lose

        # 盈利次数
        temp_df = df.copy()
        diff = temp_df[signal] != temp_df[signal].shift(1)
        temp_df['mark'] = diff.cumsum()
        # 每次开仓的收益率情况
        temp_df = temp_df.query(signal + '==1').groupby('mark')['ret'].sum()

        # 盈利次数
        win_count = np.sum(np.where(temp_df > 0, 1, 0))
        # 亏损次数
        lose_count = np.sum(np.where(temp_df < 0, 1, 0))
        # 单次最大盈利
        max_win = np.max(temp_df)
        # 单次最大亏损
        max_lose = np.min(temp_df)
        # 胜率
        win_rat = win_count / len(temp_df)
        # 平均盈利率（次）
        mean_win = np.sum(np.where(temp_df > 0, temp_df, 0)) / len(temp_df)
        # 平均亏损率（天）
        mean_lose = np.sum(np.where(temp_df < 0, temp_df, 0)) / len(temp_df)
        # 盈亏比(次)
        mean_wine_lose = win_count / lose_count

        summary_dic[signal] = [format_x(annual_ret), format_x(cum_ret_rate), sharpe_ratio, format_x(
            mdd), total, trade_count, mean_hold, win, lose, format_x(win_ratio), format_x(mean_win_ratio),
                               format_x(mean_lose_ratio), win_lose, win_count, lose_count, format_x(
                max_win), format_x(max_lose),
                               format_x(win_rat), format_x(mean_win), format_x(mean_lose), mean_wine_lose]

    summary_df = pd.DataFrame(summary_dic, index=index_name)
    summary_df.columns = col_name
    summary_df.to_excel(r'result\Strategy summary.xlsx')
    print(summary_df)


def main():
    df = pd.read_excel(r'data\HS300.xlsx', index_col=0)
    signal = signal_generation(df)
    new_signal = signal.dropna()
    ticker = '000300.SH'
    LongShortPlot(new_signal, ticker)
    summary(new_signal)


if __name__ == '__main__':
    main()
