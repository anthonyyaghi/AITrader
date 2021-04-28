import pandas as pd
import requests
from os import path
import glob
import backtrader as bt

from strategies import MACD

BASE_URL = "https://s3-eu-west-1.amazonaws.com/public.bitmex.com/data/trade/%s"

date_from = "2020-01-01"
date_to = "2021-01-01"


class BitmexComissionInfo(bt.CommissionInfo):
    params = (
        ("commission", 0.00075),
        ("mult", 1.0),
        ("margin", None),
        ("commtype", None),
        ("stocklike", False),
        ("percabs", False),
        ("interest", 0.0),
        ("interest_long", False),
        ("leverage", 1.0),
        ("automargin", False),
    )

    def getsize(self, price, cash):
        """Returns fractional size for cash operation @price"""
        return self.p.leverage * (cash / price)


def download_trade_file(filename, output_folder):
    print(f"Downloading {filename} file")
    url = BASE_URL % filename
    resp = requests.get(url)
    if resp.status_code != 200:
        print(f"Cannot download the {filename} file. Status code: {resp.status_code}")
        return
    with open(path.join(output_folder, filename), "wb") as f:
        f.write(resp.content)
    print(f"{filename} downloaded")


if __name__ == '__main__':
    dates = pd.date_range(date_from, date_to).astype(str).str.replace("-", "")
    output_folder = "/media/ubuntu/Transcend/AITraderData/Raw"
    # for date in dates:
    #     filename = date + ".csv.gz"
    #     download_trade_file(filename, output_folder)
    # exit()
    filepaths = glob.glob(path.join(output_folder, "*.csv.gz"))
    filepaths = sorted(filepaths)

    # df_list = []
    # checkpoint = 400
    # for i in range(len(filepaths)):
    #     if i <= checkpoint:
    #         continue
    #     print(f"Reading {filepaths[i]}")
    #     df_ = pd.read_csv(filepaths[i])
    #     df_ = df_[df_.symbol == "XBTUSD"]
    #     print(f"Read {df_.shape[0]} rows")
    #     df_list.append(df_)
    #     if i == checkpoint + 50:
    #         break
    # df = pd.concat(df_list)
    #
    # df.loc[:, "Datetime"] = pd.to_datetime(df.timestamp.str.replace("D", "T"))
    # df.drop("timestamp", axis=1, inplace=True)
    #
    # df = df.groupby(pd.Grouper(key="Datetime", freq="15Min")).agg(
    #     {"price": ["first", "max", "min", "last"], "foreignNotional": "sum"}
    # )
    #
    # df.columns = ["Open", "High", "Low", "Close", "Volume"]
    # df.loc[:, "OpenInterest"] = 0.0  # required by backtrader
    # df = df[df.Close.notnull()]
    # df.reset_index(inplace=True)
    # df.loc[:, "Datetime"] = df["Datetime"].astype(str).str.replace(" ", "T")

    dataset_filename = path.join(output_folder, f"XBT_USD_15min_2020-01-01_2021-01-01.csv")
    # df.to_csv(dataset_filename, mode='a', header=False, index=False)
    df = pd.read_csv(dataset_filename)
    # print(df)

    cerebro = bt.Cerebro()

    cerebro.broker.set_cash(1000)

    data = bt.feeds.GenericCSVData(
        dataname=dataset_filename,
        dtformat="%Y-%m-%dT%H:%M:%S",
        timeframe=bt.TimeFrame.Ticks,
    )

    cerebro.resampledata(data, timeframe=bt.TimeFrame.Minutes, compression=5)

    cerebro.addstrategy(MACD)

    cerebro.broker.addcommissioninfo(BitmexComissionInfo())

    # Add TimeReturn Analyzers to benchmark data
    cerebro.addanalyzer(
        bt.analyzers.TimeReturn, _name="alltime_roi", timeframe=bt.TimeFrame.NoTimeFrame
    )

    cerebro.addanalyzer(
        bt.analyzers.TimeReturn,
        data=data,
        _name="benchmark",
        timeframe=bt.TimeFrame.NoTimeFrame,
    )

    results = cerebro.run()
    st0 = results[0]

    for alyzer in st0.analyzers:
        alyzer.print()

    cerebro.plot(iplot=False, style="bar")
