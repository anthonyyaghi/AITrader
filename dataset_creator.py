from os import path
import pandas as pd
from PIL import Image, ImageDraw
import numpy as np
from tqdm import tqdm

IMG_SIZE = 32
CANDLES_PER_IMG = 32
CANDLE_WIDTH = IMG_SIZE / CANDLES_PER_IMG

data_folder = "/media/ubuntu/Transcend/AITraderData/Raw"
output_folder = "/media/ubuntu/Transcend/AITraderData/TrainSet_15min"


def scale(x, domain, out_range=(-1, 1)):
    # domain = (min, max)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2


if __name__ == '__main__':
    data_filename = path.join(data_folder, "XBT_USD_15min_2020-01-01_2021-01-01.csv")
    df = pd.read_csv(data_filename)
    rows_data_train = []
    rows_data_test = []
    done_train = False

    for i in tqdm(range(CANDLES_PER_IMG - 1, df.shape[0] - 1)):
        if i > df.shape[0]*0.75 and not done_train:
            pd.DataFrame(data=rows_data_train, columns=['image', 'action']). \
                to_csv(path.join(output_folder, 'dataset.csv'), index=False)
            output_folder = output_folder + "_test"
            done_train = True

        opens = np.zeros(CANDLES_PER_IMG)
        closes = np.zeros(CANDLES_PER_IMG)
        highs = np.zeros(CANDLES_PER_IMG)
        lows = np.zeros(CANDLES_PER_IMG)
        for j in range(CANDLES_PER_IMG):
            opens[j] = df.loc[i - (CANDLES_PER_IMG - 1) + j]['Open']
            closes[j] = df.loc[i - (CANDLES_PER_IMG - 1) + j]['Close']
            highs[j] = df.loc[i - (CANDLES_PER_IMG - 1) + j]['High']
            lows[j] = df.loc[i - (CANDLES_PER_IMG - 1) + j]['Low']
        dom = (np.min(lows), np.max(highs))

        opens = scale(opens, dom, (0, IMG_SIZE))
        closes = scale(closes, dom, (0, IMG_SIZE))
        highs = scale(highs, dom, (0, IMG_SIZE))
        lows = scale(lows, dom, (0, IMG_SIZE))

        img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), color='white')
        d = ImageDraw.Draw(img)
        # Draw candle wick and body
        for j in range(CANDLES_PER_IMG):
            wick_x1 = j * CANDLE_WIDTH
            wick_y1 = IMG_SIZE - highs[j]
            wick_x2 = wick_x1 + CANDLE_WIDTH - 1
            wick_y2 = IMG_SIZE - lows[j]
            body_x1 = j * CANDLE_WIDTH
            body_y1 = IMG_SIZE - (max(opens[j], closes[j]) - 1)
            body_x2 = body_x1 + CANDLE_WIDTH - 1
            body_y2 = IMG_SIZE - (min(opens[j], closes[j]) - 1)
            if opens[j] < closes[j]:
                fill = (0, 255, 0)
            else:
                fill = (255, 0, 0)
            d.rectangle((wick_x1, wick_y1, wick_x2, wick_y2), fill='black')
            d.rectangle((body_x1, body_y1, body_x2, body_y2), fill=fill)
        img_name = f'frame_{i - (CANDLES_PER_IMG - 1)}.png'
        img_path = path.join(output_folder, img_name)
        img.save(img_path)

        change = (df.loc[i + 1]['Open'] - df.loc[i + 1]['Close']) / df.loc[i + 1]['Open']
        if abs(change * 100) < 0.0035:
            # neutral
            action = 0
        elif change > 0:
            # buy
            action = 1
        else:
            # sell
            action = 2
        if i > df.shape[0] * 0.75:
            rows_data_test.append([img_name, action])
        else:
            rows_data_train.append([img_name, action])

    pd.DataFrame(data=rows_data_test, columns=['image', 'action']).\
        to_csv(path.join(output_folder, 'dataset.csv'), index=False)
