import time
import requests
from threading import Thread

import cv2
import numpy as np
import pandas as pd
from mss import mss


class Indicator:
    def __init__(self, config, coin):
        self.last_signals = []
        self.detected_time_buy = None
        self.symbol = coin['kucoin_name']
        self.detection_wait_buy = config['buy_wait_time']
        self.detection_timeout_buy = config['buy_wait_timeout']
        self.detected_time_sell = None
        self.detection_wait_sell = config['sell_wait_time']
        self.detection_timeout_sell = config['sell_wait_timeout']
        self.bought = ''
        self.trade_amount = ''

    def check_for_action(self):
        if self.last_signals[-1] == 'buy' and not self.bought:
            if not self.detected_time_buy:
                print(f'{self.symbol} Detected buy!')
                self.detected_time_buy = time.time()

            else:
                if self.detection_wait_buy < (time.time() - self.detected_time_buy) < self.detection_timeout_buy:
                    print(
                        f'{self.symbol} Time from detection in {self.detection_wait_buy}-{self.detection_timeout_buy} seconds range, we can buy now!')
                    try:
                        self.buy()
                    except:
                        print(f'{self.symbol} Can Not Buy!')
                    else:
                        self.detected_time_sell = None
                        self.detected_time_buy = None

        elif self.last_signals[-1] == 'sell' and self.bought:
            if not self.detected_time_sell:
                print(f'{self.symbol} Detected sell!')
                self.detected_time_sell = time.time()

            else:
                if self.detection_wait_sell < (time.time() - self.detected_time_sell) < self.detection_timeout_sell:
                    print(
                        f'{self.symbol} Time from detection in {self.detection_wait_sell}-{self.detection_timeout_sell} seconds range, we can sell now!')
                    try:
                        self.sell()
                    except:
                        print(f'{self.symbol} Can Not Sell!')
                    else:
                        self.detected_time_sell = None
                        self.detected_time_buy = None

        if self.detected_time_buy:
            if time.time() - self.detected_time_buy > self.detection_timeout_buy:
                print(f'{self.symbol} {self.detection_timeout_buy} seconds passed since buy detected, reset time!')
                self.detected_time_buy = False

        if self.detected_time_sell:
            if time.time() - self.detected_time_sell > self.detection_timeout_sell:
                print(f'{self.symbol} {self.detection_timeout_sell} seconds passed since sell detected, reset time!')
                self.detected_time_sell = False

    def add_signal(self, side):
        self.last_signals.append(side)
        self.last_signals = self.last_signals[-120:]
        self.check_for_action()

    def buy(self):
        params = {'kucoin_name': self.symbol,
                  }

        r = requests.get("https://glowing-octo-train.herokuapp.com/buy_coin", params=params)
        # r = requests.get("http://127.0.0.1:5000/buy_coin", params=params)
        print(r.text)
        self.bought, self.trade_amount = r.text.split(',')

    def sell(self):
        params = {'kucoin_name': self.symbol,
                  'bought_id': self.bought,
                  'trade_amount': self.trade_amount,
                  }

        r = requests.get("https://glowing-octo-train.herokuapp.com/sell_coin", params=params)
        # r = requests.get("http://127.0.0.1:5000/sell_coin", params=params)
        if r.text == 'true':
            self.bought = ''
            self.trade_amount = ''


class Detection(Thread):
    def __init__(self, coin_name, bounding_box, config, sct):
        Thread.__init__(self)
        self.x0 = None
        self.y0 = None
        self.x_min = None
        self.x_max = None
        self.y_min = None
        self.y_max = None
        self.frame = None
        self.img = None
        self.sct = sct
        self.coin_name = coin_name
        self.bounding_box = bounding_box
        self.config = config

    def run(self):
        self.coin = pd.Series({'kucoin_name': self.coin_name})
        self.ind = Indicator(self.config, self.coin)

        self.ind.buy()
        time.sleep(1)
        self.ind.sell()
        # self.set_watch_box()
        # self.run_detection()

    def run_detection(self):
        wind_name = f"{self.coin_name} Detection Screen"
        cv2.namedWindow(wind_name, cv2.WINDOW_AUTOSIZE)

        self.last_frame = np.array([])
        self.frame_diffs = [30]

        while True:
            sct_img = self.sct.grab(self.bounding_box)
            self.frame = np.array(sct_img)
            if self.last_frame.any():
                frame_diff = np.average(np.abs(self.frame - self.last_frame))
                if frame_diff > 10 * np.average(self.frame_diffs) + 10:
                    print("!!!!!!!  Screen changed !!!!!!!")
                else:
                    self.frame_diffs.append(frame_diff)
                    self.frame_diffs = self.frame_diffs[-160:]

            self.last_frame = self.frame.copy()
            self.img = self.frame.copy()
            # Draw watch area rectangle
            self.draw_rect(self.img)

            # Get trade side from detected signals
            filt_buy, filt_sell = self.detect_on_region(self.img)
            signal_centroids = self.fill_centroid_dict(filt_buy, filt_sell)
            trade_side = self.last_signal(signal_centroids)
            self.ind.add_signal(trade_side)

            self.put_text(self.img, self.coin['kucoin_name'])
            cv2.imshow(wind_name, cv2.resize(self.img, [s // 2 for s in self.img.shape[:2]][::-1]))
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                cv2.destroyWindow(wind_name)
                break
            time.sleep(1)

    def get_region(self, event, x, y, f, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.x0:
                self.x_min, self.x_max = sorted([self.x0, x])
                self.y_min, self.y_max = sorted([self.y0, y])
                print('Region is set')
            else:
                self.x0, self.y0 = x, y

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.x0, self.y0 = None, None
            self.x_min, self.x_max, self.y_min, self.y_max = None, None, None, None
            print("Reset")

    def draw_rect(self, img):
        if self.x0:
            cv2.rectangle(img, (self.x_min, self.y_min), (self.x_max, self.y_max), (0, 255, 0), 2)
            cv2.circle(img, (self.x0, self.y0), 3, (0, 0, 255), thickness=2)

    def set_watch_box(self):
        wind_name = f"{self.coin_name} Set Region"
        cv2.namedWindow(wind_name, cv2.WINDOW_AUTOSIZE)
        while True:
            sct_img = self.sct.grab(self.bounding_box)
            self.frame = np.array(sct_img)
            self.img = self.frame.copy()

            cv2.setMouseCallback(wind_name, self.get_region)
            self.draw_rect(self.img)
            cv2.imshow(wind_name, self.img)

            if (cv2.waitKey(1) & 0xFF) == ord('n') and self.x_min:
                cv2.destroyWindow(wind_name)
                break

    def detect_on_region(self, im, replace=True):
        watch_img = im[self.y_min:self.y_max, self.x_min:self.x_max, :]

        kernel = np.ones((5, 5), np.uint8)
        filt_buy = self.filter_color(watch_img, side='buy')
        filt_buy = cv2.morphologyEx(filt_buy, cv2.MORPH_CLOSE, kernel, iterations=5)
        filt_sell = self.filter_color(watch_img, side='sell')
        filt_sell = cv2.morphologyEx(filt_sell, cv2.MORPH_CLOSE, kernel, iterations=5)

        zeros = np.zeros_like(filt_buy)
        buy_img = np.stack((zeros, filt_buy, zeros, zeros + 128), axis=-1)
        sell_img = np.stack((zeros, zeros, filt_sell, zeros + 127), axis=-1)

        watch_img_filt = buy_img + sell_img
        if replace:
            im[self.y_min:self.y_max, self.x_min:self.x_max, :] = watch_img_filt
        return filt_buy, filt_sell

    @staticmethod
    def filter_color(im, side='buy'):
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        if side == 'buy':
            lower = np.array([140, 200, 150], np.uint8)
            upper = np.array([145, 255, 255], np.uint8)
        elif side == 'sell':
            lower = np.array([20, 200, 150], np.uint8)
            upper = np.array([30, 255, 255], np.uint8)

        mask = cv2.inRange(hsv, lower, upper)
        return mask

    @staticmethod
    def last_signal(centroid_dict):
        """Return last (most right) signal side (buy/sell) on the image"""

        def key(items):
            k, v = items
            m_val = max(list(zip(*v))[0]) if v else 0
            return m_val

        sort_dict = sorted(centroid_dict.items(), key=key, reverse=True)

        if sort_dict[0][1]:  # if centroid list is not empty
            return sort_dict[0][0]

        return None

    @staticmethod
    def fill_centroid_dict(img_buy, img_sell):
        """Fill dict with buy and sell signal Centroids for sorting"""
        buy_contours, _ = cv2.findContours(img_buy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        sell_contours, _ = cv2.findContours(img_sell, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        centroid_dict = {'buy': [], 'sell': []}
        for side, contours in zip(('buy', 'sell'), (buy_contours, sell_contours)):
            for cont in contours:
                x_0, y_0 = list(np.min(cont, axis=0)[0])
                x_1, y_1 = list(np.max(cont, axis=0)[0])
                centroid = (x_0 + x_1) // 2, (y_0 + y_1) // 2
                centroid_dict[side].append(centroid)
        return centroid_dict

    @staticmethod
    def put_text(im, text):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(im, text, (20, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)


def split_screen(coins, rows, monitor):
    l = len(coins)
    cols = l // rows + 1 * (l % rows > 0)
    W, H = monitor['width'], monitor['height']

    w, h = W // cols, H // rows
    bounding_boxes = dict()

    for row in range(rows):
        for col in range(cols):
            i = row * cols + col
            if i >= len(coins):
                break
            b_box = {'top': monitor['top'] + h * row, 'left': monitor['left'] + w * col, 'width': w, 'height': h}
            bounding_boxes[coins[i]] = b_box

    return bounding_boxes


def main():
    config = {"buy_wait_time": 30,
              "buy_wait_timeout": 60,
              "sell_wait_time": 30,
              "sell_wait_timeout": 60,
              "monitor": 1,
              "rows": 2,
              "coins": [
                  # 'SAND-USDT', 'ENJ-USDT', "GALAX-USDT",
                  #       "FTM-USDT", 'MATIC-USDT', "MANA-USDT",
                        'UOS-USDT',  "VRA-USDT",
                        ]
              }

    sct = mss()
    monitor = sct.monitors[config['monitor']]
    bounding_boxes = split_screen(config['coins'], rows=config['rows'], monitor=monitor)
    for coin, b_box in bounding_boxes.items():
        print(coin, b_box)
        det = Detection(coin, b_box, config, sct)
        det.start()
        time.sleep(5)
        # break


if __name__ == "__main__":
    main()
