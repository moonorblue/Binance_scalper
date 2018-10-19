#!/usr/bin/env python
import time
import sys
import getopt
from binance.client import Client
import os
import config
import datetime
import math

client = None


def get_ticker(symbol):
    try:
        ret = client.get_ticker(symbol=symbol)
        return float(ret["lastPrice"])
    except Exception as e:
        # printDateTime()
        print(e)


def get_order_book(symbol, limit, return_type):
    try:
        ret = client.get_order_book(symbol=symbol, limit=5)

        if return_type == 'float':
            lastBid = float(ret['bids'][0][0])
            lastAsk = float(ret['asks'][0][0])
            return lastBid, lastAsk
        else:
            return ret['bids'][0][0], ret['asks'][0][0]

    except Exception as e:
        # printDateTime()
        print(e)
        return None, None


def order_market_buy(symbol, quantity):
    try:
        client.order_market_buy(symbol=symbol, quantity=quantity)
        return True
    except Exception as e:
        # printDateTime()
        print(e)
        return False


def _chop(v, n):
    s = '%.10lf' % v
    return s[:s.find('.') + n + 1]


def checkBNB():

    try:
        ret = client.get_account()['balances']
    except Exception as e:
        print(e)
        return -1

    symbolFund = -1
    for asset in ret:
        if asset['asset'] == "BNB":
            symbolFund = float(asset['locked']) + float(asset['free'])
            break

    print "current BNB: %s" % str(symbolFund)
    if symbolFund <= 0.05:
        print "buy BNB"
        lastPrice = get_ticker("BNBBTC")
        # lastBid, lastAsk = get_order_book("BNBBTC", 5, float)
        qty = round(0.002 / float(lastPrice), 2)
        qty += 0.1
        q = _chop(qty, 3)
        order_market_buy("BNBBTC", q)


def main():
    global client
    client = Client(config.api_key,
                    config.api_secret)
    while True:
        checkBNB()
        time.sleep(5)


if __name__ == "__main__":

    main()
