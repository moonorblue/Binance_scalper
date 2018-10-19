#!/usr/bin/env python
import time
import sys
import getopt
from binance.client import Client
import os
import config
from threading import Thread
import datetime
import threading
import math
import collections
import numpy
import requests
import json
from collections import namedtuple


PRICE_MODE = ""
DEBUG_MODE = True
TRADE_MODE = "Normal"
TOGGLE_DYNAMIC = True
symbol = ""
PROFIT = 0

QUANTITY = 0
INCREASING = 0.00000001

WAIT_TIME = 1
client = None
MAXRETRY = 20

MaxBuyRetry = 5
MaxSellRetry = 5

WAIT_SELL_TIME = 1
WAIT_BUY_TIME = 1
WAIT_BUY_FILLED_TIME = 0.5
WAIT_TIME_OFFSET = 0.25


PARTIAL_FILLED_TIME = 1.5

DIGIT = 8

ConcurrentMax = 1
currentAction = 0

FailActionMax = 3
failActions = 0

ReOrderPriceMax = 3


currentMarket = None

totalProfit = 0.0


startTime = 0
# DataModel


totalTran = 0
winTran = 0
loseTran = 0


MIN_NOTIONAL = 0
LOT_SIZE = 0
stepSize = 0
HighPriority_SELL = 3
HighPriority_BUY = 1


original_fund = 0
previous_fund = 0

stopLoss = -6.0
baseCryptoName = ""

lock = threading.Lock()

lastPriceQueue = collections.deque(maxlen=5)
previousLastPrice = 0

qtyOffset = 1

featureStr = ""

tradeFlag = True

allMarket = set()

totalLossTime = 0


previousTickers = None
deQueue = collections.deque(maxlen=10)

message = ''


priceQueue = collections.deque(maxlen=10)
priceAllChangeDict = {}

volumeQueue = collections.deque(maxlen=10)
volumeAllChangeDict = {}

volumeDict = {}

v_COUNT_THRESHOLD = 4
p_COUNT_THRESHOLD = 4

currentMarketIncreaseCount = 0

tradedSymbols = set()

ISAUTOMARKET = False

MaxProfit = 0
ISNEWLISTED = False


QuantityPosition = 55
INITQuantityPosition = 55

ma7 = 0
ma25 = 0
ma7Queue = collections.deque(maxlen=2)
ma25Queue = collections.deque(maxlen=2)
klines = []

ISUSEMA = False

STOPLOSSBY = "ASSET"

NewListTradeCount = 0


# Classes


class Crypto:

    def __init__(self, symbol, balance, minNotioal, sSize):
        self.symbol = symbol
        self.balance = balance
        self.minNotioal = minNotioal
        self.sSize = sSize

    def getBalance(self, client):
        try:
            ret = client.get_account()['balances']
            bal = -1
            for asset in ret:
                if asset['asset'] == self.symbol:
                    bal = float(asset['locked']) + float(asset['free'])

            self.balance = str(bal)
            return self.balance

        except Exception as e:
            print(e)
            # sendPush(str(e))
            return


class Tran:

    def __init__(self, symbol, price, buyOrderId, sellOrderId, isSold, isBought, totalQuantity, partQuantity, partPrice, currentQty, isFinished):
        self.symbol = symbol
        self.price = price
        self.buyOrderId = buyOrderId
        self.sellOrderId = sellOrderId
        self.isSold = isSold
        self.isBought = isBought
        self.totalQuantity = totalQuantity
        self.partQuantity = partQuantity
        self.partPrice = partPrice
        self.currentQty = currentQty
        self.isFinished = isFinished


class Price:

    def __init__(self, buyPrice, sellPrice):
        self.buyPrice = buyPrice
        self.sellPrice = sellPrice


class Market:
    Long = 0
    Short = 1


# MA
def getMA(kline, k):
    sumP = 0
    lenK = k
    if len(kline) > k:
        for candle in kline[-k:]:
            closeP = float(candle[4])
            sumP += closeP
        if sumP != 0:
            return float(sumP) / lenK
        else:
            return -1
    else:
        return -1
# Detect proper market methods


def getTopPriceChange(tickers):
    report = {}
    for symbol, ticker in tickers.items():
        report[symbol] = float(ticker.priceChangePercent)

    return report


def getTicker():
    global priceAllChangeDict
    global volumeAllChangeDict
    try:
        tickers = client.get_ticker()
        tickerDict = {}
        for i in tickers:
            obj = namedtuple("ticker", i.keys())(*i.values())
            # add current base crypto and USDT market
            if baseCryptoName in obj.symbol and "USDT" not in obj.symbol:
                tickerDict[obj.symbol] = obj
                # add empty list to save changes
                if obj.symbol not in priceAllChangeDict:
                    priceAllChangeDict[
                        obj.symbol] = collections.deque(maxlen=10)

                if obj.symbol not in volumeAllChangeDict:
                    volumeAllChangeDict[
                        obj.symbol] = collections.deque(maxlen=10)

        return tickerDict
    except Exception as e:
        print(e)
        return None


def calChangeRate(tickers, change_type):
    # global allChangeDict
    report = {}
    if previousTickers != None and tickers != None:
        for symbol, ticker in tickers.items():
            if symbol in previousTickers:
                if change_type == "price":
                    currentTickerChange = float(ticker.priceChangePercent)
                    previousTickerChange = float(
                        previousTickers[symbol].priceChangePercent)
                    change = (currentTickerChange - previousTickerChange)
                    report[symbol] = change
                    # allChangeDict[symbol].append(change)
                elif change_type == "volume":
                    currentTickerVolume = float(ticker.quoteVolume)
                    previousTickerVolume = float(
                        previousTickers[symbol].quoteVolume)
                    change = (currentTickerVolume - previousTickerVolume)
                    report[symbol] = change

        return report

    return None


def calCount(queue):
    allList = []
    for symbols in queue:
        allList += symbols
    countDict = collections.Counter(allList)
    return countDict


def detectStopLoss():
    global tradeFlag

    tradeFlag = False  # stop trading
    waitCount = 0

    while currentAction != 0:
        # still in trade, wait until finished
        printMsg("Stop loss Wait for current trade")
        time.sleep(10)
        if waitCount >= 6:
            printMsg("Force stop current trade")
            # force sell current symbol trades
            cancelAllCurrentOrder()
            break
        waitCount += 1
    # reset action

    time.sleep(30)

    sellRemainAltCryptoIfNeeded()
    cancelAllCurrentOrder()

    print "Stop loss, exit"
    os._exit(0)


def sortTickersByVolume(tickers, k):
    top = int(len(tickers) * k)
    volDict = {}
    for t in tickers:
        volDict[t] = float(tickers[t].quoteVolume)

    sortedList = sorted(volDict.items(), lambda x, y: cmp(
        float(x[1]), float(y[1])), reverse=True)[:top]
    soretedSet = set()
    for i in sortedList:
        soretedSet.add(i[0])

    return soretedSet


def detectProperMarket():

    global previousTickers
    global client
    global priceQueue
    global volumeQueue
    global priceAllChangeDict
    global volumeAllChangeDict
    global message
    global failActions
    global tradeFlag
    global QuantityPosition

    print "Start monitoring proper market"
    while True:
        try:
            tickers = getTicker()
            priceReport = calChangeRate(tickers, "price")
            volumeReport = calChangeRate(tickers, "volume")
            # report = getTopPriceChange(tickers)
            # sort tickers by volume
            topVolume = sortTickersByVolume(tickers, 0.35)

            previousTickers = tickers
            # print report
            if priceReport != None:

                dateTimeMsg = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print dateTimeMsg + " tradeFlag: %s, currentAction: %d " % (tradeFlag, currentAction)

                # message = "%s<br>" % (dateTimeMsg)

                priceSortedList = sorted(priceReport.items(), lambda x,
                                         y: cmp(float(x[1]), float(y[1])), reverse=True)

                symbols = []
                for idx, val in enumerate(priceSortedList[:10]):
                    symbols.append(val[0])
                    priceAllChangeDict[val[0]].append(val[1])

                priceQueue.append(symbols)
                priceCountedDict = calCount(priceQueue)

                volumeSortedList = sorted(volumeReport.items(), lambda x,
                                          y: cmp(float(x[1]), float(y[1])), reverse=True)

                symbols = []
                for idx, val in enumerate(volumeSortedList[:10]):
                    symbols.append(val[0])
                    volumeAllChangeDict[val[0]].append(val[1])

                volumeQueue.append(symbols)
                volumeCountedDict = calCount(volumeQueue)

                currentBestSymbol = ""
                currentBestCount = 0

                for v_idx, v_val in enumerate(volumeSortedList[:10]):
                    v_symbol = v_val[0]
                    v_change = v_val[1]
                    v_count = volumeCountedDict[v_symbol]
                    # get sum from allChangeDict
                    v_changes = volumeAllChangeDict[v_symbol]
                    # print changes
                    v_sumOfChanges = 0.0

                    if len(v_changes) > 1:
                        start = len(v_changes) - 1
                        end = len(v_changes) - v_count - 1
                        for changeIdx in xrange(start, end, -1):
                            v_sumOfChanges += v_changes[changeIdx]
                    else:
                        start = 0
                        v_sumOfChanges += v_changes[0]

                    topStr = "[Volume][Top{:>2}]".format(v_idx + 1)
                    symbolStr = "{:<10}".format(v_symbol)
                    lastIncreasing = ":{:<6.2f}".format(v_change)
                    countStr = "({:<1})".format(v_count)
                    totalIncreasing = "Total:{:<6.2f} ".format(v_sumOfChanges)

                    string = "{:}{:}{:}{:} {:} ".format(
                        topStr, symbolStr, lastIncreasing, countStr, totalIncreasing)

                    # check count

                    if v_count >= v_COUNT_THRESHOLD:
                        for p_idx, p_val in enumerate(priceSortedList[:10]):
                            priceSymbol = p_val[0]
                            priceCount = priceCountedDict[priceSymbol]
                            if priceSymbol == v_symbol and priceCount >= p_COUNT_THRESHOLD and float(v_change) > 0 and priceSymbol in topVolume:
                                # push = Push(rest_key, app_id)
                                # push.sendPush(string[:-4])
                                p_symbol = p_val[0]
                                p_change = p_val[1]
                                p_count = priceCountedDict[p_symbol]
                                # get sum from allChangeDict
                                p_changes = priceAllChangeDict[p_symbol]
                                # print changes
                                p_sumOfChanges = 0.0

                                if len(p_changes) > 1:
                                    start = len(p_changes) - 1
                                    end = len(p_changes) - p_count - 1
                                    for changeIdx in xrange(start, end, -1):

                                        p_sumOfChanges += p_changes[changeIdx]
                                else:
                                    start = 0
                                    p_sumOfChanges += p_changes[0]

                                p_topStr = "[Price ][Top{:>2}]".format(
                                    p_idx + 1)
                                p_symbolStr = "{:<10}".format(p_symbol)
                                p_lastIncreasing = ":{:<6.2f}".format(p_change)
                                p_countStr = "({:<1})".format(p_count)
                                p_totalIncreasing = "Total:{:<6.2f} ".format(
                                    p_sumOfChanges)

                                p_string = "{:}{:}{:}{:} {:} ".format(
                                    p_topStr, p_symbolStr, p_lastIncreasing, p_countStr, p_totalIncreasing)

                                totalCount = p_sumOfChanges + v_sumOfChanges

                                print p_string
                                print string
                                print "Detected market: %s" % priceSymbol
                                if currentBestSymbol == "" and currentBestCount == 0:
                                    currentBestSymbol = priceSymbol
                                    currentBestCount = totalCount
                                else:
                                    if totalCount > currentBestCount:
                                        currentBestSymbol = priceSymbol
                                        currentBestCount = totalCount

                if currentBestSymbol != "":
                    failActions = 0
                    if ISAUTOMARKET == True:
                        if currentBestSymbol != symbol:
                            print "Change to market: %s" % currentBestSymbol
                            changeTradeSymbol(currentBestSymbol, "Normal")

                        else:
                            print "Remain same market: %s" % currentBestSymbol
                            if tradeFlag == False:
                                tradeFlag = True
                            QuantityPosition = INITQuantityPosition
                        # get maybe profitable market, change market
                    else:
                        if currentBestSymbol == symbol:
                            print "Remain same market: %s" % currentBestSymbol
                            if tradeFlag == False:
                                tradeFlag = True

            # message here is top 10 volume
        except Exception as e:
            print(e)

        time.sleep(25)


# General Methods


def _chop(v, n):
    s = '%.10lf' % v
    return s[:s.find('.') + n + 1]


def getSellPriceAndBuyPrice(lastAsk, lastBid, lastPrice):

    buyPrice = 0
    sellPrice = 0
    if PRICE_MODE == "increasing":
        sellPrice = lastAsk - float(INCREASING)
        buyPrice = lastBid + float(INCREASING)

    elif PRICE_MODE == "average":
        distance = lastAsk - lastBid
        sellPrice = lastAsk - distance * 0.47
        buyPrice = lastBid + distance * 0.47

    price = Price("%.*f" % (DIGIT, buyPrice), "%.*f" % (DIGIT, sellPrice))

    return price


def printDateTime():
    dateTime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if DEBUG_MODE:
        print dateTime


def tranFinalMsg(tran):

    global totalProfit
    global startTime
    global totalTran
    global winTran
    global loseTran
    global currentAction

    global failActions
    global currentMarket
    global original_fund
    global previous_fund

    global featureStr

    global tradeFlag
    global totalLossTime
    global MaxProfit

    if tran.isFinished == False:
        # lock.acquire(False)
        tran.isFinished = True
        # lock.release()
        totalTran += 1

        profit = 0.0

        btc = Crypto(baseCryptoName, 0, 0, 0)
        btcBalance = btc.getBalance(client)

        bnb = Crypto("BNB", 0, 0, 0)
        bnbBalance = bnb.getBalance(client)
        bnbPrice = get_ticker("BNB" + baseCryptoName)
        bnbFund = float(bnbBalance) * float(bnbPrice)

        sym = Crypto(symbol[:-3], 0, 0, 0)
        symBalance = sym.getBalance(client)
        symPrice = get_ticker(symbol)
        symFund = float(symBalance) * float(symPrice)

        currentBalance = float(btcBalance) + float(symFund) + float(bnbFund)

        profit = currentBalance - previous_fund
        previous_fund = currentBalance

        totalProfit += profit

        msg = ""

        if profit > 0:
            msg = "{:<6}{:>12}".format("Profit", "%.8f" % profit)
            winTran += 1
            if failActions > 0:
                failActions -= 1
            if featureStr != "":
                featureStr += "%.8f,1" % profit
        else:
            msg = "{:<6}{:>12}".format("Loss", "%.8f" % profit)
            loseTran += 1
            failActions += 1
            if featureStr != "":
                featureStr += "%.8f,0" % profit

        if failActions >= FailActionMax:
            if tradeFlag == True:
                tradeFlag = False
                print "Too many fails, stop until market is better."

        # writetoFile
        if featureStr != "":
            # print featureStr
            with open('%s-[FEATURE]-[MA].txt' % symbol, 'a') as logfile:
                logfile.write(featureStr + '\n')

        dateTime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        currentTime = time.time()
        elapsedSecs = currentTime - startTime
        elapsedHours = float(elapsedSecs) / 3600

        estimateHourProfit = float(totalProfit) / elapsedHours
        hourProfitMsg = "Est. {:<9} {:}/hour".format(
            "%.8f" % estimateHourProfit, baseCryptoName)
        totalProfitMsg = "TotalProfit:{:>14}".format("%.8f" % totalProfit)
        totalTime = "TotalHours: {:.4f}".format(elapsedHours)

        winRate = "Win Rate: {:.2f}%".format(
            (float(winTran) / totalTran) * 100)

        distance = float(currentBalance) - float(original_fund)

        per = (distance / float(original_fund)) * 100

        balance_msg = "Original %s: %s, Current %s: %s (%.2f)" % (
            baseCryptoName, original_fund, baseCryptoName, currentBalance, per)

        if distance > 0:
            original_fund = currentBalance

        if STOPLOSSBY == "ASSET":

            if per < stopLoss:
                if totalLossTime >= 1:
                    print balance_msg
                    # print "Too many loss. Exit"
                    sendPush(balance_msg + "Too many loss. Exit")
                    detectStopLoss()
                else:
                    print balance_msg
                    print "Too many loss. HOLD, retry again"
                    sendPush(balance_msg + "Too many loss. HOLD, retry again")
                    tradeFlag = False
                    totalLossTime += 1

                    time.sleep(10)
                    print "Try to sell current alt crypto"
                    sellRemainAltCryptoIfNeeded()
                    print "Try to cancel current order"
                    cancelAllCurrentOrder()

                    # lock.acquire(False)
                    tradeFlag = True
                    currentAction = 0
                    # lock.release()

                # os._exit(0)
                # sys.exit(1)
            else:
                # lock.acquire(False)
                totalLossTime = 0
                currentAction = 0
                # lock.release()

        elif STOPLOSSBY == "PROFIT":

            if MaxProfit == 0:
                MaxProfit = totalProfit

            profit_dis = totalProfit - MaxProfit
            profit_dis_per = float(profit_dis) / MaxProfit * 100

            profit_msg = "MaxProfit %.8f, TotalProfit %.8f,(%.2f)" % (
                MaxProfit, totalProfit, profit_dis_per)
            print profit_msg

            if totalProfit > MaxProfit:
                MaxProfit = totalProfit

            # if totalProfit < 0:
            #     detectStopLoss()

            if profit_dis_per < stopLoss:
                if totalLossTime >= 1:
                    detectStopLoss()
                else:
                    print "Too many loss. HOLD, retry again"
                    tradeFlag = False
                    totalLossTime += 1

                    time.sleep(5)
                    print "Try to sell current alt crypto"
                    sellRemainAltCryptoIfNeeded()
                    print "Try to cancel current order"
                    cancelAllCurrentOrder()

                    # lock.acquire(False)
                    tradeFlag = True
                    currentAction = 0
                    # lock.release()

            elif totalProfit < 0:
                if totalLossTime >= 1:
                    detectStopLoss()
                else:
                    print "Too many loss. HOLD, retry again"
                    tradeFlag = False
                    totalLossTime += 1

                    time.sleep(5)
                    print "Try to sell current alt crypto"
                    sellRemainAltCryptoIfNeeded()
                    print "Try to cancel current order"
                    cancelAllCurrentOrder()

                    # lock.acquire(False)
                    tradeFlag = True
                    currentAction = 0
                    # lock.release()
            else:
                # lock.acquire(False)
                totalLossTime = 0
                currentAction = 0
                # lock.release()

        return "[%s][%s]  %s,  %s,  %s,  %s,  %s \n\r%s" % (dateTime, tran.symbol, msg, hourProfitMsg, totalProfitMsg, totalTime, winRate, balance_msg)

    return ""


def printMsg(msg):
    if DEBUG_MODE:
        print msg


def checkIfReachMinNot(price, quantity):
    total = float(price) * float(quantity)
    if total >= MIN_NOTIONAL:
        return True
    else:
        return False


def getMinNotQuantity(price):
    quantity = MIN_NOTIONAL / float(price)
    return quantity


# checkBNB

def checkBNB():

    try:
        ret = client.get_account()['balances']
        symbolFund = -1
        for asset in ret:
            if asset['asset'] == "BNB":
                symbolFund = float(asset['locked']) + float(asset['free'])

        if symbolFund <= 0.05 and symbol != -1:
            lastPrice = get_ticker("BNB" + baseCryptoName)

            qty = round(0.002 / float(lastPrice), 2)
            qty += 0.1
            q = _chop(qty, 3)
            printMsg("current BNB: %s, Buy BNB: %s" %
                     (str(symbolFund), str(q)))
            order_market_buy("BNB" + baseCryptoName, q)

    except Exception as e:
        print(e)
        # sendPush(str(e))


# APIs


def getFilter(symbol):
    try:
        symbols = client.get_exchange_info()['symbols']
        for s in symbols:
            if s['symbol'] == symbol:
                filters = s['filters']
                return filters
    except Exception as e:
        printDateTime()
        print "getFilter error"
        print(e)
        # sendPush(str(e))


def getMinQty(symbol):
    try:
        filters = getFilter(symbol)
        for f in filters:
            if f['filterType'] == "LOT_SIZE":
                return f['minQty']
    except Exception as e:
        printDateTime()
        print "getMinQty error, use default"
        return 1


def getStepSize(symbol):
    try:
        filters = getFilter(symbol)
        for f in filters:
            if f['filterType'] == "LOT_SIZE":
                return f['stepSize']
    except Exception as e:
        printDateTime()
        print "getStepSize error, use default"
        return 1


def getTickSize(symbol):
    try:
        filters = getFilter(symbol)
        for f in filters:
            if f['filterType'] == "PRICE_FILTER":
                return f['tickSize']
    except Exception as e:
        printDateTime()
        print "getTickSize error, use default"
        return 0.00000001


def getAvailableQty(qty):
    logStep = int(abs(math.log10(stepSize)))
    roundQty = 0

    if stepSize == 1.0:
        roundQty = math.floor(float(qty))
    else:
        roundQty = math.floor(
            float(qty) * math.pow(10, logStep)) / math.pow(10, logStep)
    # printMsg("available qty:%s" % _chop(float(roundQty), 3))
    return _chop(float(roundQty), 3)


def getMinNotional(symbol):
    try:
        filters = getFilter(symbol)
        for f in filters:
            if f['filterType'] == "MIN_NOTIONAL":
                return f['minNotional']

    except Exception as e:
        printDateTime()
        print "getMinNotional error, use default"
        return 0.00000001


def order_limit_sell(symbol, quantity, price, tran):
    global currentAction
    sell_quantity = quantity
    retry = 0

    while retry <= 10:
        # printMsg("[LimitSell-Try]: %s x %s" % (price, quantity))
        try:
            response = client.order_limit_sell(
                symbol=symbol, quantity=sell_quantity, price=price)
            tran.sellOrderId = response['orderId']
            printMsg("[LimitSell]: %s x %s" % (price, sell_quantity))
            return True

        except Exception as e:
            # check buy status
            printMsg("[LimitSell-Fail]: %s x %s" % (price, sell_quantity))
            printDateTime()
            print(e)

            retry += 1
            time.sleep(1)
            # get current fund and sell again
            needToSellFund = checkSellFundIfNeed(symbol, 0)
            if needToSellFund != -1:
                qty = getAvailableQty(needToSellFund)
                sell_quantity = qty
                printMsg("Resell again")

            continue

    tran.isFinished = True
    setCurrentActionToZero()

    return False


def order_limit_buy(symbol, quantity, price, tran):
    global currentAction
    retry = 0

    # printMsg("[LimitBuy-Try]: %s x %s" % (price, quantity))
    try:
        response = client.order_limit_buy(
            symbol=symbol, quantity=quantity, price=price)
        tran.buyOrderId = response['orderId']
        # printMsg("[LimitBuy]: %s x %s" % (price, quantity))
        return True

    except Exception as e:
        printDateTime()
        printMsg("[LimitBuy-Fail]: %s x %s" % (price, quantity))

        printMsg("[LimitBuy-ReTry]: Sell at MarketPrice x %s" % quantity)
        sellCurrentFundIfNeeded()
        time.sleep(1)
        print(e)
        # sendPush(str(e))
        return False


def getPriceInHighPriority(symbol, price, side):

    try:
        ret = client.get_order_book(symbol=symbol, limit=5)
        prices = []
        qty = []

        if side == "ask":
            for i in range(HighPriority_SELL):
                prices.append(ret['asks'][i][0])
                qty.append(ret['asks'][i][1])
        else:
            for i in range(HighPriority_BUY):
                prices.append(ret['bids'][i][0])
                qty.append(ret['bids'][i][1])

        for idx, val in enumerate(prices):
            if float(val) == float(price):
                return idx, qty[idx]

        return -1, -1

    except Exception as e:
        printDateTime()
        print "get priority error"
        time.sleep(0.5)
        print(e)
        # sendPush(str(e))
        return None, None


def getCurrentFund(symbol):
    # printMsg("getCurrentFund: %s" % symbol)
    flag = True
    retry = 0
    symbolFund = -1

    while flag == True and retry < 5:
        try:
            ret = client.get_account()['balances']
        except Exception as e:
            print(e)
            # sendPush(str(e))
            return -1

        for asset in ret:
            if asset['asset'] == symbol.replace(baseCryptoName, ""):
                symbolFund = float(asset['free'])

        if symbolFund != -1 and symbolFund != 0:
            printMsg("symbol fund %f" % symbolFund)
            break
        time.sleep(0.5)
        retry += 1
        printMsg("symbol fund error, retry (%d)" % retry)

    return symbolFund


def checkSellFundIfNeed(symbol, qty):
    # printMsg("checkSellFundIfNeed:%s" % symbol)
    currentFund = getCurrentFund(symbol)

    if currentFund > float(qty):
        needToSell = currentFund - qty
        return needToSell
    else:
        return -1


def order_market_sell(symbol, quantity):
    printMsg("order_market_sell: %s" % symbol)
    global currentAction
    try:
        client.order_market_sell(symbol=symbol, quantity=quantity)
        return True
    except Exception as e:
        # print "sell market"
        printDateTime()
        print "market sell error"
        print(e)
        # sendPush(str(e))
        time.sleep(WAIT_SELL_TIME)
        return False


def order_market_buy(symbol, quantity):
    try:
        client.order_market_buy(symbol=symbol, quantity=quantity)
        return True
    except Exception as e:
        printDateTime()
        print "market buy error"
        print(e)
        # sendPush(str(e))
        return False


def get_order(symbol, orderId):
    global currentAction

    MAX = 5
    count = 0
    errorCount = 0

    while count < MAX:
        try:
            response = client.get_order(symbol=symbol, orderId=orderId)
            if errorCount > 0:
                errorCount -= 1
            return response
        except Exception as e:
            printDateTime()
            print(e)
            print "Get order error"
            time.sleep(2)
            if errorCount > 30:
                cancelAllCurrentOrder()
                sellRemainAltCryptoIfNeeded()
                print "Too many error, leave"
                os._exit(0)
            # sendPush(str(e))
            errorCount += 1
            count += 1
            continue

    # cancelAllCurrentOrder()
    time.sleep(30)
    sellCurrentFundIfNeeded()

    return None


def cancel_order(symbol, orderId):
    global currentAction
    # printMsg("Try to cancel order: %s" % (orderId))
    if orderId is not None:
        try:
            client.cancel_order(symbol=symbol, orderId=orderId)
            return True
        except Exception as e:
            printDateTime()
            print "cancel order error"
            print(e)

            # setCurrentActionToZero()
            # sendPush(str(e))
            return False


def get_ticker(symbol):
    global tradeFlag
    global currentAction
    try:
        ret = client.get_ticker(symbol=symbol)
        return float(ret["lastPrice"])
    except Exception as e:
        printDateTime()
        print "get ticker error"
        tradeFlag = True
        currentAction = 0
        print(e)
        # sendPush(str(e))


def get_order_book(symbol, limit):
    try:
        ret = client.get_order_book(symbol=symbol, limit=5)

        return ret

    except Exception as e:
        printDateTime()
        print "get orderbook error"
        print(e)
        # sendPush(str(e))
        return None


def get_price_qty_from_orderbook(orderbook, c):
    bidList = []
    askList = []
    if len(orderbook['bids']) >= c:
        count = c
    else:
        count = len(orderbook['bids'])
    for i in xrange(count):
        bidList.append((float(orderbook['bids'][i][0]),
                        float(orderbook['bids'][i][1])))
        askList.append((float(orderbook['asks'][i][0]),
                        float(orderbook['asks'][i][1])))

    return bidList, askList


def get_lastPrice(orderbook, return_type):

    priceTuple = None, None, None, None

    try:

        if return_type == 'float':
            priceTuple = float(orderbook['bids'][0][0]), float(orderbook['bids'][0][
                1]), float(orderbook['asks'][0][0]), float(orderbook['asks'][0][1])
        else:
            priceTuple = orderbook['bids'][0][0], (orderbook['bids'][0][1]), orderbook[
                'asks'][0][0], orderbook['asks'][0][1]

    except Exception as e:
        print(e)

    return priceTuple


def get_proper_qty(symbol):

    try:
        trades = client.get_recent_trades(symbol=symbol, limit=50)
        qtyList = []
        for t in trades:
            qtyList.append(float(t['qty']))

        # qty = numpy.median(qtyList) * qtyOffset
        qty = numpy.percentile(qtyList, QuantityPosition) * qtyOffset

        return qty

    except Exception as e:
        printDateTime()
        print "get proper qty error"
        print(e)
        # sendPush(str(e))
        return 0


def sellCurrentFundIfNeeded():
    printMsg("sellCurrentFundIfNeeded")
    needToSellFund = checkSellFundIfNeed(symbol, 0)
    if needToSellFund != -1:
        # try to sell
        qty = getAvailableQty(needToSellFund)
        order_market_sell(symbol, qty)

    setCurrentActionToZero()


def cancelAllCurrentOrder():
    printMsg("Cancel All CurrentOrder")
    # get all orders, and cancel it
    orders = client.get_open_orders(symbol=symbol)
    # printMsg("Current orders: %s " % orders)
    for order in orders:
        orderId = order["orderId"]
        side = order["side"]
        if cancel_order(symbol, orderId) == True:
            printMsg("[Canceled]%s Order: %s" % (side, orderId))
            # if side == "SELL":
            #     sellCurrentFundIfNeeded()

    sellCurrentFundIfNeeded()


def sellRemainAltCryptoIfNeeded():
    printMsg("sellRemainAltCryptoIfNeeded ")
    for s in tradedSymbols:
        try:
            s_balance = client.get_asset_balance(
                asset=s.replace(baseCryptoName, ""))
            free_fund = s_balance['free']
            # try to sell
            qty = getAvailableQty(free_fund)
            sellThread = Thread(target=order_market_sell, args=(s, qty,))
            sellThread.daemon = True
            sellThread.start()
            time.sleep(0.5)
        except Exception as e:
            print(e)
        # order_market_sell(s, qty)
        # sell it with market


def changeTradeSymbol(tickerSymbol, tradeMode):
    global tradeFlag
    global symbol
    global TRADE_MODE
    global FailActionMax
    global ma7Queue
    global ma25Queue
    global ma7
    global ma25
    global QuantityPosition

    tradeFlag = False  # stop trading
    waitCount = 0

    while currentAction != 0:
        # still in trade, wait until finished
        printMsg("Wait for current trade")
        time.sleep(10)
        if waitCount >= 6:
            printMsg("Force stop current trade, start new market trading")
            # force sell current symbol trades
            cancelAllCurrentOrder()
            break
        waitCount += 1
    # reset action

    # sellRemainAltCryptoIfNeeded()
    cancelAllCurrentOrder()

    symbol = tickerSymbol
    setRequirements(symbol)
    TRADE_MODE = tradeMode

    if TRADE_MODE == "NewList":
        FailActionMax = 2
    else:
        FailActionMax = 3

    ma7Queue = collections.deque(maxlen=2)
    ma25Queue = collections.deque(maxlen=2)
    ma7 = 0
    ma25 = 0

    QuantityPosition = INITQuantityPosition
    tradeFlag = True
    printMsg("Start trade %s" % symbol)
    # start trade

# Detect new listed market


def checkNewMarket():
    global allMarket
    global symbol
    global tradeFlag
    global currentAction
    global TRADE_MODE

    if ISNEWLISTED == True:

        try:
            tickers = client.get_all_tickers()
            for ticker in tickers:
                tickerSymbol = ticker['symbol']
                lastPrice = float(ticker['price'])
                if lastPrice > 0.0:
                    allMarket.add(tickerSymbol)
            print "Start monitoring new market"
            time.sleep(1)
            while True:
                try:
                    newTickers = client.get_all_tickers()
                    for t in newTickers:
                        tickerSymbol = t['symbol']
                        lastPrice = float(t['price'])
                        if tickerSymbol not in allMarket and lastPrice > 0.0:
                            if baseCryptoName in tickerSymbol:
                                print "Prepare for new listed market"
                                print "%s: %.8f" % (tickerSymbol, lastPrice)
                                changeTradeSymbol(tickerSymbol, "NewList")

                            allMarket.add(tickerSymbol)

                except Exception as e:
                    printDateTime()
                    print "check new market inside error"
                    print(e)
                    # sendPush(str(e))

                time.sleep(20)

        except Exception as e:
            printDateTime()
            print "check new market error"
            print(e)
            # sendPush(str(e))


def setRequirements(newSymbol):
    global client
    global DIGIT
    global INCREASING
    global currentMarket
    global startTime
    global MIN_NOTIONAL
    global stepSize
    global original_fund
    global previous_fund
    global baseCryptoName
    global symbol
    global INCREASING
    global tradedSymbols

    tradedSymbols.add(newSymbol)
    print "Traded symbols: %s" % tradedSymbols

    symbol = newSymbol

    print '%%%s profit for scanning %s' % (PROFIT, symbol)

    INCREASING = "%s" % getTickSize(symbol)

    print "INCREASING: %s" % INCREASING

    MIN_NOTIONAL = float(getMinNotional(symbol))

    print "MIN_NOTIONAL: %f" % MIN_NOTIONAL

    stepSize = float(getStepSize(symbol))

    print "STEP_SIZE: %f" % stepSize

    print "DEBUG_MODE: %s" % DEBUG_MODE

    print "QUANTITY: " + str(QUANTITY)

    baseCrypto = Crypto(baseCryptoName, 0, 0, 0)
    baseCrypto_fund = baseCrypto.getBalance(client)
    sym = Crypto(symbol[:-3], 0, 0, 0)
    symBalance = sym.getBalance(client)
    symPrice = get_ticker(symbol)
    symFund = float(symBalance) * float(symPrice)

    bnb = Crypto("BNB", 0, 0, 0)
    bnbBalance = bnb.getBalance(client)
    bnbPrice = get_ticker("BNB" + baseCryptoName)
    bnbFund = float(bnbBalance) * float(bnbPrice)

    if previous_fund == 0:
        previous_fund = float(baseCrypto_fund) + \
            float(symFund) + float(bnbFund)
        original_fund = previous_fund


def sendPush(msg):
    dateTime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    push = Push(rest_key, app_id)
    push.sendPush("[%s]%s: %s" % (baseCryptoName, dateTime, msg))


def setCurrentActionToZero():
    global currentAction
    printMsg("Set currentAction to 0")
    # lock.acquire(False)
    currentAction = 0
    # lock.release()


# Long Methods


def checkIfBuyFilled_long(tran):

    orderId = None
    symbol = tran.symbol

    time.sleep(WAIT_BUY_FILLED_TIME)

    while True:

        orderId = tran.buyOrderId
        if orderId != None:
            order = get_order(symbol, orderId)
            if order != None:
                status = order['status']
                printMsg("[IFBUYFILLED Check] Status: %s" % status)
                if status == "FILLED":
                    # start selling
                    if sell(tran, True) == True:
                        checkIfNeedToCancelSellThread = Thread(target=checkIfNeedToCancelSell_long, args=(
                            symbol, tran.currentQty, tran, False))
                        checkIfNeedToCancelSellThread.daemon = True
                        checkIfNeedToCancelSellThread.start()

                        checkIfNeedToCancelPartialSellThread = Thread(
                            target=checkIfNeedToCancelPartialSell_long, args=(symbol, tran.currentQty, tran,))
                        checkIfNeedToCancelPartialSellThread.daemon = True
                        checkIfNeedToCancelPartialSellThread.start()
                    else:
                        setCurrentActionToZero()

                    break
                elif status == "CANCELED":
                    break
        time.sleep(WAIT_BUY_FILLED_TIME)


def checkIfNeedToCancelBuy_long(tran):
    global currentAction

    orderId = None
    symbol = tran.symbol
    time.sleep(WAIT_BUY_FILLED_TIME)
    localWaitTime = WAIT_BUY_FILLED_TIME

    while True:
        orderId = tran.buyOrderId
        if orderId != None:
            order = get_order(symbol, orderId)
            if order != None:
                status = order['status']
                printMsg("[IFNEEDTOCANCEL BUY Check] Status: %s" % status)
                if status == "NEW":
                    priority, priorityQty = getPriceInHighPriority(
                        tran.symbol, tran.price.buyPrice, "bid")

                    if priority != -1:
                        printMsg("[WAIT]Buy Price In HIGH Priority [%d], keep waiting . Buy Order : %s" % (
                            priority, orderId))
                        printMsg("[WaitBuy]: Price: %s, Book Qty:%s" %
                                 (tran.price.buyPrice, priorityQty))
                        time.sleep(localWaitTime)
                        localWaitTime += WAIT_TIME_OFFSET
                        continue
                    else:
                        if cancel_order(symbol, orderId) == True:
                            printMsg("[Canceled]Buy Order: %s" % orderId)
                            setCurrentActionToZero()
                            break
                        else:
                            # Cancel buy fail, checkIfBuyFilled_long will sell
                            break
                else:
                    break

        time.sleep(localWaitTime)
        localWaitTime += WAIT_TIME_OFFSET


def checkIfBuyPartialFilled_long(tran):

    orderId = None
    symbol = tran.symbol
    time.sleep(WAIT_BUY_FILLED_TIME)
    localWaitTime = WAIT_BUY_FILLED_TIME

    while True:
        orderId = tran.buyOrderId
        if orderId != None:
            order = get_order(symbol, orderId)
            if order != None:
                status = order['status']
                printMsg("[PARTIALLY_FILLED BUY Check] Status: %s" % status)
                if status == "PARTIALLY_FILLED":
                    # sell(tran, True)
                    printMsg(
                        "[PARTIALLY_FILLED]PartiallyBuy Order: %s" % orderId)
                    priority, priorityQty = getPriceInHighPriority(
                        tran.symbol, tran.price.buyPrice, "bid")
                    if priority != -1:
                        printMsg("[WAIT]Buy Price In HIGH Priority [%d], keep waiting. Buy Order : %s" % (
                            priority, orderId))
                        printMsg("[WaitBuy]: Price: %s, Book Qty:%s" %
                                 (tran.price.buyPrice, priorityQty))
                        time.sleep(localWaitTime)
                        localWaitTime += WAIT_TIME_OFFSET
                        continue
                    else:
                        if cancel_order(symbol, orderId) == True:
                            printMsg(
                                "[Canceled]PartiallyBuy Order: %s" % orderId)
                            # Fail cancel part filled, start to sell
                            if sell(tran, True) == True:
                                checkIfNeedToCancelSellThread = Thread(target=checkIfNeedToCancelSell_long, args=(
                                    symbol, tran.currentQty, tran, False))
                                checkIfNeedToCancelSellThread.daemon = True
                                checkIfNeedToCancelSellThread.start()

                                checkIfNeedToCancelPartialSellThread = Thread(
                                    target=checkIfNeedToCancelPartialSell_long, args=(symbol, tran.currentQty, tran,))
                                checkIfNeedToCancelPartialSellThread.daemon = True
                                checkIfNeedToCancelPartialSellThread.start()
                            else:
                                setCurrentActionToZero()

                        else:
                            printMsg(
                                "[PARTIALLY_FILLED]Cancel Order Failed: %s" % orderId)

                elif status == "FILLED":
                    break
                elif status == "CANCELED":
                    # setCurrentActionToZero()
                    break
                else:
                    time.sleep(localWaitTime)
                    localWaitTime += WAIT_TIME_OFFSET
                    continue

        time.sleep(localWaitTime)
        localWaitTime += WAIT_TIME_OFFSET


def checkIfNeedToCancelSell_long(symbol, quantity, tran, isPart):
    global failActions
    global currentMarket
    global currentAction

    orderId = None
    time.sleep(WAIT_BUY_FILLED_TIME)

    localWaitTime = PARTIAL_FILLED_TIME

    while True:
        orderId = tran.sellOrderId
        if orderId != None:
            order = get_order(symbol, orderId)
            if order != None:
                status = order['status']
                if status == "NEW":
                    # if still in high priority, wait again
                    priority, priorityQty = getPriceInHighPriority(
                        tran.symbol, tran.price.sellPrice, "ask")
                    if priority != -1:
                        printMsg("[WAIT]Sell Price In HIGH Priority [%d], keep waiting. Sell Order : %s" % (
                            priority, orderId))
                        printMsg("[WaitSell]: Price: %s, Book Qty:%s" %
                                 (tran.price.sellPrice, priorityQty))
                        time.sleep(localWaitTime)
                        localWaitTime += WAIT_TIME_OFFSET
                        continue
                    else:
                        if cancel_order(symbol, orderId) == True:
                            printMsg("[Canceled]Sell Order: %s" % orderId)
                            # Resell with lower price
                            if sell(tran, False) == False:
                                setCurrentActionToZero()
                            # break
                        else:
                            # Cancel sell fail, trading done
                            print tranFinalMsg(tran)
                            break
                        time.sleep(1)

                elif status == "FILLED":
                    # trading done
                    print tranFinalMsg(tran)
                    break
                # elif status == "CANCELED":
                #     break
                elif tran.isFinished == True:
                    setCurrentActionToZero()
                    break
                else:
                    time.sleep(localWaitTime)
                    localWaitTime += WAIT_TIME_OFFSET
                    continue

        time.sleep(localWaitTime)
        localWaitTime += WAIT_TIME_OFFSET


def checkIfNeedToCancelPartialSell_long(symbol, quantity, tran):
    global failActions
    global currentMarket
    global currentAction

    orderId = None
    cancelTime = 0
    time.sleep(WAIT_BUY_FILLED_TIME)
    localWaitTime = PARTIAL_FILLED_TIME

    while True:
        orderId = tran.sellOrderId
        if orderId != None:
            order = get_order(symbol, orderId)
            if order != None:
                status = order['status']
                printMsg("[PARTIALLY_FILLED SELL Check] Status: %s" % status)
                if status == "PARTIALLY_FILLED":
                    printMsg(
                        "[PARTIALLY_FILLED]PartiallySell Order: %s" % orderId)
                    priority, priorityQty = getPriceInHighPriority(
                        tran.symbol, tran.price.sellPrice, "ask")

                    if priority != -1:
                        printMsg("[WAIT]Sell Price In HIGH Priority [%d], keep waiting. Sell Order : %s" % (
                            priority, orderId))
                        printMsg("[WaitSell]: Price: %s, Book Qty:%s" %
                                 (tran.price.sellPrice, priorityQty))
                        time.sleep(localWaitTime)
                        localWaitTime += WAIT_TIME_OFFSET
                        continue

                    else:
                        if cancel_order(symbol, orderId) == True:
                            printMsg(
                                "[Canceled]PartiallySell Order: %s" % orderId)
                            if sell(tran, False) == False:
                                setCurrentActionToZero()
                        else:
                            # order filled
                            printMsg(
                                "[PARTIALLY_FILLED]Cancel Order Failed: %s" % orderId)
                            print tranFinalMsg(tran)
                        time.sleep(PARTIAL_FILLED_TIME)

                elif status == "FILLED":
                    break
                elif status == "CANCELED":
                    if cancelTime >= 20:
                        break
                    cancelTime += 1
                #     break
                elif tran.isFinished == True:
                    # currentAction = 0
                    setCurrentActionToZero()
                    break
                else:
                    time.sleep(localWaitTime)
                    localWaitTime += WAIT_TIME_OFFSET
                    continue

        time.sleep(localWaitTime)
        localWaitTime += WAIT_TIME_OFFSET


def sell(tran, isNeedCmpPrice):
    needToSellFund = checkSellFundIfNeed(symbol, 0)
    retry = 0

    while True:
        if needToSellFund != 0 and float(getAvailableQty(needToSellFund)) != 0:
            break
        else:
            if retry >= 60:
                break
            retry += 1
            time.sleep(1)
            printMsg("fund error, get again")
            needToSellFund = checkSellFundIfNeed(symbol, 0)

    if needToSellFund != -1:
        qty = getAvailableQty(needToSellFund)
        tran.currentQty = qty

        lastPrice = get_ticker(symbol)
        orderBook = get_order_book(symbol, 5)
        lastBid, lastBidQty, lastAsk, lastAskQty = get_lastPrice(
            orderBook, "float")

        if all((lastBid, lastAsk)):
            newPrice = getSellPriceAndBuyPrice(
                lastAsk, lastBid, lastPrice)
            newSellPrice = newPrice.sellPrice
            oldSellPrice = tran.price.sellPrice
            oldBuyPrice = tran.price.buyPrice

            finalSellPrice = ''

            if isNeedCmpPrice:
                if float(newSellPrice) < float(oldSellPrice) and float(newSellPrice) > float(oldBuyPrice):
                    finalSellPrice = newSellPrice
                else:
                    finalSellPrice = oldSellPrice
            else:
                finalSellPrice = newSellPrice

            tran.price.sellPrice = finalSellPrice

            return order_limit_sell(symbol, tran.currentQty, tran.price.sellPrice, tran)

        return False

    else:
        return False


# Main action


def trade(symbol, price):

    global currentAction

    BNBThread = Thread(target=checkBNB, args=())
    BNBThread.daemon = True
    BNBThread.start()

    global featureStr

    qty = 0

    try:

        if QUANTITY == "MAX":
            btc_c = Crypto(baseCryptoName, 0, 0, 0)
            btc_bal = btc_c.getBalance(client)
            qty = float(getAvailableQty(float(btc_bal) /
                                        float(price.buyPrice)))
            # printMsg("Use Max Qty:%.8f" % qty)

        elif QUANTITY == "AUTO":
            market_proper_qty = float(getAvailableQty(
                get_proper_qty(symbol)))

            # printMsg("AUTO QTY: %.8f" % market_proper_qty)
            btc_c = Crypto(baseCryptoName, 0, 0, 0)
            btc_bal = btc_c.getBalance(client)
            max_qty = float(getAvailableQty(float(btc_bal) /
                                            float(price.buyPrice)))

            if market_proper_qty > max_qty:
                qty = max_qty
                # printMsg("Use Max Qty:%.8f" % qty)
            else:
                if checkIfReachMinNot(price.buyPrice, market_proper_qty):
                    qty = market_proper_qty
                    # printMsg("Use Auto Qty:%.8f" % qty)
                else:
                    minQty = float(MIN_NOTIONAL) / \
                        float(price.buyPrice)
                    qty = math.ceil(
                        float(minQty) * math.pow(10, logStep)) / math.pow(10, logStep)
                    # printMsg("Use Min Qty:%.8f" % qty)

        else:
            qty = float(QUANTITY)
            # printMsg("Use Manual Qty:%.8f" % qty)

        featureStr += "%f," % (qty)

        tran = Tran(symbol, price, None, None, None, None,
                    _chop(qty, 3), "", None, "", False)

        buyThread = None
        checkFillThread = None
        checkCancelThread = None

        if qty > 0.0:

            if order_limit_buy(tran.symbol, tran.totalQuantity, tran.price.buyPrice, tran) == True:
                checkFillThread = Thread(
                    target=checkIfBuyFilled_long, args=(tran,))
                checkFillThread.daemon = True

                checkCancelThread = Thread(
                    target=checkIfNeedToCancelBuy_long, args=(tran,))
                checkCancelThread.daemon = True

                checkIfBuyPartialFillThread = Thread(
                    target=checkIfBuyPartialFilled_long, args=(tran,))
                checkIfBuyPartialFillThread.daemon = True

                checkFillThread.start()
                checkCancelThread.start()
                checkIfBuyPartialFillThread.start()

                printMsg("[LimitBuy]: %s x %s" %
                         (tran.price.buyPrice, tran.totalQuantity))
            printDateTime()
            printMsg("[Long] buy %s sell %s" %
                     (price.buyPrice, price.sellPrice))

    except Exception as e:
        print(e)
        currentAction = 0


def MAWatcher():

    global ma7
    global ma25
    global ma7Queue
    global ma25Queue
    global klines

    while True:

        try:
            klines = client.get_historical_klines(
                symbol, Client.KLINE_INTERVAL_1MINUTE, "30 minutes ago UTC")
            ma7 = getMA(klines, 7)
            ma25 = getMA(klines, 25)

            if ma7 != -1 and ma25 != -1:
                if len(ma7Queue) == 0:
                    ma7Queue.append(ma7)
                else:
                    if ma7 != ma7Queue[len(ma7Queue) - 1]:
                        ma7Queue.append(ma7)

                if len(ma25Queue) == 0:
                    ma25Queue.append(ma25)
                else:
                    if ma25 != ma25Queue[len(ma25Queue) - 1]:
                        ma25Queue.append(ma25)

        except Exception as e:
            print(e)

        time.sleep(30)


def isTrendGrow():
    global featureStr
    if len(ma7Queue) == 2 and len(ma25Queue) == 2:
        previousMA7 = ma7Queue[0]
        previousMA25 = ma25Queue[0]
        lastMA7 = ma7Queue[1]
        lastMA25 = ma25Queue[1]

        slopeMA7 = (lastMA7 - previousMA7)
        slopeM25 = (lastMA25 - previousMA25)

        if slopeMA7 > slopeM25 and slopeMA7 > 0 and lastMA7 > lastMA25:
            printMsg("slopeMA7: %.8f, slopeM25: %.8f" % (slopeMA7, slopeM25))
            printMsg("LastMA7:%.8f, LastMA25:%.8f, PreviousMA7:%.8f, PreviousMA25:%.8f" % (
                lastMA7, lastMA25, previousMA7, previousMA25))
            featureStr += "%.8f,%.8f," % (slopeMA7, slopeM25)
            return True
        else:
            return False
    elif ma7 == -1 or ma25 == -1:  # no ma7 or no ma25, new listed market
        return True
    else:
        return False


def action(symbol):

    global lastPriceQueue
    global previousLastPrice
    global featureStr
    global currentAction
    global NewListTradeCount
    global QuantityPosition

    if currentAction == 0:

        orderBook = get_order_book(symbol, 5)
        if orderBook != None:

            lastBid, lastBidQty, lastAsk, lastAskQty = get_lastPrice(
                orderBook, "float")

            if all((lastBid, lastAsk)) and lastAsk > lastBid:

                price = getSellPriceAndBuyPrice(lastAsk, lastBid, lastBid)
                profitablePrice = lastBid + (lastBid * PROFIT / 100)

                # print "lastAsk: %.8f
                # profitablePrice:%.8f,buyPrice:%.8f,sellPrice:%.8f" %
                # (lastAsk, profitablePrice,float(price.buyPrice),
                # float(price.sellPrice))

                if (lastAsk > profitablePrice) and float(price.sellPrice) - float(price.buyPrice) > 0.00000002:

                    # if TRADE_MODE == "Normal" or TRADE_MODE == "NewList":
                    featureStr = ""

                    if ISUSEMA == False:
                        trendGrow = True
                    else:
                        trendGrow = isTrendGrow()

                    if trendGrow == True:

                        if TRADE_MODE == "NewList":
                            currentAction = 1
                            trade(symbol, price)

                        elif TRADE_MODE == "Normal":
                            currentAction = 1
                            trade(symbol, price)

                        if QuantityPosition >= 25:
                            QuantityPosition -= 5

                        if len(klines) > 0:

                            if currentAction == 1:
                                featureStr += "%.8f,%.8f," % (ma7, ma25)
                                bidList, askList = get_price_qty_from_orderbook(
                                    orderBook, 5)
                                featureStr += "%s,%s," % (price.buyPrice,
                                                          price.sellPrice)
                                for ask in askList:
                                    ask_price = ask[0]
                                    ask_qty = ask[1]
                                    # printMsg("[ASK]%.8f,%.8f" % (ask_price, ask_qty))
                                    featureStr += "%.8f,%.8f," % (
                                        ask_price, ask_qty)

                                for bid in bidList:
                                    bid_price = bid[0]
                                    bid_qty = bid[1]
                                    # printMsg("[BID]%.8f,%.8f" % (bid_price, bid_qty))
                                    featureStr += "%.8f,%.8f," % (
                                        bid_price, bid_qty)

                                candle = klines[0]
                                openP = float(candle[1])
                                closeP = float(candle[4])

                                featureStr += "%s,%s,%s,%s,%s,%s,%d," % (
                                    candle[1], candle[2], candle[3], candle[4], candle[5], candle[7], candle[8])

                                lastPrice = get_ticker(symbol)
                                featureStr += "%f," % (lastPrice)

                                # feature Sting format
                                # (slopeMA7,slopeMA25,buyPrice,sellPrice,ask1,askqty1,ask2,askqty2,ask3,askqty3,ask4,askqty4,ask5,askqty5,bid1,bidqty1,bid2,bidqty2,bid3,bidqty3,bid4,bidqty4,bid5,bidqty5,open,high,low,close,volume,quotevolume,lastprice,qty,profit,result)


def getInfo(argv):

    global symbol
    global PROFIT
    global QUANTITY
    global currentMarket
    global DEBUG_MODE
    global TOGGLE_DYNAMIC
    global baseCryptoName
    global stopLoss
    global PRICE_MODE
    global TRADE_MODE
    global ISAUTOMARKET
    global ISNEWLISTED
    global ISUSEMA
    global STOPLOSSBY
    global QuantityPosition
    global INITQuantityPosition

    try:
        opts, args = getopt.getopt(
            argv, "hb:s:p:q:d:l:m:t:a:n:u:x:w:", ["BASE=", "SYMBOL=", "PROFIT=", "QUANTITY=", "DEBUGMODE=", "STOPLOSS=", "PRICEMODE=", "TRADEMODE=", "AUTOMARKET=", "NEWLISTED=", "MA=", "STOPLOSSBY=", "POSITION="])
    except getopt.GetoptError:
        print 'python trader.py -b <BASE> -s <SYMBOL> -p <PROFIT> -q <QUANTITY> -d <DEBUGMODE> -l <STOPLOSS> -m <PRICEMODE> -t <TRADEMODE> -a <AUTOMARKET> -n <NEWLISTED> -u <MA> -x <STOPLOSSBY> -w <POSITION>'
        sys.exit(2)

    for opt, arg in opts:

        if opt == '-h':
            print 'python trader.py -b <BASE> -s <SYMBOL> -p <PROFIT> -q <QUANTITY> -d <DEBUGMODE> -l <STOPLOSS> -m <PRICEMODE> -t <TRADEMODE> -a <AUTOMARKET> -n <NEWLISTED> -u <MA> -x <STOPLOSSBY> -w <POSITION>'
            sys.exit()
        elif opt in ("-b", "--BASE"):
            baseCryptoName = arg
        elif opt in ("-s", "--SYMBOL"):
            symbol = arg
        elif opt in ("-p", "--PROFIT"):
            PROFIT = float(arg)
        elif opt in ("-q", "--QUANTITY"):
            if arg == "MAX":
                QUANTITY = "MAX"
            elif arg == "AUTO":
                QUANTITY = "AUTO"
            else:
                QUANTITY = float(arg)

        elif opt in ("-d", "--DEBUGMODE"):
            if arg == "t":
                DEBUG_MODE = True
            else:
                DEBUG_MODE = False

        elif opt in ("-t", "--TOGGLEDYNAMIC"):

            if arg == "t":
                TOGGLE_DYNAMIC = True
            else:
                TOGGLE_DYNAMIC = False
        elif opt in ("-l", "--STOPLOSS"):
            stopLoss = float(arg)
        elif opt in ("-m", "--PRICEMODE"):
            if arg == "i":
                PRICE_MODE = "increasing"
            elif arg == "a":
                PRICE_MODE = "average"

        elif opt in ("-t", "--TRADEMODE"):
            if arg == "n":
                TRADE_MODE = "Normal"
            elif arg == "l":
                TRADE_MODE = "NewList"

        elif opt in ("-a", "--AUTOMARKET"):
            if arg == "t":
                ISAUTOMARKET = True
            else:
                ISAUTOMARKET = False
            print "ISAUTOMARKET: %d" % ISAUTOMARKET

        elif opt in ("-n", "--NEWLISTED"):
            if arg == "t":
                ISNEWLISTED = True
            else:
                ISNEWLISTED = False
            print "NEWLISTED: %d" % ISNEWLISTED

        elif opt in ("-u", "--MA"):
            if arg == "t":
                ISUSEMA = True
            else:
                ISUSEMA = False
            print "ISUSEMA: %d" % ISUSEMA

        elif opt in ("-x", "--STOPLOSSBY"):
            if arg == "a":
                STOPLOSSBY = "ASSET"
            elif arg == "p":
                STOPLOSSBY = "PROFIT"

            print "STOPLOSSBY: %s" % STOPLOSSBY

        elif opt in ("-w", "--POSITION"):
            QuantityPosition = float(arg)
            INITQuantityPosition = QuantityPosition
            print "QuantityPosition: %s" % arg

    return (baseCryptoName, symbol, PROFIT, QUANTITY)


def main():
    global startTime
    global tradeFlag
    global client

    client = Client(config.api_key,
                    config.api_secret)

    newSymbol = symbol + baseCryptoName
    setRequirements(newSymbol)

    checkNewMarketThread = Thread(target=checkNewMarket, args=())
    checkNewMarketThread.daemon = True
    checkNewMarketThread.start()

    checkProperMarketThread = Thread(target=detectProperMarket, args=())
    checkProperMarketThread.daemon = True
    checkProperMarketThread.start()

    MAWatcherThread = Thread(target=MAWatcher, args=())
    MAWatcherThread.daemon = True
    MAWatcherThread.start()

    startTime = time.time()

    while True:
        if tradeFlag == True and currentAction == 0:
            thread = Thread(target=action, args=(symbol,))
            thread.daemon = True
            thread.start()
        time.sleep(WAIT_TIME)


if __name__ == "__main__":
    info = getInfo(sys.argv[1:])
    if info != ('', '', 0, 0):
        main()
    else:
        print 'python trader.py -b <BASE> -s <SYMBOL> -p <PROFIT> -q <QUANTITY> -d <DEBUGMODE> -l <STOPLOSS> -m <PRICEMODE> -t <TRADEMODE> -a <AUTOMARKET> -n <NEWLISTED> -u <MA> -x <STOPLOSSBY> -w <POSITION>'
