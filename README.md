# trader
An experimental binance trader inspired by @yasinkuyu 's [binance-trader](https://github.com/yasinkuyu/binance-trader)

## Usage
    Create api key from https://www.binance.com/userCenter/createApi.html
    
    rename sample_config.py to config.py and fill in api_key and api_secret
    
    pip install -r requirements.txt
    
    python trader.py -b <BASE> -s <SYMBOL> -p <PROFIT> -q <QUANTITY> -d <DEBUGMODE> -l <STOPLOSS> -m <PRICEMODE> -t <TRADEMODE> -a <AUTOMARKET> -n <NEWLISTED>
    
    <BASE> is your base currency, eg. BTC, ETH, BNB, USDT 
    <SYMBOL> is the currency you want to trade, eg. TRX, XVG, VIB....
    <PROFIT> is the profit that bot will try to make for you, eg. 0.5 (means 0.5%)
    <QUANTITY> is the quantity to trade, can be a fixed number or MAX(will trade all your fund) or AUTO (auto detect proper quantity), eg. 1000, MAX, AUTO
    <DEBUGMODE> is the bool to show the trade detail message, eg. t, f
    <STOPLOSS> if reach stoploss, the script will stop trading. (loss percentage of total fund), eg. -6.0, -4.0
    <PRICEMODE> the mode of calculating buy/sell price, Increasing mode means add and decrease a little of the last bid price, Average mode means use the average of last bid and last ask price,
     eg. i, a
    <TRADEMODE> is the strategy of normal market and new list market, they are the same right now, eg. n, l
    <AUTOMARKET> is the bool to automatically change trading market, eg. t, f
    <NEWLISTED> is the boll to automatically detect new listed market and trade, eg. t, f
    
    eg. python trader.py -b ETH -s VIB -p 0.4 -q MAX -d f -l -6.0 -m i -t n -a t -n f
    means trade VIB with ETH, target profit is 0.4%, use all fund to trade, disable debug message, stop loss is -6%, PRICEMODE is Increasing mode, TRADEMODE is normal and will change trading market when detected, and will not trade new-lsited markket
    
    eg. python trader.py -b BTC -s VIB -p 0.6 -q 1000 -d t -l -6.0 -m a -t n -a f -n t
    means trade VIB with BTC, target profit is 0.6%, quantity is 1000, enable debug message, stop loss is -6%, PRICEMODE is Average mode, TRADEMODE is normal and will not change trading market, and will trade new-lsited markket
    
    eg. python trader.py -b BTC -s VIB -p 0.6 -q AUTO -d t -l -6.0 -m a -t n -a f -n f
    means trade VIB with BTC, target profit is 0.6%, AUTO detect quantity, enable debug message, stop loss is -6%, PRICEMODE is Average mode, TRADEMODE is normal and will not change trading market, and will not trade new-lsited markket
    


 
## DISCLAIMER

    I am not responsible for anything done with this bot. 
    You use it at your own risk. 
    There are no warranties or guarantees expressed or implied. 
    You assume all responsibility and liability.
     
