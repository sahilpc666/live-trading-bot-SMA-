import os
from alpaca.data.live import StockDataStream

API_KEY = os.environ.get("ALPACA_API_KEY")
SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")

symbols = ["AAPL", "TSLA", "MSFT"]

stream = StockDataStream(API_KEY, SECRET_KEY)

# async def handle_quote(data):
#     print(f"Quote: {data.symbol} | Bid: {data.bid_price} @ {data.bid_size} | Ask: {data.ask_price} @ {data.ask_size} | Time: {data.timestamp}")

# async def handle_trade(data):
#     print(f"Trade: {data.symbol} | Price: {data.price} | Size: {data.size} | Time: {data.timestamp}")

# # Subscribe to quotes (bid/ask) and/or trades (executed prices)
# stream.subscribe_quotes(handle_quote, *symbols)
# stream.subscribe_trades(handle_trade, *symbols)

# if __name__ == "__main__":
#     print(f"Starting stream for: {symbols}")
#     stream.run()

def mock_fetch_data():
    return {'quotes': {'TSLA': {'ap': 373.26, 'as': 40, 'ax': 'V', 'bp': 336.78, 'bs': 40, 'bx': 'V', 'c': ['R'], 't': '2026-09-04T20:00:01.425745139Z', 'z': 'C'}, 'AAPL': {'ap': 338.27, 'as': 40, 'ax': 'V', 'bp': 305.33, 'bs': 40, 'bx': 'V', 'c': ['R'], 't': '2026-09-04T20:00:00.006211747Z', 'z': 'C'}, 'MSFT': {'ap': 525.57, 'as': 40, 'ax': 'V','bp': 474.49, 'bs': 40, 'bx': 'V', 'c': ['R'], 't': '2026-09-04T20:00:00.009232644Z', 'z': 'C'}}}