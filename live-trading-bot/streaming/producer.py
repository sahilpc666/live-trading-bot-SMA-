import os
import json
import time
from confluent_kafka import Producer
from alpaca.data.live import StockDataStream

API_KEY = os.environ.get("ALPACA_API_KEY")
SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")
symbols = ["AAPL", "TSLA", "MSFT"]

producer = Producer({"bootstrap.servers": "kafka:9092"})

stream = StockDataStream(API_KEY, SECRET_KEY)

def delivery_report(err, msg):
    if err is not None:
        print(f"Failed to deliver: {err}")
    else:
        print(f"Sent to Kafka: {msg.value()[:50]}...") 

async def handle_quote(data):
    quote = {
        "symbol": data.symbol,
        "timestamp": str(data.timestamp),
        "bid_price": data.bid_price,
        "bid_size": data.bid_size,
        "bid_exchange": data.bid_exchange,
        "ask_price": data.ask_price,
        "ask_size": data.ask_size,
        "ask_exchange": data.ask_exchange,
        "conditions": ",".join(data.conditions) if data.conditions else None,
        "tape": data.tape,
    }
    producer.produce(
        "stock-quotes",
        value=json.dumps(quote, default=str).encode("utf-8"),
        callback=delivery_report
    )
    producer.poll(0)
    print(f"Queued: {quote['symbol']}")

stream.subscribe_quotes(handle_quote, *symbols)

if __name__ == "__main__":
    while True:
        try:
            print("Starting WebSocket stream...")
            stream.run()
        except Exception as e:
            print(f"WebSocket error: {e}. Reconnecting in 5s...")
            time.sleep(5)
        finally:
            producer.flush()