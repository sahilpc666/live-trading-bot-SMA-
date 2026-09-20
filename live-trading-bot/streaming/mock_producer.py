import json
from confluent_kafka import Producer
from stream_quotes import mock_fetch_data

producer = Producer({"bootstrap.servers": "kafka:9092"})

def normalize_quote(symbol, raw):
    return {
        "symbol": symbol,
        "timestamp": raw["t"],
        "bid_price": raw["bp"],
        "bid_size": raw["bs"],
        "bid_exchange": raw["bx"],
        "ask_price": raw["ap"],
        "ask_size": raw["as"],
        "ask_exchange": raw["ax"],
        "conditions": ",".join(raw.get("c", [])),
        "tape": raw.get("z"),
    }

def delivery_report(err, msg):
    if err is not None:
        print(f"Failed to deliver: {err}")
    else:
        print(f"Confirmed delivered: {msg.key()} -> partition {msg.partition()}, offset {msg.offset()}")

if __name__ == "__main__":
    data = mock_fetch_data()
    for symbol, raw in data["quotes"].items():
        quote = normalize_quote(symbol, raw)
        producer.produce(
            "stock-quotes",
            value=json.dumps(quote, default=str).encode("utf-8"),
            callback=delivery_report
        )
        producer.poll(0)
    producer.flush()