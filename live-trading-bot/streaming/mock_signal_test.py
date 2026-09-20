import time
import json
from confluent_kafka import Producer


producer = Producer({"bootstrap.servers": "kafka:9092"})


def send_quote(symbol, price):
    quote = {
        "symbol": symbol,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000000Z", time.gmtime()),

        "bid_price": round(price - 0.05, 2),
        "bid_size": 40,
        "bid_exchange": "V",

        "ask_price": round(price + 0.05, 2),
        "ask_size": 40,
        "ask_exchange": "V",

        "conditions": "R",
        "tape": "C",
    }

    producer.produce(
        "stock-quotes",
        value=json.dumps(quote).encode("utf-8"),
    )
    producer.poll(0)


if __name__ == "__main__":
    symbol = "AAPL"
    base_price = 230.0

    # 1. Fill signal engine's 50-tick window
    print("1. Sending flat prices...")
    for i in range(55):
        send_quote(symbol, base_price)
        time.sleep(0.1)
    print("Window filled.")

    # 2. Ramp UP
    print("2. Ramping price UP...")
    for i in range(30):
        price = base_price + (i * 0.05)
        send_quote(symbol, price)
        time.sleep(0.1)
    print("UP ramp complete.")

    # 3. Wait for cooldown
    print("3. Waiting for cooldown...")
    time.sleep(65)

    # 4. Ramp DOWN
    print("4. Ramping price DOWN...")
    for i in range(30):
        price = base_price + 1.5 - (i * 0.05)
        send_quote(symbol, price)
        time.sleep(0.1)
    print("DOWN ramp complete.")

    # 5. Flush Kafka producer
    producer.flush()
    print("Test complete.")