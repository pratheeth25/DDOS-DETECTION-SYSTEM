import random
import csv


OUTPUT_FILE = "live_traffic.csv"

PROTOCOLS = ["TCP", "UDP"]


def random_ip():
    choice = random.choice([1, 2, 3])

    if choice == 1:
        return f"192.168.{random.randint(0,255)}.{random.randint(1,254)}"
    elif choice == 2:
        return f"10.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
    else:
        return f"172.16.{random.randint(0,255)}.{random.randint(1,254)}"


def generate_row():

    traffic_type = random.choices(
        ["legit", "bot", "ddos"],
        weights=[0.4, 0.3, 0.3]
    )[0]

    # Legit traffic
    if traffic_type == "legit":
        rps = random.randint(5, 50)
        ports = random.randint(1, 4)
        packet = random.randint(200, 700)

    # Bot traffic
    elif traffic_type == "bot":
        rps = random.randint(100, 400)
        ports = random.randint(3, 8)
        packet = random.randint(600, 1100)

    # DDoS traffic
    else:
        rps = random.randint(700, 1500)
        ports = random.randint(8, 20)
        packet = random.randint(1000, 1500)

    ip = random_ip()
    protocol = random.choice(PROTOCOLS)

    return [
        ip,
        rps,
        ports,
        packet,
        protocol
    ]


def generate_file(rows=100):

    with open(OUTPUT_FILE, "w", newline="") as f:

        writer = csv.writer(f)

        # Header (NO label)
        writer.writerow([
            "ip",
            "requests_per_sec",
            "ports",
            "packet_size",
            "protocol"
        ])

        for _ in range(rows):
            writer.writerow(generate_row())

    print(f"Generated {rows} rows in {OUTPUT_FILE}")


if __name__ == "__main__":

    generate_file(200)   # Change number if you want
