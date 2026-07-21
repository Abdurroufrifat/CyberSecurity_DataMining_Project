"""Create a tiny CICIDS-shaped dataset for smoke testing; not research data."""

from pathlib import Path
import numpy as np
import pandas as pd


FILES = [
    ("Monday-WorkingHours.pcap_ISCX.csv", ["BENIGN"]),
    ("Tuesday-WorkingHours.pcap_ISCX.csv", ["BENIGN", "FTP-Patator", "SSH-Patator"]),
    ("Wednesday-workingHours.pcap_ISCX.csv", ["BENIGN", "DoS Hulk", "DoS slowloris"]),
    ("Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv", ["BENIGN", "Web Attack ï¿½ XSS"]),
    ("Friday-WorkingHours-Morning.pcap_ISCX.csv", ["BENIGN", "Bot", "DDoS", "PortScan"]),
]


def main() -> None:
    out = Path(__file__).resolve().parent / "synthetic_dataset"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    for file_index, (name, labels) in enumerate(FILES):
        rows = []
        for i in range(240):
            label = labels[i % len(labels)]
            malicious = label != "BENIGN"
            rows.append({
                " Destination Port": float(80 + (i % 7) + file_index),
                " Flow Duration": float(rng.normal(1000 + 600 * malicious + 100 * file_index, 80)),
                " Total Fwd Packets": float(rng.integers(1, 12) + 4 * malicious),
                " Total Backward Packets": float(rng.integers(1, 10)),
                " Fwd Packet Length Mean": float(rng.normal(120 + 45 * malicious, 8)),
                " Bwd Packet Length Std": float(rng.normal(30 + 20 * malicious + 5 * file_index, 4)),
                " Flow Bytes/s": float(rng.normal(2000 + 900 * malicious, 100)),
                " SYN Flag Count": float(malicious),
                " Label": label,
            })
        frame = pd.DataFrame(rows)
        frame.to_csv(out / name, index=False, encoding="latin1", errors="replace")
    print(out)


if __name__ == "__main__":
    main()

