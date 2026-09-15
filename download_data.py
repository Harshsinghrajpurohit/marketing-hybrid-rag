# download_martech.py — MarTech vendor 10-Ks from SEC EDGAR (public records).

import time
import urllib.request

# (label, CIK, accession) — all verified on SEC EDGAR.
VENDORS = [
    ("salesforce", 1108524, "000110852426000060"),
    ("hubspot",    1404655, "000119312526046646"),
    ("adobe",      796343,  "000079634326000003"),
    ("twilio",     1447669, "000144766926000021"),
]

for label, cik, acc in VENDORS:
    folder = acc
    fname = f"{acc[:10]}-{acc[10:12]}-{acc[12:]}.txt"
    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{folder}/{fname}"
    req = urllib.request.Request(url, headers={
        "User-Agent": "Marketing Assistant student project (local demo; contact: you@example.com)"
    })
    with urllib.request.urlopen(req, timeout=180) as r:
        raw = r.read().decode("utf-8", errors="replace")
    out = f"data/{label}_10k.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write(raw)
    print(f"{label}: {len(raw):,} chars -> {out}")
    time.sleep(3)  # polite to SEC's rate limit
print("Done.")
