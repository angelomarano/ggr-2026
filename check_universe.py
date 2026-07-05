import sys
sys.path.insert(0, ".")
from data.universe import load_membership, constituents_at

m = load_membership("data/raw/sp500_membership.csv")
n = len(constituents_at(m, "2010-01-04"))
print("Tickers al 2010-01-04:", n, "(atteso: 499)")
