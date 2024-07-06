from urllib.request import urlretrieve
import pandas as pd
import requests
import json

TSD = "TIME_SERIES_DAILY"
SYMBOLS = ["LIN", "NEM", "GOOG", "META", "MTCH", "TMUS", "NFLX", "AMZN", "EBAY", 
           "TSLA", "GM", "MCD", "WMT", "PG", "KO", "XOM", "COP", "ET", "JPM", "C", 
           "V", "MA", "BX", "BLK", "BRK-B", "LLY", "JNJ", "UNH", "ABT", "VRTX", "GE",
           "RTX", "ETN", "CAT", "PLD", "AMT", "NVDA", "QCOM", "MSFT", "AAPL", "CRM", 
           "UBER", "AVGO", "ORCL", "ADBE", "IBM", "NEE", "SO"]

def gen_url(sym, key, func=TSD, size="full", dtype="csv"):
    base = r"https://www.alphavantage.co/query?"
    function = f"function={func}"
    symbol = f"symbol={sym}"
    outputsize = f"outputsize={size}"
    apikey = f"apikey={key}"
    datatype = f"datatype={dtype}"
    return f"{base}{function}&{symbol}&{outputsize}&{apikey}&{datatype}"
    

def gen_filename(sym, func_str="TSD", dtype="csv"):
    return f"data/{sym}_{func_str}.{dtype}"
    

def download_stock(sym, key, func=TSD, func_str="TSD", size="full", dtype="csv"):
    if sym in downloaded:
        print(f"{key} already downloaded")
        return
    filename = gen_filename(sym, func_str, dtype)
    url = gen_url(sym, key, func, size, dtype)
    response = requests.get(url)
    with open(filename, mode="wb") as file:
        print(f"creating {filename}")
        file.write(response.content)
        

def load_downloaded(dtype="csv") -> list:
    with open("downloaded.json", mode="r", encoding="utf-8") as file:
        downloaded = json.load(file)
    return downloaded[dtype]
    

def write_downloaded(downloaded: dict):
    with open("downloaded.json", mode="w", encoding="utf-8") as file:
        json.dump(downloaded, file)
        

def download_std(num_stocks, key):
    downloaded = load_downloaded()
    i = 0
    while num_stocks > 0 and i < len(SYMBOLS):
        if SYMBOLS[i] in downloaded:
            i = i + 1
            continue
        download_stock(SYMBOLS[i], key)
        downloaded.append(SYMBOLS[i])
        num_stocks = num_stocks - 1
        i = i + 1
    dl_dict = {"csv": downloaded}
    write_downloaded(dl_dict)