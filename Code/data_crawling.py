### DATA API - Time series examples ###
import yfinance as yf
import pandas as pd

start = "2009-05-01"
end = "2019-12-31"

SP500_daily = yf.download("^GSPC", start, end)['Close']
HANGSENG_daily = yf.download("^HSI", start, end)['Close']
KOSPI_daily = yf.download("^KS11", start, end)['Close']

SP500_daily.name = 'SP500'
HANGSENG_daily.name = 'HANGSENG'
KOSPI_daily.name = 'KOSPI'

# 동일한 인덱스(날짜)를 기준으로 합침
result = pd.concat([SP500_daily, HANGSENG_daily, KOSPI_daily], axis=1, join='inner')
print(f"result.shape: {result.shape}")

result.to_csv('./COSCI-GAN/Dataset/stocks.csv')
