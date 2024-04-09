### DATA API - Time series examples ###
import yfinance as yf
import pandas as pd

start = "2009-05-01"
end = "2019-12-31"

SP500_daily = yf.download("^GSPC", start, end)['Close']
HANGSENG_daily = yf.download("^HSI", start, end)['Close']
GOLD_daily = yf.download("GC=F", start, end)['Close']
WTI_daily = yf.download("CL=F", start, end)['Close']

SP500_daily.name = 'S&P500'
HANGSENG_daily.name = 'HANGSENG'
GOLD_daily.name = 'GOLD'
WTI_daily.name = 'WTI_OIL'

# 동일한 인덱스를 기준으로 합침
result = pd.concat([SP500_daily, HANGSENG_daily, GOLD_daily, WTI_daily], axis=1, join='inner')
print(f"result.shape: {result.shape}")

result.to_csv('./COSCI-GAN_Journal/Dataset/indices.csv')
