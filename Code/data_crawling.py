### DATA API - Time series examples ###
import yfinance as yf
import pandas as pd
import numpy as np

start = "2003-01-01"
end = "2023-12-31"

SP500_daily = yf.download("^GSPC", start, end)['Close']
DOWJONES_daily = yf.download("^DJI", start, end)['Close']
HANGSENG_daily = yf.download("^HSI", start, end)['Close']
GOLD_daily = yf.download("GC=F", start, end)['Close']
WTI_daily = yf.download("CL=F", start, end)['Close']

SP500_daily.name = 'S&P500'
DOWJONES_daily.name = 'DOWJONES'
HANGSENG_daily.name = 'HANGSENG'
GOLD_daily.name = 'GOLD'
WTI_daily.name = 'WTI_OIL'

# 동일한 인덱스를 기준으로 합침
result = pd.concat([SP500_daily, DOWJONES_daily, HANGSENG_daily, GOLD_daily, WTI_daily], axis=1, join='inner')

# 음수 값을 포함하는 행 제거: 모든 수치형 열에서 음수 검사
result = result[(result.select_dtypes(include=[np.number]) > 0).all(axis=1)]

print(f"result.shape: {result.shape}")

result.to_csv('./COSCI-GAN_Journal/Dataset/indices.csv')
