import time
import numpy as np
import pandas as pd
from ai_engine import estimate
rng=np.random.default_rng(42)
now=int(time.time())//60*60
n=300
t=np.arange(now-n*60,now,60)
close=75000+np.cumsum(rng.normal(0,18,n))
open_=np.r_[close[0],close[:-1]]
df=pd.DataFrame({'timestamp':t,'open':open_,'high':np.maximum(open_,close)+3,'low':np.minimum(open_,close)-3,'close':close,'volume':rng.uniform(1,10,n)})
for h in (10,15):
 r=estimate(df,h,now=time.time());assert 0<=r['probability']<=1 and r['validation_samples']>=35
 print(h,r['signal'],r['validation_accuracy'],r['baseline'])
try:
 estimate(df.iloc[:30],10)
 raise AssertionError('Should reject insufficient data')
except ValueError: pass
print('AI offline tests passed (synthetic data only)')
