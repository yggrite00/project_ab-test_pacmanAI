# project_ab-test_pacmanAI
#import packages dan import dataset
```python
#data https://data.jakarta.go.id/dataset/data-penumpang-bus-transjakarta-januari-2021
#menggabungkan file csv menjadi satu

import numpy as np
import pandas as pd
from pandas import Series, DataFrame

df=pd.read_csv('marketing_AB.csv')
df.head()
```
#menghitung jumlah sample
```python
#menentukan jumlah sample 
#mencari standar deviasi conversion rate  psa
std = np.std(df['converted'])
std

# z alpha/2 dan Z beta
z_stats= stats.norm.ppf(1-0.05/2)
z_sb = stats.norm.ppf(1-0.2)

#menghitung jumlah sample yang di butuhkan untuk 2 variant
n = 2*(z_stats+z_sb)**2 * std**2/0.01**2# perbedaan yang diinginkan antara control dan threatment
print(f'jumlah sample yang dibutuhkan adalah {n*2} atau {n} sample per variant')
```

#sampling control variant dan treatment variant
```python
#sampling control variant (psa) 
data_psa = df.loc[df['test group']=='psa']
sample_psa = data_psa.sample(3862)
#cek apakah ada sample yang duplikat
x = sample_psa.duplicated(['user id']).sum()
print(f' jumlah sample yang duplikat adalah {x}')
#cek ukuran sample
print (f'ukuran sample adalah {sample_psa.shape}')


#sampling treatment variant (ad) 
data_ad = df.loc[df['test group']=='ad']
sample_ad = data_ad.sample(3862)
#cek apakah ada sample yang duplikat
x = sample_ad.duplicated(['user id']).sum()
print(f' jumlah sample yang duplikat adalah {x}')
#cek ukuran sample
print (f'ukuran sample adalah {sample_ad.shape}')
```

#conversion rate masing-masing variant

```python
#mencari banyak user yang convert
psa_con= sample_psa['converted']==True
ad_con= sample_ad['converted']==True
print(f' jumlah psa yang convert adalah {psa_con.sum()}')
print(f' jumlah ad yang convert adalah {ad_con.sum()}')


#menghitung conversion rate psa dan ad
conversion_psa =psa_con.sum()/sample_psa.shape[0]
conversion_ad =ad_con.sum()/sample_ad.shape[0]
print(conversion_psa)
print(conversion_ad)
```

#menghitung z statistik, z critical, dan p-value

```python

#calculate Z statistic
import math
std =  math.sqrt((conversion_ad*(1-conversion_ad)/3862) + (conversion_psa*(1-conversion_psa)/3862))
Z = (conversion_ad-conversion_psa-0.01)/math.sqrt(std)

print(f'nilai z statistic adalah {Z}')


#critical region (left tail)
z_crit = stats.norm.ppf(0.05)
z_crit

#menghitung p-value
p_value = stats.norm.cdf(Z)
p_value

#kesimpulan
if (Z > z_crit) & (p_value >0.05):
    print('gagal tolak Ho')
else:
    print('tolak Ho')
    ```
