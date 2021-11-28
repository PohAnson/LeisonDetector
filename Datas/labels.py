import pandas as pd
import os
import numpy as np

labels = np.array([])

df = pd.read_csv('DL_info.csv')
print(df.columns)
print(df.head)
ctr = 0
for i in os.listdir('tnpHU'):
    print(i)
    tmp = np.zeros((9))
    ctr +=1
    fn = i[:-4]+ '.png'
    ty = df[df['File_name'] == fn]['Coarse_lesion_type'].iloc[:]
    for i in ty:
        tmp[i] = 1
    if ctr != 1:
        labels = np.append(labels, [tmp], axis=0)
    else:
        labels = np.array([tmp])

np.save('labels', labels)