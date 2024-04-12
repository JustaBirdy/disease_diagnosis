import pandas as pd
import os
import pickle
import numpy as np

order_list = []
for root,dirs,files in os.walk('./'):
    for i in files:
        if 'tsv' in i:
            file_name = i
            file_address = root + file_name
            df = pd.read_csv(file_address, delimiter='\t')
            print(i)
            order_data = df[df['rank'] == 'order']
            for index,rows in order_data.iterrows():
                temp = rows['taxName'].replace(' ','')
                if temp not in order_list:
                    order_list.append(temp)

with open('./order_list.pkl','wb') as f:
    pickle.dump(order_list,f)

with open('./order_list.pkl','rb') as f:
    order_list = pickle.load(f)
# print(order_list)

container = np.zeros((58,len(order_list)))
count = 0
for root,dirs,files in os.walk('./'):
    for i in files:
        if 'tsv' in i:
            file_name = i
            file_address = root + file_name
            df = pd.read_csv(file_address, delimiter='\t')
            order_data = df[df['rank'] == 'order']
            for index,rows in order_data.iterrows():
                temp = rows['taxName'].replace(' ','')
                pos = order_list.index(temp)
                container[count,pos] = rows['%']
            count+=1

df = pd.DataFrame(container)
df.to_csv('./order_data.csv',index=False)

df = pd.read_csv('./order_data.csv')
df.columns = order_list
df.to_csv('./order_data.csv',index=False)


# tsv_file = './ERR9452445.tsv'

# df = pd.read_csv(tsv_file, delimiter='\t')
# # df.to_csv(csv_file, index=False)

# # d = pd.read_csv('./output.csv')
# order_data = df[df['rank'] == 'order']
# # order_data = df[df['rank'] == 'order']