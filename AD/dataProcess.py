import pandas as pd
import os
import pickle
import numpy as np

species_list = []
for root,dirs,files in os.walk('./'):
    for i in files:
        if 'tsv' in i:
            file_name = i
            file_address = root + file_name
            df = pd.read_csv(file_address, delimiter='\t')
            genus_data = df[df['rank'] == 'order']
            for index,rows in genus_data.iterrows():
                temp = rows['taxName'].replace(' ','')
                if temp not in species_list:
                    species_list.append(temp)

with open('./order_list.pkl','wb') as f:
    pickle.dump(species_list,f)

with open('./order_list.pkl','rb') as f:
    species_list = pickle.load(f)
# print(genus_list)

container = np.zeros((50,len(species_list)))
count = 0
for root,dirs,files in os.walk('./'):
    for i in files:
        if 'tsv' in i:
            file_name = i
            file_address = root + file_name
            df = pd.read_csv(file_address, delimiter='\t')
            species_data = df[df['rank'] == 'order']
            for index,rows in species_data.iterrows():
                temp = rows['taxName'].replace(' ','')
                pos = species_list.index(temp)
                container[count,pos] = rows['%']
            count+=1

df = pd.DataFrame(container)
df.to_csv('./order_data.csv',index=False)

df = pd.read_csv('./order_data.csv')
df.columns = species_list
df.to_csv('./order_data.csv',index=False)


# tsv_file = './ERR9452445.tsv'

# df = pd.read_csv(tsv_file, delimiter='\t')
# # df.to_csv(csv_file, index=False)

# # d = pd.read_csv('./output.csv')
# genus_data = df[df['rank'] == 'genus']
# # species_data = df[df['rank'] == 'species']