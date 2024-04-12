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
            print(i)
            species_data = df[df['rank'] == 'species']
            for index,rows in species_data.iterrows():
                temp = rows['taxName'].replace(' ','')
                if temp not in species_list:
                    species_list.append(temp)

with open('./species_list.pkl','wb') as f:
    pickle.dump(species_list,f)

with open('./species_list.pkl','rb') as f:
    species_list = pickle.load(f)
# print(species_list)

container = np.zeros((87,len(species_list)))
count = 0
for root,dirs,files in os.walk('./'):
    for i in files:
        if 'tsv' in i:
            file_name = i
            file_address = root + file_name
            df = pd.read_csv(file_address, delimiter='\t')
            species_data = df[df['rank'] == 'species']
            for index,rows in species_data.iterrows():
                temp = rows['taxName'].replace(' ','')
                pos = species_list.index(temp)
                container[count,pos] = rows['%']
            count+=1

df = pd.DataFrame(container)
df.to_csv('./species_data.csv',index=False)

df = pd.read_csv('./species_data.csv')
df.columns = species_list
df.to_csv('./species_data.csv',index=False)


# tsv_file = './ERR9452445.tsv'

# df = pd.read_csv(tsv_file, delimiter='\t')
# # df.to_csv(csv_file, index=False)

# # d = pd.read_csv('./output.csv')
# species_data = df[df['rank'] == 'species']
# # species_data = df[df['rank'] == 'species']