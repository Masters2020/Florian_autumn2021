#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


path = os.path.abspath('03_EDA.ipynb')  #this saves the path to this notebook and uses it as a base to access the required csvs
dname = os.path.dirname(path)
os.chdir(dname)


# In[3]:


#this code requires the csvs to be placed in a subfolder called data within the folder in which this notebook is placed
youtube = dname + '\\01_data\\you_text_label.csv'
reddit = dname + '\\01_data\\reddit.csv'


# ## **Youtube data**

# In[4]:


you_df = pd.read_csv(youtube)


# In[ ]:


labels = you_df["label"].tolist()
keys, counts = np.unique(labels, return_counts=True)


# In[8]:


labels_ = []
for l in labels:
    if l == 1:
        labels_.append('non-conspiratorial')
    else:
        labels_.append('conspiratorial')


# In[10]:


labels.count(1)


# In[11]:


labels.count(2)


# In[9]:


sns.set_theme(style="darkgrid")
sns.countplot(x = labels_)
plt.savefig('label distribution.png')
plt.show()


# In[16]:


#this code requires the csvs to be placed in a subfolder called data within the folder in which this notebook is placed
you_train = dname + '\\01_data\\train_you_text.csv'
you_test = dname + '\\01_data\\val_you_text.csv'


# In[17]:


you_train_ = pd.read_csv(you_train)
you_test_ = pd.read_csv(you_test)


# In[18]:


labels_train = you_train_["label"].tolist()
keys, counts = np.unique(labels, return_counts=True)


# In[19]:


labels_test = you_test_["label"].tolist()
keys, counts = np.unique(labels, return_counts=True)


# In[21]:


print(labels_train.count(1))
print(labels_train.count(2))


# In[22]:


print(labels_test.count(1))
print(labels_test.count(2))


# In[23]:


labelstrain_ = []
for l in labels_train:
    if l == 1:
        labelstrain_.append('non-conspiratorial')
    else:
        labelstrain_.append('conspiratorial')


# In[24]:


labelstest_ = []
for l in labels_test:
    if l == 1:
        labelstest_.append('non-conspiratorial')
    else:
        labelstest_.append('conspiratorial')


# In[39]:


plt.figure(figsize=(11, 4))
plt.subplot(121)
sns.set_theme(style="darkgrid")
sns.countplot(x = labelstrain_, order = ['non-conspiratorial', 'conspiratorial'])
plt.title('data distribution training set', pad = 15)
plt.subplot(122)
sns.set_theme(style="darkgrid")
sns.countplot(x = labelstest_, order = ['non-conspiratorial', 'conspiratorial'])
plt.title('data distribution test set', pad = 15)
plt.savefig('label distribution split.png')
plt.show()


# In[22]:


with open('added_tokens_you.json', 'r') as f:
    addtok_you = json.load(f)


# In[23]:


addtkeys = list(addtok_you.keys())
print(len(addtkeys))


# In[24]:


#this code requires the csvs to be placed in a subfolder called data within the folder in which this notebook is placed
train = dname + '\\01_data\\train_you_text.csv'
val = dname + '\\01_data\\val_you_text.csv'


# In[25]:


you_t = pd.read_csv(train)


# In[26]:


len(you_t)


# In[20]:


you_v = pd.read_csv(val)


# In[27]:


len(you_v)


# In[28]:


label_t = you_t["label"].tolist()


# In[29]:


label_v = you_v["label"].tolist()


# In[31]:


l1 = 0
l2 = 0
for i in label_t:
    if i == 1:
        l1 += 1
    else:
        l2 += 1
print(l1)
print(l2)
print(l1+l2)
print(l1/(l1+l2))
print(l2/(l1+l2))


# In[32]:


l1 = 0
l2 = 0
for i in label_v:
    if i == 1:
        l1 += 1
    else:
        l2 += 1
print(l1)
print(l2)
print(l1+l2)
print(l1/(l1+l2))
print(l2/(l1+l2))


# ## **Reddit data**

# In[31]:


red_df = pd.read_csv(reddit)


# In[32]:


print(red_df.head())


# In[9]:


subs = ['conspiracy', 'conspiracytheories', 'conspiracyNOPOL']
organizer = ['top', 'controversial', 'hot', 'new']
eda_red = {}
for i in subs:
    for j in organizer:
        new_df = red_df.loc[(red_df['subreddit'] == i) & (red_df['organizer'] == j)]
        eda_red[i + ' ' + j] = [len(new_df), new_df['created'].max(), new_df['created'].min()]
print(eda_red)


# In[12]:


with open('added_tokens_red.json', 'r') as f:
    addtok_red = json.load(f)


# In[15]:


addtkeys = list(addtok_red.keys())
print(len(addtkeys))


# In[34]:


redtext_dict = red_df.to_dict(orient = 'index')
print(redtext_dict[0])


# In[35]:


inputs = []
for i in redtext_dict:
    inputs.append(redtext_dict[i]['text'])


# In[38]:


print(inputs[536])


# In[8]:


with open ('entities_red_lem.json', 'r') as f:
    ent_red = json.load(f)
with open ('entities_you_lem.json', 'r') as f:
    ent_you = json.load(f)
with open ('entities_youtrain_lem.json', 'r') as f:
    ent_youtrain = json.load(f)
with open ('entities_youtest_lem.json', 'r') as f:
    ent_youtest = json.load(f)
with open ('entities_action_red_lem.json', 'r') as f:
    entact_red = json.load(f)
with open ('entities_action_you_lem.json', 'r') as f:
    entact_you = json.load(f)
with open ('entities_action_youtrain_lem.json', 'r') as f:
    entact_youtrain = json.load(f)
with open ('entities_action_youtest_lem.json', 'r') as f:
    entact_youtest = json.load(f)
with open ('triplets_red_lem02.json', 'r') as f:
    trips_red = json.load(f)
with open ('triplets_you_lem02.json', 'r') as f:
    trips_you = json.load(f)
with open ('triplets_youtrain_lem02.json', 'r') as f:
    trips_youtrain = json.load(f)
with open ('triplets_youtest_lem02.json', 'r') as f:
    trips_youtest = json.load(f)


# In[9]:


def getlen (dic, name):
    alle = []
    for text in dic:
        for x in dic[text]:
            if x not in alle:
                alle.append(x)
    print(name, 'total of', len(alle))


# In[10]:


getlen(ent_red, 'entities reddit')
getlen(ent_you, 'entities youtube')
getlen(ent_youtrain, 'entities youtube train')
getlen(ent_youtest, 'entities youtube test')


# In[11]:


getlen(entact_red, 'pairs reddit')
getlen(entact_you, 'pairs youtube')
getlen(entact_youtrain, 'pairs youtube train')
getlen(entact_youtest, 'pairs youtube test')


# In[12]:


getlen(trips_red, 'triplets reddit')
getlen(trips_you, 'triplets youtube')
getlen(trips_youtrain, 'triplets youtube train')
getlen(trips_youtest, 'triplets youtube test')

