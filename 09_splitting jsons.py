#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import scipy
import scipy.spatial
import numpy as np
import json
import spacy
import copy


# In[2]:


path = os.path.abspath('09_splitting jsons.ipynb')  #this saves the path to this notebook and uses it as a base to access the required csvs
dname = os.path.dirname(path)
os.chdir(dname)


# In[3]:


with open('entities_youbase.json', 'r') as f:
    entities = json.load(f)
#with open ('entities_action_all5.json', 'r') as f:
    #ent_act = json.load(f)
with open ('wordkeys_verbs_youbase.json', 'r') as f:
    wordkeys_verbs = json.load(f)
with open ('wordkeys_nouns_youbase.json', 'r') as f:
    wordkeys_nouns = json.load(f)


# In[7]:


#this code requires the csvs to be placed in a subfolder called data within the folder in which this notebook is placed
train = dname + '\\data\\train_you_text.csv'
val = dname + '\\data\\val_you_text.csv'


# In[8]:


you_df = pd.read_csv(train)


# In[9]:


you_v = pd.read_csv(val)


# In[10]:


print(you_df)


# In[11]:


ind = list(you_df['index'])


# In[30]:


print(ind)


# In[13]:


ind_v = list(you_v['index'])


# In[31]:


print(ind_v)


# In[29]:


ind_v.sort()


# In[15]:


ent_new = {}
for ent in entities:
    ent_new[ent] = {}
    for text in entities[ent]:
        if int(text) in ind:
            ent_new[ent][text] = entities[ent][text]
            


# In[16]:


ent_val = {}
for ent in entities:
    ent_val[ent] = {}
    for text in entities[ent]:
        if int(text) in ind_v:
            ent_val[ent][text] = entities[ent][text]
            


# In[17]:


for ent in ent_new:
    print('new:', len(ent_new[ent]))
    print('old:', len(entities[ent]))


# In[18]:


print(ent_val)


# In[19]:


ent_new_new = {}
for ent in ent_new:
    if len(ent_new[ent]) != 0:
        ent_new_new[ent] = ent_new[ent]


# In[20]:


ent_val_val = {}
for ent in ent_val:
    if len(ent_val[ent]) != 0:
        ent_val_val[ent] = ent_val[ent]


# In[21]:


kt = list(ent_new_new.keys())
kv = list(ent_val_val.keys())


# In[22]:


for k in kt:
    if k not in kv:
        print('ent', k, 'in kt but not in kv')


# In[23]:


wordkeysvt = {}
for text in wordkeys_verbs:
    if int(text) in ind:
        wordkeysvt[text] = wordkeys_verbs[text]


# In[24]:


wordkeysnt = {}
for text in wordkeys_nouns:
    if int(text) in ind:
        wordkeysnt[text] = wordkeys_nouns[text]


# In[25]:


wordkeysvv = {}
for text in wordkeys_verbs:
    if int(text) in ind_v:
        wordkeysvv[text] = wordkeys_verbs[text]


# In[26]:


wordkeysnv = {}
for text in wordkeys_nouns:
    if int(text) in ind_v:
        wordkeysnv[text] = wordkeys_nouns[text]


# In[32]:


print(wordkeysvt['2'])


# In[41]:


test = list(wordkeysnv.keys())


# In[42]:


test.sort()
print(len(test))
print(len(ind_v))
print(len(wordkeys_verbs))


# In[45]:


with open ('wordkeys_nouns_youtrain.json', 'w') as f:
    json.dump(wordkeysnt, f)
with open ('wordkeys_nouns_youtest.json', 'w') as f:
    json.dump(wordkeysnv, f)
with open ('wordkeys_verbs_youtrain.json', 'w') as f:
    json.dump(wordkeysvt, f)
with open ('wordkeys_verbs_youtest.json', 'w') as f:
    json.dump(wordkeysvv, f)
with open ('entities_youtrain.json', 'w') as f:
    json.dump(ent_new_new, f)
with open ('entities_youtest.json', 'w') as f:
    json.dump(ent_val_val, f)


# In[ ]:




