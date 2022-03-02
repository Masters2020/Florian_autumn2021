#!/usr/bin/env python
# coding: utf-8

# ## **Set-Up**

# In[1]:


import spacy
print(spacy.__version__)
import en_core_web_trf
import spacy_transformers
import pandas as pd
import os
from spacy import displacy
import re
import json


# In[2]:


path = os.path.abspath('04_ner.ipynb')  #this saves the path to this notebook and uses it as a base to access the required csvs
dname = os.path.dirname(path)
os.chdir(dname)


# ## **Import text**

# ### **Reddit**

# In[3]:


red_text = dname + '\\data\\red_text.csv' #exported as one from the notebook preprocessing


# In[4]:


redtext_df = pd.read_csv(red_text, skipinitialspace=True)


# In[5]:


redtext_dict = redtext_df.to_dict(orient = 'index')
print(redtext_dict[0])


# In[6]:


inputs = []
for i in redtext_dict:
    inputs.append(redtext_dict[i]['text'])


# ### **Youtube**

# In[3]:


you_text = dname + '\\data\\you_text.csv'


# In[4]:


youtext_df = pd.read_csv(you_text, skipinitialspace=True)


# In[5]:


youtext_dict = youtext_df.to_dict(orient = 'index')
print(youtext_dict[0])


# In[6]:


inputs_you = []
for i in youtext_dict:
    inputs_you.append(youtext_dict[i]['text'])


# ## **Entity Recognition**

# In[8]:


def getentities (input_list, threshold, filename, filename2, write_to_file = 'yes'):
    nlp = spacy.load("en_core_web_trf")
    ent_dic_all = {}
    fin = {}
    for i in range(len(input_list)):
        doc = nlp(input_list[i])
        ent_list = []
        for ent in doc.ents:
            if ent.label_ != 'CARDINAL' and ent.label_ != 'LANGUAGE' and ent.label_ != 'DATE' and ent.label_ != 'TIME' and ent.label_ != 'PERCENT' and ent.label_ != 'MONEY' and ent.label_ != 'QUANTITY' and ent.label_ != 'ORDINAL':
                ent_list.append(ent.text)
            ent_list = [string.lower() for string in ent_list] #necessary for comparison with bert tokenized text that was feed to model
        ent_dic_all[i] = ent_list
        
    ent_list_all = []
    for j in ent_dic_all:
        for k in ent_dic_all[j]:
            ent_list_all.append(k)
            
    freq_dic = {}
    for l in ent_list_all:
        if l[-2:] == "'s" or l[-2:] == "s'" or l[-2:] == "’s" or l[-2:] == "s’":
            m = l[:-2] 
            if m not in freq_dic:
                freq_dic[m] = ent_list_all.count(m)
            else:
                pass
        else:
            if l not in freq_dic:
                freq_dic[l] = ent_list_all.count(l)
            else:
                pass
            
    list_freq_all = []
    for n in freq_dic:
        if freq_dic[n] >= threshold and len(n) > 1:
            list_freq_all.append(n)
            
    for o in list_freq_all:
        fin[o] = []
        for p in ent_dic_all:
            if o in ent_dic_all[p] and p not in fin[o]:
                fin[o].append(p)
    
    print('The total amount of entites is', len(list_freq_all))
    
    if write_to_file == 'yes':
        with open(filename, 'w', encoding="utf-8") as f:
            for q in list_freq_all:
                f.write(q + '\n')
        with open(filename2, 'w') as f:
            json.dump(fin, f)
            
    return list_freq_all, fin


# ### **Reddit**

# In[8]:


entities, ent_dic = getentities(inputs, 0, 'NER_vocab_0.txt', 'ent_dic_0.json', write_to_file = 'yes')


# In[9]:


entities, ent_dic = getentities(inputs, 5, 'NER_vocab_5.txt', 'ent_dic_5.json', write_to_file = 'yes')


# In[10]:


entities, ent_dic = getentities(inputs, 10, 'NER_vocab_10.txt', 'ent_dic_10.json', write_to_file = 'yes')


# In[11]:


entities, ent_dic = getentities(inputs, 20, 'NER_vocab_20.txt', 'ent_dic_20.json', write_to_file = 'yes')


# ### **Youtube**

# In[11]:


entities0, ent_dic0 = getentities(inputs_you, 0, 'you_NER_vocab_0.txt', 'you_ent_dic_0.json', write_to_file = 'yes')


# In[12]:


entities5, ent_dic5 = getentities(inputs_you, 5, 'you_NER_vocab_5.txt', 'you_ent_dic_5.json', write_to_file = 'yes')


# In[13]:


entities10, ent_dic10 = getentities(inputs_you, 10, 'you_NER_vocab_10.txt', 'you_ent_dic_10.json', write_to_file = 'yes')


# In[14]:


entities20, ent_dic20 = getentities(inputs_you, 20, 'you_NER_vocab_20.txt', 'you_ent_dic_20.json', write_to_file = 'yes')


# ### **Processing Reddit**

# In[7]:


#continue here
with open('NER_vocab_0.txt', 'r', encoding = 'utf-8') as file1:
    new_tokens = file1.readlines()
clean_tokens_0 = []
for i in new_tokens:
    j = i.replace('\n', '')
    clean_tokens_0.append(j)
with open('NER_vocab_5.txt', 'r', encoding = 'utf-8') as file1:
    new_tokens = file1.readlines()
clean_tokens_5 = []
for i in new_tokens:
    j = i.replace('\n', '')
    clean_tokens_5.append(j)
with open('NER_vocab_10.txt', 'r', encoding = 'utf-8') as file1:
    new_tokens = file1.readlines()
clean_tokens_10 = []
for i in new_tokens:
    j = i.replace('\n', '')
    clean_tokens_10.append(j)
with open('NER_vocab_20.txt', 'r', encoding = 'utf-8') as file1:
    new_tokens = file1.readlines()
clean_tokens_20 = []
for i in new_tokens:
    j = i.replace('\n', '')
    clean_tokens_20.append(j)


# In[9]:


entities_5 = []
for i in clean_tokens_5:  
    if i != 'r/' and i != '/r' and not i.startswith('/r/') and not i.startswith('r/') and not i.startswith('/r'):
        entities_5.append(i)
entities_10 = []
for i in clean_tokens_10:  
    if i != 'r/' and i != '/r' and not i.startswith('/r/') and not i.startswith('r/') and not i.startswith('/r'):
        entities_10.append(i)
entities_20 = []
for i in clean_tokens_20:  
    if i != 'r/' and i != '/r' and not i.startswith('/r/') and not i.startswith('r/') and not i.startswith('/r'):
        entities_20.append(i)


# In[10]:


print(len(entities_5))
print(len(entities_10))
print(len(entities_20))


# In[11]:


with open('NER_vocab_5.txt', 'w', encoding="utf-8") as f:
    for q in entities_5:
        f.write(q + '\n')
with open('NER_vocab_10.txt', 'w', encoding="utf-8") as f:
    for q in entities_10:
        f.write(q + '\n')
with open('NER_vocab_20.txt', 'w', encoding="utf-8") as f:
    for q in entities_20:
        f.write(q + '\n')


# In[12]:


with open ('ent_dic_5.json', 'r') as f:
    ent_dic5 = json.load(f)
with open ('ent_dic_10.json', 'r') as f:
    ent_dic10 = json.load(f)
with open ('ent_dic_20.json', 'r') as f:
    ent_dic20 = json.load(f)


# In[15]:


text_5 = []
for i in ent_dic5:
    for j in ent_dic5[i]:
        text_5.append(j)
text_5set = set(text_5)
text5u = list(text_5set)
text_10 = []
for i in ent_dic10:
    for j in ent_dic10[i]:
        text_10.append(j)
text_10set = set(text_10)
text10u = list(text_10set)
text_20 = []
for i in ent_dic20:
    for j in ent_dic20[i]:
        text_20.append(j)
text_20set = set(text_20)
text20u = list(text_20set)


# In[16]:


print(len(text5u))
print(len(text10u))
print(len(text20u))


# ### **Processing Youtube**

# In[7]:


with open ('you_ent_dic_0.json', 'r') as f:
    ent_dic0 = json.load(f)
with open ('you_ent_dic_5.json', 'r') as f:
    ent_dic5 = json.load(f)
with open ('you_ent_dic_10.json', 'r') as f:
    ent_dic10 = json.load(f)
with open ('you_ent_dic_20.json', 'r') as f:
    ent_dic20 = json.load(f)


# In[10]:


print(ent_dic0['22'])


# In[14]:


keys = list(ent_dic0.keys())
print(keys)
print(int(keys[0]))


# In[18]:


keys = list(ent_dic0.keys())
to_del0 = []
for key in keys:
    try:
        to_del0.append(int(key))
    except ValueError:
        pass
keys = list(ent_dic5.keys())
to_del5 = []
for key in keys:
    try:
        to_del5.append(int(key))
    except ValueError:
        pass
keys = list(ent_dic10.keys())
to_del10 = []
for key in keys:
    try:
        to_del10.append(int(key))
    except ValueError:
        pass
keys = list(ent_dic20.keys())
to_del20 = []
for key in keys:
    try:
        to_del20.append(int(key))
    except ValueError:
        pass


# In[22]:


print(to_del0)


# In[26]:


for key in to_del10:
    if str(key) in ent_dic10:
        del ent_dic10[str(key)]
        print('deleted key:', key)


# In[29]:


print(len(ent_dic0))
print(len(ent_dic5))
print(len(ent_dic10))
print(len(ent_dic20))


# In[28]:


with open ('you_ent_dic_0.json', 'w') as f:
    json.dump(ent_dic0, f)
with open ('you_ent_dic_5.json', 'w') as f:
    json.dump(ent_dic5, f)
with open ('you_ent_dic_10.json', 'w') as f:
    json.dump(ent_dic10, f)


# In[8]:


text0 = []
text5 = []
text10 = []
text20 = []


# In[30]:


for ent in ent_dic0:
    for text in ent_dic0[ent]:
        text0.append(text)
s = set(text0)
text0 = list(s)

for ent in ent_dic5:
    for text in ent_dic5[ent]:
        text5.append(text)
s = set(text5)
text5 = list(s)

for ent in ent_dic10:
    for text in ent_dic10[ent]:
        text10.append(text)
s = set(text10)
text10 = list(s)

for ent in ent_dic20:
    for text in ent_dic20[ent]:
        text20.append(text)
s = set(text20)
text20 = list(s)


# In[49]:


text_all = list(range(0,602))
print(text_all)


# In[50]:


not0 = []
not5 = []
not10 = []
not20 = []


# In[51]:


for text in text_all:
    if text not in text0:
        not0.append(text)
for text in text_all:
    if text not in text5:
        not5.append(text)
for text in text_all:
    if text not in text10:
        not10.append(text)
for text in text_all:
    if text not in text20:
        not20.append(text)


# In[52]:


print(len(not0))
print(len(not5))
print(len(not10))
print(len(not20))


# In[53]:


print(not10)


# In[48]:


print(len(inputs_you))

