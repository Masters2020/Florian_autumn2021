#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import os
import pandas as pd
import numpy as np


# In[2]:


path = os.path.abspath('10_cleaning jsons.ipynb')  #this saves the path to this notebook and uses it as a base to access the required csvs
dname = os.path.dirname(path)
os.chdir(dname)


# ## **Entity-Action dictionary - YouTube**

# In[3]:


with open ('entity_action_youbase_01.json', 'r') as f:
    ent_act1 = json.load(f)
with open ('entity_action_youbase_02.json', 'r') as f:
    ent_act2 = json.load(f)
with open ('entity_action_youbase_03.json', 'r') as f:
    ent_act3 = json.load(f)
with open ('entity_action_youbase_04.json', 'r') as f:
    ent_act4 = json.load(f)
with open ('entity_action_youbase_05.json', 'r') as f:
    ent_act5 = json.load(f)
with open ('entity_action_youbase_06.json', 'r') as f:
    ent_act6 = json.load(f)
with open ('entity_action_youbase_07.json', 'r') as f:
    ent_act7 = json.load(f)
with open ('entity_action_youbase_08.json', 'r') as f:
    ent_act8 = json.load(f)
with open ('entity_action_youbase_09.json', 'r') as f:
    ent_act9 = json.load(f)


# In[4]:


l1 = len(ent_act1)
l2 = len(ent_act2)
l3 = len(ent_act3)
l4 = len(ent_act4)
l5 = len(ent_act5)
l6 = len(ent_act6)
l7 = len(ent_act7)
l8 = len(ent_act8)
l9 = len(ent_act9)
lengths = [l1,l2,l3,l4,l5,l6,l7,l8,l9]
print(lengths)
print(sum(lengths))


# In[5]:


for i in ent_act2:
    ent_act1[i] = ent_act2[i]
for i in ent_act3:
    ent_act1[i] = ent_act3[i]
for i in ent_act4:
    ent_act1[i] = ent_act4[i]
for i in ent_act5:
    ent_act1[i] = ent_act5[i]
for i in ent_act6:
    ent_act1[i] = ent_act6[i]
for i in ent_act7:
    ent_act1[i] = ent_act7[i]
for i in ent_act8:
    ent_act1[i] = ent_act8[i]
for i in ent_act9:
    ent_act1[i] = ent_act9[i]
print(len(ent_act1))


# In[7]:


with open ('entities_action_youbase.json', 'w') as f:
    json.dump(ent_act1, f)


# ## **Entity-Action dictionary - Reddit**

# In[4]:


with open ('entity_action_red_01.json', 'r') as f:
    ent_act1 = json.load(f)
with open ('entity_action_red_02.json', 'r') as f:
    ent_act2 = json.load(f)
with open ('entity_action_red_03.json', 'r') as f:
    ent_act3 = json.load(f)
with open ('entity_action_red_04.json', 'r') as f:
    ent_act4 = json.load(f)
with open ('entity_action_red_05.json', 'r') as f:
    ent_act5 = json.load(f)
with open ('entity_action_red_06.json', 'r') as f:
    ent_act6 = json.load(f)
with open ('entity_action_red_07.json', 'r') as f:
    ent_act7 = json.load(f)
with open ('entity_action_red_08.json', 'r') as f:
    ent_act8 = json.load(f)


# In[6]:


l1 = len(ent_act1)
l2 = len(ent_act2)
l3 = len(ent_act3)
l4 = len(ent_act4)
l5 = len(ent_act5)
l6 = len(ent_act6)
l7 = len(ent_act7)
l8 = len(ent_act8)
lengths = [l1,l2,l3,l4,l5,l6,l7,l8]
print(lengths)
print(sum(lengths))


# In[8]:


for i in ent_act2:
    ent_act1[i] = ent_act2[i]
for i in ent_act3:
    ent_act1[i] = ent_act3[i]
for i in ent_act4:
    ent_act1[i] = ent_act4[i]
for i in ent_act5:
    ent_act1[i] = ent_act5[i]
for i in ent_act6:
    ent_act1[i] = ent_act6[i]
for i in ent_act7:
    ent_act1[i] = ent_act7[i]
for i in ent_act8:
    ent_act1[i] = ent_act8[i]
print(len(ent_act1))


# In[79]:


with open ('entities_action_red1.json', 'w') as f:
    json.dump(ent_act1, f)


# ### **Embedding index entity errors**

# In[10]:


with open ('error_emb_ind_ent_red_01.json', 'r') as f:
    ember1 = json.load(f)
with open ('error_emb_ind_ent_red_02.json', 'r') as f:
    ember2 = json.load(f)
with open ('error_emb_ind_ent_red_03.json', 'r') as f:
    ember3 = json.load(f)
with open ('error_emb_ind_ent_red_04.json', 'r') as f:
    ember4 = json.load(f)


# In[11]:


l1 = len(ember1)
l2 = len(ember2)
l3 = len(ember3)
l4 = len(ember4)
lengths = [l1,l2,l3,l4]
print(lengths)
print(sum(lengths))


# In[12]:


for i in ember2:
    ember1[i] = ember2[i]
for i in ember3:
    ember1[i] = ember3[i]
for i in ember4:
    ember1[i] = ember4[i]
print(len(ember1))


# In[80]:


with open ('error_pairs_entity.json', 'w') as f:
    json.dump(ember1, f)


# ### **Embedding index action errors**

# In[29]:


with open ('error_ind_act_red_01.json', 'r') as f:
    errind1 = json.load(f)


# In[30]:


l1 = 0
for i in errind1:
    l1 += len(errind1[i]['text'])
print(l1)


# In[31]:


e2 = 'error_ind_act_red_02.json'
e3 = 'error_ind_act_red_03.json'
e4 = 'error_ind_act_red_04.json'
e5 = 'error_ind_act_red_05.json'
e6 = 'error_ind_act_red_06.json'
e7 = 'error_ind_act_red_07.json'
lengths = [l1]
for a in [e2,e3,e4,e5,e6,e7]:
    with open (a, 'r') as f:
        errind = json.load(f)
    l = 0
    for i in errind:
        l += len(errind[i]['text'])
    lengths.append(l)
    for i in errind:
        if i not in errind1:
            errind1[i] = errind[i]
        elif i in errind1:
            for j in errind[i]['text']:
                errind1[i]['text'].append(j)
            for k in errind[i]['pos']:
                errind1[i]['pos'].append(k)
print(lengths)
print(sum(lengths))


# In[81]:


with open ('errors_pairs_actions.json', 'w') as f:
    json.dump(errind1, f)


# ### **Analysis errors**

# In[35]:


red_text = dname + '\\01_data\\red_text.csv' #exported as one from the notebook preprocessing


# In[36]:


redtext_df = pd.read_csv(red_text, skipinitialspace=True)


# In[37]:


redtext_dict = redtext_df.to_dict(orient = 'index')
print(redtext_dict[0])


# In[38]:


inputs_red = []
for i in redtext_dict:
    inputs_red.append(redtext_dict[i]['text'])


# In[46]:


data = np.load('E:\\embeddings_red01.npz')


# In[47]:


ent_emb_k = list(ember1.keys())


# In[48]:


print(ent_emb_k)


# In[49]:


print(ember1)


# In[50]:


texts = []
indeces = []
for i in ember1:
    for j in ember1[i]['text']:
        texts.append(int(j))
    for k in ember1[i]['ind']:
        indeces.append(int(k))
ts = set(texts)
u_texts = list(ts)
is_ = set(indeces)
u_indeces = list(is_)


# In[51]:


print(u_texts)


# In[52]:


print(u_indeces)


# In[63]:


texts1 = []
indeces1 = []
for i in errind1:
    for j in errind1[i]['text']:
        texts1.append(int(j))
    indeces1.append(errind1[i]['pos'])
ts = set(texts1)
u_texts1 = list(ts)


# In[64]:


u_texts1.sort()
print(u_texts1)


# In[65]:


u_texts.sort()
print(u_texts)


# In[66]:


print(indeces1)


# In[70]:


print(u_texts1.index(7401))


# In[71]:


print(u_texts1[35])
print(indeces1[35])


# In[90]:


print(data['text2148'].shape)


# ## **Fixing the embedding errors**

# ### **Create json with previous error entities**

# In[67]:


with open ('entity_verbs_red.json', 'r') as f:
    entities_red = json.load(f)


# In[68]:


new_entity = {}
for ent in entities_red:
    keys = list(entities_red[ent].keys())
    for k in keys:
        if int(k) in u_texts1:
            new_entity[ent] = {}
            new_entity[ent][k] = entities_red[ent][k]


# In[69]:


print(new_entity)


# In[74]:


import os
import pandas as pd
import torch
import scipy
import scipy.spatial
import numpy as np
import tensorflow as tf
import keras
import transformers
import time
import json
import spacy
import copy


# In[77]:


def getactions (entities, embeddings_file, outfile, outfile2, outfile3):
    data = np.load(embeddings_file)
    entity_action = {}
    error = {}
    error_index = {}
    error_emb_ind = {}
    for ent in entities:
        start_ent = time.time()
        for tex in entities[ent]:
            start_tex = time.time()
            ra = [*range(int(len(entities[ent][tex])/2))]
            for r in ra:
                try:
                    pos = entities[ent][tex]['pos'+str(r)]
                    emb_ent = data['text'+str(tex)][pos] 

                    for act in entities[ent][tex]['act'+str(r)]:
                        if act != ent:
                            try:
                                for ind in range(len(entities[ent][tex]['act'+str(r)][act])):
                                    if len(entities[ent][tex]['act'+str(r)][act][ind]) == 1: #the action only appears once in the text so just need to do one check with one embedding
                                        emb_act = data['text' + str(tex)][entities[ent][tex]['act'+str(r)][act][ind][0]]
                                        sim = 1 - scipy.spatial.distance.cosine(emb_ent, emb_act)
                                        if sim >= 0.65:
                                            ent_act = ent + '-' + act
                                            if ent_act in entity_action:
                                                entity_action[ent_act]['text'].append(tex)
                                                entity_action[ent_act]['pos-ent'].append([pos])
                                                entity_action[ent_act]['pos-act'].append(entities[ent][tex]['act'+str(r)][act][ind])
                                                entity_action[ent_act]['sim'].append(sim)
                                            elif ent_act not in entity_action:
                                                entity_action[ent_act] = {}
                                                entity_action[ent_act]['text'] = [tex]
                                                entity_action[ent_act]['pos-ent'] = [[pos]]
                                                entity_action[ent_act]['pos-act'] = [entities[ent][tex]['act'+str(r)][act][ind]]
                                                entity_action[ent_act]['sim'] = [sim]

                                            with open(outfile, 'w') as file:
                                                json.dump(entity_action, file)

                                    elif len(entities[ent][tex]['act'+str(r)][act][ind]) > 1: #action appears more than once in the text, either because it was split into subwords or the word as a whole just appears multiple times
                                        i_list = []
                                        for i, v in enumerate (entities[ent][tex]['act'+str(r)][act][ind]):
                                            if i == 0:
                                                start = np.expand_dims(data['text' + str(tex)][v], axis=0) #build the array to be averaged with the first subword token
                                                i_list.append(v)
                                            else:
                                                start = np.concatenate((start, np.expand_dims(data['text' + str(tex)][v], axis = 0))) #add the other subword tokens
                                                i_list.append(v)
                                            emb_act = np.average(start, axis = 0) #average the final array and use as embedding for action
                                            sim = 1 - scipy.spatial.distance.cosine(emb_ent, emb_act)
                                            if sim >= 0.65:
                                                ent_act = ent + '-' + act
                                                if ent_act in entity_action:
                                                    entity_action[ent_act]['text'].append(tex)
                                                    entity_action[ent_act]['pos-ent'].append([pos])
                                                    entity_action[ent_act]['pos-act'].append([i_list])
                                                    entity_action[ent_act]['sim'].append(sim)
                                                elif ent_act not in entity_action:
                                                    entity_action[ent_act] = {}
                                                    entity_action[ent_act]['text'] = [tex]
                                                    entity_action[ent_act]['pos-ent'] = [[pos]]
                                                    entity_action[ent_act]['pos-act'] = [[i_list]]
                                                    entity_action[ent_act]['sim'] = [sim]

                                                with open(outfile, 'w') as file:
                                                    json.dump(entity_action, file)
                                                        

                            except IndexError:
                                if act in error_index:
                                    error_index[act]['text'].append(tex)
                                    error_index[act]['pos'].append(entities[ent][tex]['act'+str(r)])
                                    with open(outfile2, 'w') as file:
                                        json.dump(error_index, file)
                                elif act not in error_index:
                                    error_index[act] = {}
                                    error_index[act]['text'] = [tex]
                                    error_index[act]['pos'] = [entities[ent][tex]['act'+str(r)]]
                                    with open(outfile2, 'w') as file:
                                        json.dump(error_index, file)
                                pass

                except IndexError:
                    if ent in error_emb_ind:
                        error_emb_ind[ent]['text'].append(tex)
                        error_emb_ind[ent]['ind'].append(pos)
                        with open(outfile3, 'w') as file:
                            json.dump(error_emb_ind, file)
                    if ent not in error_emb_ind:
                        error_emb_ind[ent] = {}
                        error_emb_ind[ent]['text'] = [tex]
                        error_emb_ind[ent]['ind'] = [pos]
                        with open(outfile3, 'w') as file:
                            json.dump(error_emb_ind, file)
                    pass
                    
            print('done with text', tex, 'in:', time.time() - start_tex, 'seconds')
        print('done with entity', ent, 'in:', time.time() - start_ent, 'seconds')
                    
    return entity_action, error_index, error_emb_ind


# In[91]:


entity_action_red, error_ind_act_red, error_emb_ind_act_red = getactions (new_entity, 'E:\\embeddings_red03.npz', 'entity_action_red10_new.json', 'error_ind_act_red10_new.json', 'error_emb_ind_ent_red10_new.json')


# ### **Create combined json**

# In[4]:


with open ('entities_action_red1.json', 'r') as f:
    ent_act = json.load(f)


# In[5]:


with open ('entity_action_red10_new.json', 'r') as f:
    entity_action_red = json.load(f)


# In[6]:


print(len(ent_act))


# In[7]:


print(len(entity_action_red))


# In[8]:


tot = 0
for i in entity_action_red:
    if i in ent_act:
        print(i, 'is included')
        tot += 1
print(tot)


# In[11]:


del ent_act['god-exists']


# In[13]:


for i in entity_action_red:
    if i in ent_act:
        for index in range(len(entity_action_red[i]['text'])):
            ent_act[i]['text'].append(entity_action_red[i]['text'][index])
            ent_act[i]['pos-ent'].append(entity_action_red[i]['pos-ent'][index])
            ent_act[i]['pos-act'].append(entity_action_red[i]['pos-act'][index])
            ent_act[i]['sim'].append(entity_action_red[i]['sim'][index])
    if i not in ent_act:
        ent_act[i] = entity_action_red[i]


# In[14]:


print(len(ent_act))


# In[16]:


with open ('entities_action_red.json', 'w') as f:
    json.dump(ent_act, f)


# ## **Triplets dictionary - Reddit**

# In[10]:


with open ('triplets_red_01.json', 'r') as f:
    trip1 = json.load(f)
with open ('triplets_red_02.json', 'r') as f:
    trip2 = json.load(f)
with open ('triplets_red_03.json', 'r') as f:
    trip3 = json.load(f)


# In[11]:


l1 = len(trip1)
l2 = len(trip2)
l3 = len(trip3)
lengths = [l1,l2,l3]
print(lengths)
print(sum(lengths))


# In[12]:


for i in trip2:
    trip1[i] = trip2[i]
for i in trip3:
    trip1[i] = trip3[i]
print(len(trip1))


# In[13]:


with open ('triplets_red.json', 'w') as f:
    json.dump(trip1, f)


# In[14]:


keys = trip1.keys()
with open('triplets_red.txt', 'w', encoding="utf-8") as f:
    for i in keys:
        f.write(i + '\n')


# ### **Filter triplets by ner as targets**

# In[15]:


with open (dname + '\\02_reddit_transfer\\entities_red.json', 'r') as f:
    ent = json.load(f)


# In[16]:


triplets_ner = {}
for i in keys:
    check = i.split('-')
    if check[2] in ent:
        triplets_ner[i] = trip1[i]


# In[17]:


print(triplets_ner)


# In[18]:


keys = triplets_ner.keys()
with open('triplets_ner_red_lookup.txt', 'w', encoding="utf-8") as f:
    for i in keys:
        f.write(i + '\n')


# ## **Triplets dictionary - Youtube**

# In[19]:


with open ('triplets_you_01.json', 'r') as f:
    trip1 = json.load(f)
with open ('triplets_you_02.json', 'r') as f:
    trip2 = json.load(f)


# In[20]:


l1 = len(trip1)
l2 = len(trip2)
lengths = [l1,l2]
print(lengths)
print(sum(lengths))


# In[21]:


for i in trip2:
    trip1[i] = trip2[i]
print(len(trip1))


# In[22]:


with open ('triplets_you.json', 'w') as f:
    json.dump(trip1, f)


# In[23]:


keys = trip1.keys()
with open('triplets_you.txt', 'w', encoding="utf-8") as f:
    for i in keys:
        f.write(i + '\n')


# ### **Filter triplets by ner as targets**

# In[24]:


with open (dname + '\\03_youtube_baseline\\entities_youtrain.json', 'r') as f:
    ent = json.load(f)


# In[25]:


triplets_ner = {}
for i in keys:
    check = i.split('-')
    if check[2] in ent:
        triplets_ner[i] = trip1[i]


# In[26]:


print(triplets_ner)


# In[27]:


keys = triplets_ner.keys()
with open('triplets_you_ner.txt', 'w', encoding="utf-8") as f:
    for i in keys:
        f.write(i + '\n')

