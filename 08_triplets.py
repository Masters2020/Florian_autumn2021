#!/usr/bin/env python
# coding: utf-8

# ## **Set-Up**

# In[1]:


import os
import pandas as pd
import scipy
import scipy.spatial
import numpy as np
import json
import spacy
import copy
import time


# In[2]:


path = os.path.abspath('08_triplets.ipynb')  #this saves the path to this notebook and uses it as a base to access the required csvs
dname = os.path.dirname(path)
os.chdir(dname)


# ## **Import texts**

# ### **Reddit**

# In[5]:


from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('flboehm/reddit-bert-text_10', model_max_length = 512, padding_side = 'right')


# In[6]:


print(len(tokenizer))


# In[13]:


red_text = dname + '\\01_data\\red_text.csv' #exported as one from the notebook preprocessing


# In[14]:


redtext_df = pd.read_csv(red_text, skipinitialspace=True)


# In[15]:


redtext_dict = redtext_df.to_dict(orient = 'index')
print(redtext_dict[0])


# In[16]:


inputs = []
for i in redtext_dict:
    inputs.append(redtext_dict[i]['text'])


# ### **Youtube**

# In[16]:


from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('flboehm/youtube-bert', model_max_length = 512, padding_side = 'right')


# In[17]:


print(len(tokenizer))


# In[6]:


you_text = dname + '\\01_data\\you_text_label.csv' #exported as one from the notebook preprocessing


# In[7]:


youtext_df = pd.read_csv(you_text, skipinitialspace=True)


# In[8]:


youtext_dict = youtext_df.to_dict(orient = 'index')
print(youtext_dict[0])


# In[9]:


inputs_you = []
for i in youtext_dict:
    inputs_you.append(youtext_dict[i]['text'])


# In[10]:


print(inputs_you[600])


# ## **Import dictionaries**

# ### **Reddit**

# In[3]:


with open('entity_verbs_red.json', 'r') as f:
    ent_verbs_red = json.load(f)
with open('act_nouns_red.json', 'r') as f:
    act_noun_red = json.load(f)
with open('entities_action_red.json', 'r') as f:
    ent_act_red = json.load(f)


# ### **Youtube**

# In[14]:


with open('entity_verbs_youtrain.json', 'r') as f:
    ent_verbs_you = json.load(f)
with open('act_nouns_youtrain.json', 'r') as f:
    act_noun_you = json.load(f)
with open('entities_action_youtrain.json', 'r') as f:
    ent_act_you = json.load(f)


# ## **Extracting triplets**

# ### **Enrich by actions**

# In[16]:


keys = list(ent_verbs_red.keys())


# In[18]:


entkeys = keys[0:150]


# In[34]:


print(entkeys)


# In[20]:


def getactions (entities, entities_keys, embeddings_file, outfile, outfile2, outfile3):
    data = np.load(embeddings_file)
    entity_action = {}
    error = {}
    error_index = {}
    error_emb_ind = {}
    for ent in entities_keys:
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
                                    if ent_verbs[ent][tex]['pos'+str(r)] != ent_verbs[ent][tex]['act'+str(r)][act][ind][0]:
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


# In[21]:


entity_action, error_ind_act, error_emb_ind_act = getactions (ent_verbs, entkeys, 'E:\\embeddings_red01.npz', 'entity_action_red_08.json', 'error_ind_act_red_08.json', 'error_emb_ind_ent_red_08.json')


# ### **Enrich by targets**

# In[5]:


keys = list(act_noun_red.keys())


# In[6]:


len(keys)


# In[7]:


pairkeys = keys[0:401]


# In[8]:


print(pairkeys)


# In[9]:


def gettargets (pair_noun, pair_keys, ent_act, embeddings_file, outfile, outfile2, outfile3):
    data = np.load(embeddings_file)
    triplets = {}
    error = {}
    error_index = {}
    error_emb_ind = {}
    for pair in pair_keys:
        check = pair.split('-')
        start_pair = time.time()
        for tex in pair_noun[pair]:
            start_tex = time.time()
            ra = [*range(int(len(pair_noun[pair][tex])/2))]
            for r in ra:
                try:
                    pos = pair_noun[pair][tex]['pos'+str(r)][0]
                    emb_act = data['text'+str(tex)][pos] 

                    for target in pair_noun[pair][tex]['tar'+str(r)]:
                        if target != check[0] and target != check[1]:
                            try:
                                for ind in range(len(pair_noun[pair][tex]['tar'+str(r)][target])):
                                    if len(pair_noun[pair][tex]['tar'+str(r)][target][ind]) == 1: #the action only appears once in the text so just need to do one check with one embedding
                                        emb_tar = data['text' + str(tex)][pair_noun[pair][tex]['tar'+str(r)][target][ind][0]]
                                        sim = 1 - scipy.spatial.distance.cosine(emb_act, emb_tar)
                                        if sim >= 0.65:
                                            trip = pair + '-' + target
                                            if trip in triplets:
                                                triplets[trip]['text'].append(tex)
                                                triplets[trip]['pos-ent'].append(ent_act[pair]['pos-ent'][ind])
                                                triplets[trip]['pos-act'].append(ent_act[pair]['pos-act'][ind])
                                                triplets[trip]['pos-tar'].append(pair_noun[pair][tex]['tar'+str(r)][target][ind])
                                                triplets[trip]['sim-ent-act'].append(round(ent_act[pair]['sim'][ind],3))
                                                triplets[trip]['sim-act-tar'] = round(sim,3)
                                            elif trip not in triplets:
                                                triplets[trip] = {}
                                                triplets[trip]['text'] = [tex]
                                                triplets[trip]['pos-ent'] = [ent_act[pair]['pos-ent'][ind]]
                                                triplets[trip]['pos-act'] = [ent_act[pair]['pos-act'][ind]]
                                                triplets[trip]['pos-tar'] = [pair_noun[pair][tex]['tar'+str(r)][target][ind]]
                                                triplets[trip]['sim-ent-act'] = [round(ent_act[pair]['sim'][ind],3)]
                                                triplets[trip]['sim-act-tar'] = [round(sim,3)]

                                            with open(outfile, 'w') as file:
                                                json.dump(triplets, file)

                                    elif len(pair_noun[pair][tex]['tar'+str(r)][target][ind]) > 1: #action appears more than once in the text, either because it was split into subwords or the word as a whole just appears multiple times
                                        i_list = []
                                        for i, v in enumerate (pair_noun[pair][tex]['tar'+str(r)][target][ind]):
                                            if i == 0:
                                                start = np.expand_dims(data['text' + str(tex)][v], axis=0) #build the array to be averaged with the first subword token
                                                i_list.append(v)
                                            else:
                                                start = np.concatenate((start, np.expand_dims(data['text' + str(tex)][v], axis = 0))) #add the other subword tokens
                                                i_list.append(v)
                                            emb_act = np.average(start, axis = 0) #average the final array and use as embedding for action
                                            sim = 1 - scipy.spatial.distance.cosine(emb_act, emb_tar)
                                            if sim >= 0.65:
                                                trip = pair + '-' + target
                                                if trip in triplets:
                                                    triplets[trip]['text'].append(tex)
                                                    triplets[trip]['pos-ent'].append(ent_act[pair]['pos-ent'][ind])
                                                    triplets[trip]['pos-act'].append(ent_act[pair]['pos-act'][ind])
                                                    triplets[trip]['pos-tar'].append([i_list])
                                                    triplets[trip]['sim-ent-act'].append(round(ent_act[pair]['sim'][ind],3))
                                                    triplets[trip]['sim-act-tar'].append(round(sim,3))
                                                elif trip not in triplets:
                                                    triplets[trip] = {}
                                                    triplets[trip]['text'] = [tex]
                                                    triplets[trip]['pos-ent'] = [ent_act[pair]['pos-ent'][ind]]
                                                    triplets[trip]['pos-act'] = [ent_act[pair]['pos-act'][ind]]
                                                    triplets[trip]['pos-tar'] = [[i_list]]
                                                    triplets[trip]['sim-ent-act'] = [round(ent_act[pair]['sim'][ind],3)]
                                                    triplets[trip]['sim-act-tar'] = [round(sim,3)]

                                                with open(outfile, 'w') as file:
                                                    json.dump(triplets, file)
                                                        

                            except IndexError:
                                if noun in error_index_noun:
                                    error_index[noun]['text'].append(tex)
                                    error_index[noun]['pos'].append(pair_noun[pair][tex]['tar'+str(r)][target][ind])
                                    with open(outfile2, 'w') as file:
                                        json.dump(error_index_noun, file)
                                elif act not in error_index_noun:
                                    error_index[noun] = {}
                                    error_index[noun]['text'] = [tex]
                                    error_index[noun]['pos'] = [pair_noun[pair][tex]['tar'+str(r)][target][ind]]
                                    with open(outfile2, 'w') as file:
                                        json.dump(error_index_noun, file)
                                pass

                except IndexError:
                    if check[1] in error_index_act:
                        error_index_act[check[1]]['text'].append(tex)
                        error_index_act[check[1]]['pos'].append(pos)
                        with open(outfile3, 'w') as file:
                            json.dump(error_index_act, file)
                    if check[1] not in error_index_act:
                        error_index_act[check[1]] = {}
                        error_index_act[check[1]]['text'] = [tex]
                        error_index_act[check[1]]['pos'] = [pos]
                        with open(outfile3, 'w') as file:
                            json.dump(error_index_act, file)
                    pass
                    
            print('done with text', tex, 'in:', time.time() - start_tex, 'seconds')
        print('done with pair', pair, 'in:', time.time() - start_pair, 'seconds')
                    
    return entity_action, error_index_noun, error_index_act


# In[10]:


triplets, error_trip_noun, error_trip_act = gettargets (act_noun_red, pairkeys, ent_act_red, 'E:\\embeddings_red01.npz', 'triplets_red_01.json', 'error_trip_noun_red_01.json', 'error_trip_act_red_01.json')

