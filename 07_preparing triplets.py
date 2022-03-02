#!/usr/bin/env python
# coding: utf-8

# ## **Set-up**

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


path = os.path.abspath('07_preparing_triplets.ipynb')  #this saves the path to this notebook and uses it as a base to access the required csvs
dname = os.path.dirname(path)
os.chdir(dname)


# ## **Combine verb wordkeys and entities positions**

# ### **Reddit**

# In[3]:


with open(dname + '\\02_reddit\\entities_red.json', 'r') as f:
    entities = json.load(f)
with open (dname + '\\02_reddit\\wordkeys_verbs_red.json', 'r') as f:
    wordkeys_verbs = json.load(f)


# ### **Youtube - train**

# In[4]:


with open(dname + '\\03_youtube\\entities_youtrain.json', 'r') as f:
    entities = json.load(f)
with open (dname + '\\03_youtube\\wordkeys_verbs_youtrain.json', 'r') as f:
    wordkeys_verbs = json.load(f)


# ### **Youtube - whole**

# In[5]:


with open(dname + '\\03_youtube\\entities_you.json', 'r') as f:
    entities = json.load(f)
with open (dname + '\\03_youtube\\wordkeys_verbs_you.json', 'r') as f:
    wordkeys_verbs = json.load(f)


# ### **Youtube - test**

# In[152]:


with open(dname + '\\03_youtube\\entities_youtest.json', 'r') as f:
    entities = json.load(f)
with open (dname + '\\03_youtube\\wordkeys_verbs_youtest.json', 'r') as f:
    wordkeys_verbs = json.load(f)


# ### **prepare new ent dic**

# In[153]:


new_ent = {}
for ent in entities:
    new_ent[ent] = {}
    for text in entities[ent]:
        if text in wordkeys_verbs:
            new_ent[ent][text] = {}
            for ind, pos_ent in enumerate(entities[ent][text]):
                new_ent[ent][text]['pos'+str(ind)] = pos_ent


# ### **prepare new wordkey verb dic**

# In[154]:


for i in wordkeys_verbs:
    for act in wordkeys_verbs[i]:
        pos = []
        bol = []
        if len(wordkeys_verbs[i][act]) != 1:
            if wordkeys_verbs[i][act][1] == wordkeys_verbs[i][act][0]+1 and wordkeys_verbs[i][act][-1] == wordkeys_verbs[i][act][-2]+1:
                for j in range(len(wordkeys_verbs[i][act])-1):
                    if wordkeys_verbs[i][act][j]+1 == wordkeys_verbs[i][act][j+1]:
                        bol.append(True)
                    else:
                        bol.append(False)
                if len(bol) != 0:
                    p = []
                    for a in range(len(bol)):
                        if bol[a] == True:
                            if a == len(bol)-1:
                                p.append(wordkeys_verbs[i][act][a])
                                p.append(wordkeys_verbs[i][act][a+1])
                                pos.append(p)
                            else:
                                if bol[a+1] == True:
                                    p.append(wordkeys_verbs[i][act][a])
                                elif bol[a+1] == False:
                                    p.append(wordkeys_verbs[i][act][a])
                                    p.append(wordkeys_verbs[i][act][a+1])
                                    pos.append(p)
                                    p = []
                        elif bol[a] == False:
                            if i == len(bol)-1:
                                pos.append(wordkeys_verbs[i][act][a])
                                pos.append(wordkeys_verbs[i][act][a+1])
                            else:
                                if bol[a-1] == True:
                                    pass
                                else:
                                    pos.append(wordkeys_verbs[i][act][a])
                wordkeys_verbs[i][act] = pos
            else:
                pos2 = []
                for v in wordkeys_verbs[i][act]:
                    pos2.append([v])
                    wordkeys_verbs[i][act] = pos2
        else:
            wordkeys_verbs[i][act] = [wordkeys_verbs[i][act]]


# In[155]:


print(wordkeys_verbs['0'])
print(wordkeys_verbs['8'])


# ### **combine dictionaries - #1**
# 
# This includes the criteria that the action appears maximum ten positions after the entity

# In[160]:


original_wordkeys = dname + '\\03_youtube\\wordkeys_verbs_youtest.json'


# In[161]:


with open (original_wordkeys, 'r') as f:
    wordkeys_verbs_orig = json.load(f)


# In[170]:


new_new = {}
err = {'text': [], 'verb': []}
for ent in new_ent:
    start_ent = time.time()
    new_new[ent] = {}
    for text in new_ent[ent]:
        new_new[ent][text] = {}
        n = 0
        for pos in new_ent[ent][text]:
            new_new[ent][text][pos] = new_ent[ent][text][pos]
            new_new[ent][text]['act'+str(n)] = {}
            posrange = [*range(new_ent[ent][text][pos], new_ent[ent][text][pos]+10)]
            for verb in wordkeys_verbs[text]:
                new_new[ent][text]['act'+str(n)][verb] = []
                for ind in range(len(wordkeys_verbs[text][verb])):
                    try:
                        if wordkeys_verbs[text][verb][ind][0] in posrange:
                            new_new[ent][text]['act'+str(n)][verb].append(wordkeys_verbs[text][verb][ind])
                    except TypeError:
                        if text not in err['text'] and verb not in err['verb']:
                            err['text'].append(text)
                            err['verb'].append(verb)
            n += 1
    print('done with entity', ent, 'in', time.time()-start_ent, 'seconds')


# In[171]:


print(err)


# ### **If errors not empty, use code below and then execute function again**

# In[167]:


lol = []
for ind, text in enumerate(err['text']):
    lol.append(wordkeys_verbs_orig[text][err['verb'][ind]])
print(lol)


# In[168]:


for ind in range(len(lol)):
    new = []
    for i in lol[ind]:
        new.append([i])
    text = err['text'][ind]
    verb = err['verb'][ind]
    wordkeys_verbs[text][verb] = new
    print('changed', text, verb)


# In[169]:


for ind, text in enumerate(err['text']):
    print(wordkeys_verbs[text][err['verb'][ind]])


# ### **clean dictionary**

# In[172]:


ent_dic = {}
for ent in new_new:
    ent_dic[ent] = {}
    for text in new_new[ent]:
        n = 0
        ent_dic[ent][text] = {}
        ra = [*range(int(len(new_new[ent][text])/2))]
        for r in ra:
            ent_dic[ent][text]['pos'+str(r)] = new_new[ent][text]['pos'+str(r)]
            ent_dic[ent][text]['act'+str(r)] = {}
            for act in new_new[ent][text]['act'+str(r)]:
                if len(new_new[ent][text]['act'+str(r)][act]) != 0:
                    ent_dic[ent][text]['act'+str(r)][act] = new_new[ent][text]['act'+str(r)][act]


# In[173]:


print(ent_dic['hong kong'])


# In[174]:


ent_verb = {}
for ent in ent_dic:
    ent_verb[ent] = {}
    for text in ent_dic[ent]:
        ra = [*range(int(len(ent_dic[ent][text])/2))]
        for r in ra:
            if len(ent_dic[ent][text]['act'+str(r)]) != 0:
                ent_verb[ent][text] = ent_dic[ent][text]


# In[175]:


print(ent_verb['hong kong'])


# ### **Export**
# 
# change file name accordingly

# In[176]:


export = 'entity_verbs_youtest.json'


# In[177]:


with open(export, 'w') as file:
    json.dump(ent_verb, file)


# ## **Combine nouns wordkeys and entities_action pair positions**

# ### **Reddit**

# In[3]:


with open('entities_action_red.json', 'r') as f:
    ent_act = json.load(f)
with open (dname + '\\02_reddit\\wordkeys_nouns_red.json', 'r') as f:
    wordkeys_nouns = json.load(f)


# ### **Youtube - train**

# In[4]:


with open('entities_action_youtrain.json', 'r') as f:
    ent_act = json.load(f)
with open (dname + '\\03_youtube\\wordkeys_nouns_youtrain.json', 'r') as f:
    wordkeys_nouns = json.load(f)


# ### **Youtube - whole**

# In[3]:


with open('entities_action_you.json', 'r') as f:
    ent_act = json.load(f)
with open (dname + '\\03_youtube\\wordkeys_nouns_you.json', 'r') as f:
    wordkeys_nouns = json.load(f)


# ### **Youtube - test**

# In[178]:


with open('entities_action_youtest.json', 'r') as f:
    ent_act = json.load(f)
with open (dname + '\\03_youtube\\wordkeys_nouns_youtest.json', 'r') as f:
    wordkeys_nouns = json.load(f)


# ### **prepare new wordkey noun dic**
# 
# ### **code below not for test corpora**

# In[181]:


for i in wordkeys_nouns:
    for act in wordkeys_nouns[i]:
        pos = []
        bol = []
        if len(wordkeys_nouns[i][act]) != 1:
            if wordkeys_nouns[i][act][1] == wordkeys_nouns[i][act][0]+1 and wordkeys_nouns[i][act][-1] == wordkeys_nouns[i][act][-2]+1:
                for j in range(len(wordkeys_nouns[i][act])-1):
                    if wordkeys_nouns[i][act][j]+1 == wordkeys_nouns[i][act][j+1]:
                        bol.append(True)
                    else:
                        bol.append(False)
                if len(bol) != 0:
                    p = []
                    for a in range(len(bol)):
                        if bol[a] == True:
                            if a == len(bol)-1:
                                p.append(wordkeys_nouns[i][act][a])
                                p.append(wordkeys_nouns[i][act][a+1])
                                pos.append(p)
                            else:
                                if bol[a+1] == True:
                                    p.append(wordkeys_nouns[i][act][a])
                                elif bol[a+1] == False:
                                    p.append(wordkeys_nouns[i][act][a])
                                    p.append(wordkeys_nouns[i][act][a+1])
                                    pos.append(p)
                                    p = []
                        elif bol[a] == False:
                            if i == len(bol)-1:
                                pos.append(wordkeys_nouns[i][act][a])
                                pos.append(wordkeys_nouns[i][act][a+1])
                            else:
                                if bol[a-1] == True:
                                    pass
                                else:
                                    pos.append(wordkeys_nouns[i][act][a])
                wordkeys_nouns[i][act] = pos
            else:
                pos2 = []
                for v in wordkeys_nouns[i][act]:
                    pos2.append([v])
                    wordkeys_nouns[i][act] = pos2
        else:
            wordkeys_nouns[i][act] = [wordkeys_nouns[i][act]]


# In[182]:


print(wordkeys_nouns['0'])


# In[183]:


with open('wordkeys_nouns_youtest_new.json', 'w') as file:
    json.dump(wordkeys_nouns, file)


# ### **prepare new pair dic**

# In[179]:


new_pair = {}
for pair in ent_act:
    new_pair[pair] = {}
    for text in ent_act[pair]['text']:
        if text in wordkeys_nouns:
            new_pair[pair][text] = {}
            for ind, pos_act in enumerate(ent_act[pair]['pos-act']):
                new_pair[pair][text]['pos'+str(ind)] = pos_act


# In[180]:


print(wordkeys_nouns['237'])


# ### **combine dictionaries**

# In[8]:


original_wordkeys = dname + '\\03_youtube\\wordkeys_nouns_you.json'


# In[9]:


with open (original_wordkeys, 'r') as f:
    wordkeys_nouns_orig = json.load(f)


# In[53]:


new_pair2 = {}
err = {'pair': [], 'text': [], 'noun': []}
range_error = {'pair': [], 'text': [], 'pos': []}
for pair in to_check:
    new_pair2[pair] = {}
    for text in new_pair[pair]:
        new_pair2[pair][text] = {}
        n = 0
        for pos in new_pair[pair][text]:
            new_pair2[pair][text][pos] = new_pair[pair][text][pos]
            new_pair2[pair][text]['tar'+str(n)] = {}
            try:
                if len(pos) == 1:
                    posrange = [*range(new_pair[pair][text][pos][0], new_pair[pair][text][pos][0] + 10)]
                elif len(pos) > 1:
                    posrange = [*range(new_pair[pair][text][pos][-1], new_pair[pair][text][pos][-1] + 10)]
                for noun in wordkeys_nouns[text]:
                    new_pair2[pair][text]['tar'+str(n)][noun] = []
                    for ind in range(len(wordkeys_nouns[text][noun])):
                        try:
                            if wordkeys_nouns[text][noun][ind][0] in posrange:
                                new_pair2[pair][text]['tar'+str(n)][noun].append(wordkeys_nouns[text][noun][ind])
                        except TypeError:
                            if text not in err['text'] and noun not in err['noun']:
                                err['pair'].append(pair)
                                err['text'].append(text)
                                err['noun'].append(noun)
                                pass
            except TypeError:
                if pair not in range_error['pair'] and text not in range_error['text'] and pos not in range_error['pos']:
                    range_error['pair'].append(pair)
                    range_error['text'].append(text)
                    range_error['pos'].append(pos)
                    pass
                
            n += 1


# In[138]:


# if there are errors, execute the four blocks of code below to fix it and then run the above code again, then the errors should be empty
print(err)


# In[139]:


lol = []
for ind, text in enumerate(err['text']):
    lol.append(wordkeys_nouns_orig[text][err['noun'][ind]])
print(lol)


# In[140]:


for ind in range(len(lol)):
    new = []
    for i in lol[ind]:
        new.append([i])
    text = err['text'][ind]
    noun = err['noun'][ind]
    wordkeys_nouns[text][noun] = new
    print('changed', text, noun)


# In[141]:


for ind, text in enumerate(err['text']):
    print(wordkeys_nouns[text][err['noun'][ind]])


# ### **clean dictionary #1**

# In[114]:


ent_act2 = {}
for pair in new_pair2:
    ent_act2[pair] = {}
    for text in new_pair2[pair]:
        n = 0
        ent_act2[pair][text] = {}
        ra = [*range(int(len(new_pair2[pair][text])/2))]
        for r in ra:
            ent_act2[pair][text]['pos'+str(r)] = new_pair2[pair][text]['pos'+str(r)]
            ent_act2[pair][text]['tar'+str(r)] = {}
            for target in new_pair2[pair][text]['tar'+str(r)]:
                if len(new_pair2[pair][text]['tar'+str(r)][target]) != 0:
                    ent_act2[pair][text]['tar'+str(r)][target] = new_pair2[pair][text]['tar'+str(r)][target]


# In[115]:


print(ent_act2['us-are'])


# In[116]:


print(len(ent_act2['us-are']['134']['tar1']))


# In[117]:


act_noun = {}
for pair in ent_act2:
    act_noun[pair] = {}
    for text in ent_act2[pair]:
        ra = [*range(int(len(ent_act2[pair][text])/2))]
        act_noun[pair][text] = {}
        for r in ra:
            if len(ent_act2[pair][text]['tar'+str(r)]) != 0:
                act_noun[pair][text]['pos'+str(r)] = ent_act2[pair][text]['pos'+str(r)]
                act_noun[pair][text]['tar'+str(r)] = ent_act2[pair][text]['tar'+str(r)]


# In[119]:


print(act_noun['us-are'])


# In[120]:


act_noun2 = {}
for pair in act_noun:
    act_noun2[pair] = {}
    for text in act_noun[pair]:
        if len(act_noun[pair][text]) != 0:
            act_noun2[pair][text] = act_noun[pair][text]


# In[121]:


print(len(act_noun2))


# In[122]:


print(len(ent_act))


# In[123]:


act_noun3 = {}
for pair in act_noun2:
    if len(act_noun2[pair]) != 0:
        act_noun3[pair] = act_noun2[pair]


# In[124]:


print(len(act_noun3))


# In[125]:


act_noun3 = {}
for pair in act_noun2:
    if len(act_noun2[pair]) != 0:
        act_noun3[pair] = act_noun2[pair]


# In[126]:


print(len(act_noun3))


# In[127]:


act_noun4 = {}
for pair in act_noun3:
    act_noun4[pair] = {}
    for text in act_noun3[pair]:
        k = list(act_noun3[pair][text].keys())
        rr = [*range(int(len(act_noun3[pair][text])/2))]
        ra = [*range(0,int(len(act_noun3[pair][text])),2)]
        rt = [*range(1,int(len(act_noun3[pair][text])),2)]
        act_noun4[pair][text] = {}
        for r in rr:
            act_noun4[pair][text]['pos'+str(r)] = act_noun3[pair][text][k[ra[r]]]
            act_noun4[pair][text]['tar'+str(r)] = act_noun3[pair][text][k[rt[r]]]


# In[128]:


print(len(act_noun4))


# ### **Export**
# 
# change file name accordingly

# In[129]:


export = 'act_nouns_you.json'


# In[130]:


with open(export, 'w') as file:
    json.dump(act_noun4, file)

