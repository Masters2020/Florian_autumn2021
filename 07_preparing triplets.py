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


# ## **Preparing triplets**

# ### **Wordkeys**

# In[20]:


def wordkeys_corpus (input_list, tokenizer, outfile):
    wordkeys = {}
    
    for i in range(len(input_list)):
        ### tokenize the input text and get the words and their ids for later
        tokenized = tokenizer(input_list[i], add_special_tokens=False) #need to rewrite so if input bigger than 512, it gets tokenized the same way as for the embedding, create the tokens and word_ids and append them with everything 
        tokens = tokenizer.convert_ids_to_tokens(tokenized['input_ids'])
        word_ids = tokenized.word_ids()
        
        ###prepare the initial dictionary which includes the word ids as keys and the index (or indeces if subwords) of the word in a list as values
        init_dic = {} 
        for id_ in word_ids:
            init_dic[id_] = []
        init_keys = list(init_dic.keys())
        for index, value in enumerate(word_ids):
            init_dic[value].append(index)
        
        ### using the dictionary with the word ids and indeces, iterate through and replace the word ids with the actual word 
        ### if a word appears several times, the indeces are combined together in a list, so at the end the dictionary include the word and all of its positions
        wordkeys[i] = {}
        for index, key in enumerate(init_keys):
            if len(init_dic[key]) == 1:
                new = tokens[init_dic[key][0]]
                if new not in wordkeys[i]:
                    wordkeys[i][new] = init_dic[key]
                elif new in wordkeys[i]:
                    wordkeys[i][new].append(init_dic[key][0])
            elif len(init_dic[key]) > 1:
                range_of_length = list(range(1,len(init_dic[key])))
                str_ = []
                for tok in init_dic[key]:
                    str_.append(tokens[tok].strip('#'))
                    new = ''.join(str_)
                    if new not in wordkeys[i]:
                        wordkeys[i][new] = init_dic[key]
                    elif new in wordkeys[i]:
                        for item in init_dic[key]:
                            wordkeys[i][new].append(item)
                keys_wordkeys = list(wordkeys[i].keys())
                new_ind = keys_wordkeys.index(new)
                for ind_sub in range_of_length:
                    ind_to_del = keys_wordkeys[new_ind - ind_sub]
                    del wordkeys[i][ind_to_del]
                
    with open(outfile, 'w') as file:
        json.dump(wordkeys, file)
    
    return wordkeys


# In[13]:


wordkeys_red = wordkeys_corpus(inputs_red, tokenizer_red, "wordkeys_red.json")


# In[21]:


wordkeys_you = wordkeys_corpus(inputs_you, tokenizer_you, "wordkeys_you.json")


# ### **Extract entities and positions**

# In[11]:


with open(dname + '\\03_youtube_baseline\\wordkeys_you.json') as f:
    wordkeys_you = json.load(f)


# In[12]:


def getentities (vocab, wordkeys, outfile):
    entity_dic = {}
    with open(vocab, 'r', encoding = 'utf-8') as file1:
        new_tokens = file1.readlines()
    clean_tokens = []
    for i in new_tokens:
        j = i.replace('\n', '')
        clean_tokens.append(j)
    
    for text in wordkeys:
        for ent in clean_tokens:
            if ent in wordkeys[text]:
                if ent in entity_dic and text in entity_dic[ent]:
                    for ind in wordkeys[text][ent]:
                        entity_dic[ent][text].append(wordkeys[text][ent][ind])
                elif ent in entity_dic and text not in entity_dic[ent]:
                    entity_dic[ent][text] = wordkeys[text][ent]
                elif ent not in entity_dic:
                    entity_dic[ent] = {}
                    entity_dic[ent][text] = wordkeys[text][ent]
    
    with open(outfile, 'w') as outfile:
        json.dump(entity_dic, outfile)
    
    return entity_dic


# In[16]:


entities_red = getentities('NER_vocab_10.txt', wordkeys_red, 'entities_red.json')


# In[14]:


entities_you = getentities(dname + '\\03_youtube_baseline\\NER_vocab_10_you.txt', wordkeys_you, 'entities_you.json')


# ### **PoS tags**
# 
# The formula will only include texts into the dictionary with verbs that also have an entity marked in them

# In[26]:


def postags (input_list, entities, pos_type, outfile):
    nlp = spacy.load("en_core_web_trf")
    pos = {}
    for ent in entities:
        for tex in entities[ent]:
            if tex not in pos:
                doc = nlp(input_list[tex]) 

                pos_list = []
                for token in doc:
                    if token.pos_ == pos_type:
                        pos_list.append(token.text)
                pos_list = [string.lower() for string in pos_list] #necessary for comparison with bert tokenized text that was feed to model 
                pos_set = set(pos_list)
                pos_list_final = []
                for i in pos_set:
                    if len(i) > 1 and i not in pos_list_final:
                        pos_list_final.append(i)

                pos[tex] = pos_list_final
            
            elif tex in pos:
                pass
    
    with open(outfile, 'w') as file:
        json.dump(pos, file)
    
    return pos


# In[18]:


pos_verbs_red = postags (inputs_red, entities_red, 'VERB', 'pos_verbs_red10.json')


# In[19]:


pos_nouns_red = postags (inputs_red, entities_red, 'NOUN', 'pos_nouns_red10.json')


# In[27]:


pos_verbs_you = postags (inputs_you, entities_you, 'VERB', 'pos_verbs_you.json')


# In[28]:


pos_nouns_you = postags (inputs_you, entities_you, 'NOUN', 'pos_nouns_you.json')


# ### **Wordkeys for verbs and nouns**

# In[15]:


with open(dname + '\\03_youtube_baseline\\pos_nouns_you.json') as f:
    pos_nouns_you = json.load(f)
with open(dname + '\\03_youtube_baseline\\pos_verbs_you.json') as f:
    pos_verbs_you = json.load(f)


# In[16]:


def wordkeys_for_pos (pos, wordkeys, outfile1, outfile2):
    wordkeys_pos = {}
    error_pos = {}
    for text in pos:
        wordkeys_pos[text] = {}
        for act in pos[text]:
            if act in wordkeys[text]:
                if act in wordkeys_pos[text] and len(wordkeys[text][act]) > 1:
                    for v in range(len(wordkeys[text][act])):
                        wordkeys_pos[text][act].append(wordkeys[text][act][v])
                elif act in wordkeys_pos[text] and len(wordkeys[text][act]) == 1:
                    wordkeys_pos[text][act].append(wordkeys[text][act][0])
                elif act not in wordkeys_pos[text]:
                    wordkeys_pos[text][act] = wordkeys[text][act]
            else:
                error_pos[text] = []
                error_pos[text].append(act)
        
    with open(outfile1, 'w') as file:
        json.dump(wordkeys_pos, file)
    with open(outfile2, 'w') as file:
        json.dump(error_pos, file)
    
    return wordkeys_pos, error_pos


# In[36]:


wordkeys_verbs_red, error_verbs_red = wordkeys_for_pos (pos_verbs_red, wordkeys_red, 'wordkeys_verbs_red10.json', 'error_verbs_red10.json')


# In[37]:


wordkeys_nouns_red, error_nouns_red = wordkeys_for_pos (pos_nouns_red, wordkeys_red, 'wordkeys_nouns_red10.json', 'error_nouns_red10.json')


# In[17]:


wordkeys_verbs_you, error_verbs_you = wordkeys_for_pos (pos_verbs_you, wordkeys_you, 'wordkeys_verbs_you.json', 'error_verbs_you.json')


# In[18]:


wordkeys_nouns_you, error_nouns_you = wordkeys_for_pos (pos_nouns_you, wordkeys_you, 'wordkeys_nouns_you.json', 'error_nouns_you.json')


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

