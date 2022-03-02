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

