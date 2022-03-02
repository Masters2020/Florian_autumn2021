#!/usr/bin/env python
# coding: utf-8

# ## **Set-up**

# In[2]:


import os
import pandas as pd
import scipy
import scipy.spatial
import numpy as np
import json
import spacy
import copy
import time
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns


# In[3]:


path = os.path.abspath('10_classification.ipynb')  #this saves the path to this notebook and uses it as a base to access the required csvs
dname = os.path.dirname(path)
os.chdir(dname)


# ## **Prepare triplets for texts to be classified**

# ### **Whole YouTube corpus - transfer approach**

# In[18]:


with open('entity_verbs_you.json') as f:
    ent_verbs_you = json.load(f)


# In[64]:


def getpairs (ent_verbs, outfile):
    entity_action = {}
    for ent in ent_verbs:
        for tex in ent_verbs[ent]:
            ra = [*range(int(len(ent_verbs[ent][tex])/2))]
            for r in ra:
                pos = ent_verbs[ent][tex]['pos'+str(r)]
                for act in ent_verbs[ent][tex]['act'+str(r)]:
                    if act != ent:
                        for ind in range(len(ent_verbs[ent][tex]['act'+str(r)][act])):
                            if ent_verbs[ent][tex]['pos'+str(r)] != ent_verbs[ent][tex]['act'+str(r)][act][ind][0]:
                                ent_act = ent + '-' + act
                                if ent_act in entity_action:
                                    entity_action[ent_act]['text'].append(tex)
                                    entity_action[ent_act]['pos-ent'].append([pos])
                                    entity_action[ent_act]['pos-act'].append(ent_verbs[ent][tex]['act'+str(r)][act][ind])
                                elif ent_act not in entity_action:
                                    entity_action[ent_act] = {}
                                    entity_action[ent_act]['text'] = [tex]
                                    entity_action[ent_act]['pos-ent'] = [[pos]]
                                    entity_action[ent_act]['pos-act'] = [ent_verbs[ent][tex]['act'+str(r)][act][ind]]
    
    with open(outfile, 'w') as f:
        json.dump(entity_action, f)
    
    return entity_action


# In[65]:


ent_act_you = getpairs(ent_verbs_you, 'entities_action_you.json')


# In[84]:


with open('wordkeys_nouns_you_new.json', 'r') as f:
    wordkeys_noun = json.load(f)


# In[121]:


def gettriplets (ent_act, wordkeys_noun, outfile):
    triplets = {}
    err = {'pair': [], 'text': [], 'noun': []}
    for pair in ent_act:
        check = pair.split('-')
        for ind, text in enumerate(ent_act[pair]['text']):
            pos = ent_act[pair]['pos-act'][ind][-1]
            posrange = [*range(pos, pos + 10)]
            for noun in wordkeys_noun[text]:
                if noun != check[0] and noun != check[1]:
                    for ind_ in range(len(wordkeys_noun[text][noun])):
                        try:
                            if wordkeys_noun[text][noun][ind_][0] in posrange and wordkeys_noun[text][noun][ind_][0] != pos:
                                trip = pair + '-' + noun
                                if trip in triplets:
                                    triplets[trip]['text'].append(text)
                                    triplets[trip]['pos-ent'].append(ent_act[pair]['pos-ent'][ind])
                                    triplets[trip]['pos-act'].append(ent_act[pair]['pos-act'][ind])
                                    triplets[trip]['pos-tar'].append(wordkeys_noun[text][noun][ind_])
                                elif trip not in triplets:
                                    triplets[trip] = {}
                                    triplets[trip]['text'] = [text]
                                    triplets[trip]['pos-ent'] = [ent_act[pair]['pos-ent'][ind]]
                                    triplets[trip]['pos-act'] = [ent_act[pair]['pos-act'][ind]]
                                    triplets[trip]['pos-tar'] = [wordkeys_noun[text][noun][ind_]]
                        except TypeError:
                            if text not in err['text'] and noun not in err['noun']:
                                err['pair'].append(pair)
                                err['text'].append(text)
                                err['noun'].append(noun)
                                
    with open (outfile, 'w') as f:
        json.dump(triplets, f)
        
    return triplets, err


# In[122]:


triplets, err = gettriplets(ent_act_you, wordkeys_noun, 'triplets_you.json')


# In[123]:


print(err)


# #### **If errors not empty, use code below and then execute function again**

# In[111]:


original_wordkeys = dname + '\\03_youtube\\wordkeys_nouns_you.json'


# In[112]:


with open (original_wordkeys, 'r') as f:
    wordkeys_nouns_orig = json.load(f)


# In[113]:


lol = []
for ind, text in enumerate(err['text']):
    lol.append(wordkeys_nouns_orig[text][err['noun'][ind]])
print(lol)


# In[115]:


for ind in range(len(lol)):
    new = []
    for i in lol[ind]:
        new.append([i])
    text = err['text'][ind]
    noun = err['noun'][ind]
    wordkeys_noun[text][noun] = new
    print('changed', text, noun)


# In[117]:


for ind, text in enumerate(err['text']):
    print(wordkeys_noun[text][err['noun'][ind]])


# ### **Test YouTube corpus - baseline approach**

# In[124]:


with open('entity_verbs_youtest.json') as f:
    ent_verbs_youtest = json.load(f)


# In[125]:


def getpairs (ent_verbs, outfile):
    entity_action = {}
    for ent in ent_verbs:
        for tex in ent_verbs[ent]:
            ra = [*range(int(len(ent_verbs[ent][tex])/2))]
            for r in ra:
                pos = ent_verbs[ent][tex]['pos'+str(r)]
                for act in ent_verbs[ent][tex]['act'+str(r)]:
                    if act != ent:
                        for ind in range(len(ent_verbs[ent][tex]['act'+str(r)][act])):
                            ent_act = ent + '-' + act
                            if ent_act in entity_action:
                                entity_action[ent_act]['text'].append(tex)
                                entity_action[ent_act]['pos-ent'].append([pos])
                                entity_action[ent_act]['pos-act'].append(ent_verbs[ent][tex]['act'+str(r)][act][ind])
                            elif ent_act not in entity_action:
                                entity_action[ent_act] = {}
                                entity_action[ent_act]['text'] = [tex]
                                entity_action[ent_act]['pos-ent'] = [[pos]]
                                entity_action[ent_act]['pos-act'] = [ent_verbs[ent][tex]['act'+str(r)][act][ind]]
    
    with open(outfile, 'w') as f:
        json.dump(entity_action, f)
    
    return entity_action


# In[126]:


ent_act_youtest = getpairs(ent_verbs_youtest, 'entities_action_youtest.json')


# In[127]:


with open('wordkeys_nouns_youtest_new.json', 'r') as f:
    wordkeys_noun = json.load(f)


# In[155]:


def gettriplets (ent_act, wordkeys_noun, outfile):
    triplets = {}
    err = {'pair': [], 'text': [], 'noun': []}
    for pair in ent_act:
        check = pair.split('-')
        for ind, text in enumerate(ent_act[pair]['text']):
            pos = ent_act[pair]['pos-act'][ind][-1]
            posrange = [*range(pos, pos + 10)]
            for noun in wordkeys_noun[text]:
                if noun != check[0] and noun != check[1]:
                    for ind_ in range(len(wordkeys_noun[text][noun])):
                        try:
                            if wordkeys_noun[text][noun][ind_][0] in posrange and wordkeys_noun[text][noun][ind_][0] != pos:
                                trip = pair + '-' + noun
                                if trip in triplets:
                                    triplets[trip]['text'].append(text)
                                    triplets[trip]['pos-ent'].append(ent_act[pair]['pos-ent'][ind])
                                    triplets[trip]['pos-act'].append(ent_act[pair]['pos-act'][ind])
                                    triplets[trip]['pos-tar'].append(wordkeys_noun[text][noun][ind_])
                                elif trip not in triplets:
                                    triplets[trip] = {}
                                    triplets[trip]['text'] = [text]
                                    triplets[trip]['pos-ent'] = [ent_act[pair]['pos-ent'][ind]]
                                    triplets[trip]['pos-act'] = [ent_act[pair]['pos-act'][ind]]
                                    triplets[trip]['pos-tar'] = [wordkeys_noun[text][noun][ind_]]
                        except TypeError:
                            if text not in err['text'] and noun not in err['noun']:
                                err['pair'].append(pair)
                                err['text'].append(text)
                                err['noun'].append(noun)
                                
    with open (outfile, 'w') as f:
        json.dump(triplets, f)
        
    return triplets, err


# In[163]:


triplets, err = gettriplets(ent_act_youtest, wordkeys_noun, 'triplets_youtest.json')


# In[164]:


print(err)


# #### **If errors not empty, use code below and then execute function again**

# In[158]:


original_wordkeys = dname + '\\03_youtube\\wordkeys_nouns_youtest.json'


# In[159]:


with open (original_wordkeys, 'r') as f:
    wordkeys_nouns_orig = json.load(f)


# In[160]:


lol = []
for ind, text in enumerate(err['text']):
    lol.append(wordkeys_nouns_orig[text][err['noun'][ind]])
print(lol)


# In[161]:


for ind in range(len(lol)):
    new = []
    for i in lol[ind]:
        new.append([i])
    text = err['text'][ind]
    noun = err['noun'][ind]
    wordkeys_noun[text][noun] = new
    print('changed', text, noun)


# In[162]:


for ind, text in enumerate(err['text']):
    print(wordkeys_noun[text][err['noun'][ind]])


# ## **Format triplets**

# ### **restructure**

# In[165]:


with open('triplets_youtrain.json') as f:
    trip_youtrain = json.load(f)
with open('triplets_youtest.json') as f:
    trip_youtest = json.load(f)
with open('triplets_you.json') as f:
    trip_you = json.load(f)
with open('triplets_red.json') as f:
    trip_red = json.load(f)


# In[330]:


with open('entities_action_youtrain.json') as f:
    entact_youtrain = json.load(f)
with open('entities_action_youtest.json') as f:
    entact_youtest = json.load(f)
with open('entities_action_you.json') as f:
    entact_you = json.load(f)
with open('entities_action_red.json') as f:
    entact_red = json.load(f)


# In[3]:


with open(dname + '\\03_youtube\\entities_youtrain.json') as f:
    ent_youtrain = json.load(f)
with open(dname + '\\03_youtube\\entities_youtest.json') as f:
    ent_youtest = json.load(f)
with open(dname + '\\03_youtube\\entities_you.json') as f:
    ent_you = json.load(f)
with open(dname + '\\02_reddit\\entities_red.json') as f:
    ent_red = json.load(f)


# In[9]:


def restructure (triplets, outfile):
    new_trips = {}
    text_list = []
    for trip in triplets:
        for text in triplets[trip]['text']:
            if text not in text_list:
                text_list.append(text)

    for text_ in text_list:
        new_trips[text_] = {}
    for trip_ in triplets:
        for _text in triplets[trip_]['text']:
            if trip_ in new_trips[_text]:
                new_trips[_text][trip_] = new_trips[_text][trip_] + 1
            elif trip_ not in new_trips[_text]:
                new_trips[_text][trip_] = 1
    
    with open (outfile, 'w') as f:
        json.dump(new_trips, f)
    
    return new_trips, text_list


# #### **triplets**

# In[187]:


trip_red_new, text_list = restructure (trip_red, 'triplets_red_new.json')


# In[198]:


trip_you_new, text_list = restructure (trip_you, 'triplets_you_new.json')


# In[199]:


trip_youtrain_new, text_list = restructure (trip_youtrain, 'triplets_youtrain_new.json')


# In[200]:


trip_youtest_new, text_list = restructure (trip_youtest, 'triplets_youtest_new.json')


# #### **pairs**

# In[331]:


entact_red_new, text_list = restructure (entact_red, 'entities_action_red_new.json')


# In[332]:


entact_you_new, text_list = restructure (entact_you, 'entities_action_you_new.json')


# In[333]:


entact_youtrain_new, text_list = restructure (entact_youtrain, 'entities_action_youtrain_new.json')


# In[334]:


entact_youtest_new, text_list = restructure (entact_youtest, 'entities_action_youtest_new.json')


# #### **entities**

# In[142]:


def restructure (triplets, outfile):
    new_trips = {}
    text_list = []
    for trip in triplets:
        for text in triplets[trip]:
            if text not in text_list:
                text_list.append(text)

    for text_ in text_list:
        new_trips[text_] = {}
    for trip_ in triplets:
        for _text in triplets[trip_]:
            am = len(triplets[trip_][_text])
            if trip_ in new_trips[_text]:
                new_trips[_text][trip_] = new_trips[_text][trip_] + am
            elif trip_ not in new_trips[_text]:
                new_trips[_text][trip_] = am
    
    with open (outfile, 'w') as f:
        json.dump(new_trips, f)
    
    return new_trips, text_list


# In[143]:


ent_red_new, text_list = restructure (ent_red, 'entities_red_new.json')


# In[144]:


ent_you_new, text_list = restructure (ent_you, 'entities_you_new.json')


# In[145]:


ent_youtrain_new, text_list = restructure (ent_youtrain, 'entities_youtrain_new.json')


# In[146]:


ent_youtest_new, text_list = restructure (ent_youtest, 'entities_youtest_new.json')


# #### **tests**

# In[190]:


print(text_list)


# In[191]:


text_list_youtrain = []
for item in text_list:
    text_list_youtrain.append(int(item))
text_list_youtrain.sort()


# In[193]:


text_list_youtest = []
for item in text_list:
    text_list_youtest.append(int(item))
text_list_youtest.sort()


# In[223]:


print(text_list_youtrain)


# In[239]:


print(text_list_youtest)


# In[210]:


test = dname + '\\01_data\\val_you_text.csv'
train = dname + '\\01_data\\train_you_text.csv'


# In[211]:


test_data = pd.read_csv(test)
train_data = pd.read_csv(train)


# In[213]:


test_texts = test_data["index"].tolist()
train_texts = train_data["index"].tolist()


# In[225]:


test = []
for item in text_list_youtrain:
    if item not in test_texts:
        test.append(item)


# ### **lemmatise**

# In[140]:


def lemmatise (triplets, outfile):
    nlp = spacy.load("en_core_web_trf")
    triplets_lem = {}
    for text in triplets:
        triplets_lem[text] = {}
        for trip in triplets[text]:
            tok = trip.split('-')
            new = []
            new_ = ''
            for t in tok:
                if t == 'us':
                    new.append('us')
                elif t == 'fed':
                    new.append('fed')
                else:
                    doc = nlp(t)
                    for token in doc:
                        new.append(token.lemma_)
                        
            for part in new:
                new_ += part + '-'
            new_trip = new_[:-1]
                
            if new_trip in triplets_lem[text]:
                triplets_lem[text][new_trip] = triplets_lem[text][new_trip] + 1
            elif new_trip not in triplets_lem[text]:
                triplets_lem[text][new_trip] = triplets[text][trip]
                
    with open (outfile, 'w') as f:
        json.dump(triplets_lem, f)
    
    return triplets_lem    


# #### **triplets**

# In[235]:


trip_red_lem = lemmatise (trip_red_new, 'triplets_red_lem.json')


# In[236]:


trip_you_lem = lemmatise (trip_you_new, 'triplets_you_lem.json')


# In[237]:


trip_youtrain_lem = lemmatise (trip_youtrain_new, 'triplets_youtrain_lem.json')


# In[238]:


trip_youtest_lem = lemmatise (trip_youtest_new, 'triplets_youtest_lem.json')


# #### **pairs**

# In[342]:


entact_red_lem = lemmatise (entact_red_new, 'entities_action_red_lem.json')


# In[ ]:


entact_you_lem = lemmatise (entact_you_new, 'entities_action_you_lem.json')


# In[ ]:


entact_youtrain_lem = lemmatise (entact_youtrain_new, 'entities_action_youtrain_lem.json')


# In[ ]:


entact_youtest_lem = lemmatise (entact_youtest_new, 'entities_action_youtest_lem.json')


# #### **entities**

# In[147]:


ent_red_lem = lemmatise (ent_red_new, 'entities_red_lem.json')


# In[148]:


ent_you_lem = lemmatise (ent_you_new, 'entities_you_lem.json')


# In[149]:


ent_youtrain_lem = lemmatise (ent_youtrain_new, 'entities_youtrain_lem.json')


# In[150]:


ent_youtest_lem = lemmatise (ent_youtest_new, 'entities_youtest_lem.json')


# #### **implemented in code above, keep in case something goes wrong**

# In[335]:


#fix for us being converted to we
def fix (triplets, outfile):
    new_trip = {}
    for text in triplets:
        new_trip[text] = {}
        for trip in triplets[text]:
            tripsplit = trip.split('-')
            new = []
            newstr = ''
            for part in tripsplit:
                if part == 'we':
                    new.append('us')
                elif part == 'feed':
                    new.append('fed')
                else:
                    new.append(part)
            for part_ in new:
                newstr += part_ + '-'
            key = newstr[:-1]
            new_trip[text][key] = triplets[text][trip]
   
    with open (outfile, 'w') as f:
        json.dump(new_trip, f)
        
    return new_trip


# In[322]:


trip_red_lem02 = fix (trip_red_lem, 'triplets_red_lem02.json')


# In[323]:


trip_you_lem02 = fix (trip_you_lem, 'triplets_you_lem02.json')


# In[324]:


trip_youtrain_lem02 = fix (trip_youtrain_lem, 'triplets_youtrain_lem02.json')


# In[325]:


trip_youtest_lem02 = fix (trip_youtest_lem, 'triplets_youtest_lem02.json')


# ## **Classification**

# In[151]:


you = dname + '\\01_data\\you_all.csv'


# In[152]:


you_data = pd.read_csv(you)


# In[153]:


print(you_data)


# ### **Whole YouTube corpus - transfer approach**

# In[386]:


with open('entities_action_you_lem.json') as f:
    entact_you_lem = json.load(f)
with open('entities_action_red_lem.json') as f:
    entact_red_lem = json.load(f)


# In[17]:


def classification_transfer (train, test, data):
    results = {}
    labels = data['label'].tolist()
    texts = data['index'].tolist() 
    for ind, tex in enumerate(texts):
        results[str(tex)] = {'label': labels[ind], 'total': 0, 'matches': 0, 'triplets': [], 'binary': 0, 'confidence': -1}
        
    knowledge_base = []
    for text in train:
        for trip in train[text]:
            if trip not in knowledge_base:
                knowledge_base.append(trip)
                
    for text in test:
        for trip in test[text]:
            results[text]['total'] = results[text]['total'] + 1
            if trip in knowledge_base:
                results[text]['triplets'].append(trip)
                results[text]['matches'] = results[text]['matches'] + 1
                
    for text in results:
        if results[text]['matches'] == 0:
            results[text]['binary'] = 1
        else:
            results[text]['binary'] = 2
            
        if results[text]['total'] == 0:
            results[text]['confidence'] = 0
        else:
            results[text]['confidence'] = round(results[text]['matches']/results[text]['total'], 3)
    
    return results


# In[356]:


results_transfer = classification_transfer(trip_red_lem02, trip_you_lem02, you_data)


# In[387]:


results_transfer_entact = classification_transfer(entact_red_lem, entact_you_lem, you_data)


# In[18]:


results_transfer_ent = classification_transfer(ent_red_lem, ent_you_lem, you_data)


# In[357]:


with open ('results_transfer.json', 'w') as f:
    json.dump(results_transfer, f)


# In[389]:


with open ('results_transfer_entact.json', 'w') as f:
    json.dump(results_transfer_entact, f)


# In[19]:


with open ('results_transfer_ent.json', 'w') as f:
    json.dump(results_transfer_ent, f)


# ### **Test YouTube corpus - baseline approach**

# In[392]:


with open('entities_action_youtrain_lem.json') as f:
    entact_youtrain_lem = json.load(f)
with open('entities_action_youtest_lem.json') as f:
    entact_youtest_lem = json.load(f)


# In[254]:


def confidence_trips (triplets, data):
    labels = data['label'].tolist()
    texts = data['index'].tolist()
    texts_str = [str(x) for x in texts]
    
    trip_text = {}
    for text in triplets:
        l = labels[texts_str.index(text)]
        for trip in triplets[text]:
            if trip in trip_text:
                trip_text[trip]['text'].append(text)
                trip_text[trip]['label'].append(l)
                trip_text[trip]['total'] = trip_text[trip]['total'] + 1
                if l == 2:
                    trip_text[trip]['con'] = trip_text[trip]['con'] + 1
            elif trip not in trip_text:
                trip_text[trip] = {}
                trip_text[trip]['text'] = [text]
                trip_text[trip]['label'] = [l]
                trip_text[trip]['total'] = 1
                trip_text[trip]['con'] = 0
                if l == 2:
                    trip_text[trip]['con'] = trip_text[trip]['con'] + 1
                    
    conf_trip = {}
    for trip in trip_text:
        conf_trip[trip] = round(trip_text[trip]['con']/trip_text[trip]['total'],3)
    
    return conf_trip, trip_text


# In[255]:


conf_trip, trip_text = confidence_trips(trip_youtrain_lem, you_data)


# In[256]:


conf_pair, pair_text = confidence_trips(entact_youtrain_lem, you_data)


# In[257]:


conf_ent, pair_text = confidence_trips(ent_youtrain_lem, you_data)


# In[314]:


def getstuff (conf):
    results = {'max': [], 'maxv': 0, 'min': [], 'minv': 1, 'avg': 0}
    alle = []
    summe = 0
    for x in conf:
        alle.append(round(conf[x],1))
        summe += conf[x]
        if conf[x] > results['maxv']:
            results['maxv'] = conf[x]
            results['max'] = x
        if conf[x] < results['minv']:
            results['minv'] = conf[x]
            results['min'] = x
    avg = summe/len(alle)
    results['avg'] = avg
    
    return results, alle  


# In[315]:


res_con_ent, alle_con_ent = getstuff(conf_ent)


# In[316]:


res_con_pair, alle_con_pair = getstuff(conf_pair)


# In[317]:


res_con_trip, alle_con_trip = getstuff(conf_trip)


# In[318]:


print(res_con_trip)


# In[319]:


print(res_con_pair)


# In[320]:


print(res_con_ent)


# In[322]:


plt.figure(figsize=(20, 6))
plt.subplot(131)
sns.set_theme(style="darkgrid")
sns.countplot(x = alle_con_ent)
plt.title('confidence score distribution entities', pad = 15)
plt.subplot(132)
sns.set_theme(style="darkgrid")
sns.countplot(x = alle_con_pair)
plt.title('confidence score distribution pairs', pad = 15)
plt.subplot(133)
sns.set_theme(style="darkgrid")
sns.countplot(x = alle_con_trip)
plt.title('confidence score distribution triplets', pad = 15)
plt.savefig('distribution confidence scores.png')
plt.show()


# In[259]:


with open ('confidence_triplets.json', 'w') as f:
    json.dump(conf_trip, f)


# In[260]:


with open ('confidence_pairs.json', 'w') as f:
    json.dump(conf_pair, f)


# In[261]:


with open ('confidence_ent.json', 'w') as f:
    json.dump(conf_ent, f)


# In[238]:


with open('confidence_ent.json') as f:
    conf_ent = json.load(f)
with open('confidence_pairs.json') as f:
    conf_pair = json.load(f)
with open('confidence_triplets.json') as f:
    conf_trip = json.load(f)


# In[245]:


def classification_base (train, test, conf_trip, data):
    results = {}
    labels = data['label'].tolist()
    texts = list(test.keys())
    for tex in texts:
        results[str(tex)] = {'label': labels[int(tex)], 'total': 0, 'matches': 0, 'triplets': [], 'binary': -1, 'confidence': -1}
        
    knowledge_base = []
    for text in train:
        for trip in train[text]:
            if trip not in knowledge_base:
                knowledge_base.append(trip)
                
    for text in test:
        for trip in test[text]:
            results[text]['total'] = results[text]['total'] + 1
            if trip in knowledge_base:
                results[text]['triplets'].append(trip)
                results[text]['matches'] = results[text]['matches'] + 1
        
    for text in results:
        if results[text]['total'] == 0:
            results[text]['binary'] = 0
        else:
            co = 0
            for trip in results[text]['triplets']:
                co += conf_trip[trip]
            co_ = co/results[text]['total']
            
        if co >= 0.5:
            results[text]['binary'] = 2
        else:
            results[text]['binary'] = 1
    
    return results


# In[411]:


results_base = classification_base (trip_youtrain_lem, trip_youtest_lem, conf_trip, you_data)


# In[412]:


results_base_entact = classification_base (entact_youtrain_lem, entact_youtest_lem, conf_pair, you_data)


# In[24]:


results_base_ent = classification_base (ent_youtrain_lem, ent_youtest_lem, conf_ent, you_data)


# In[415]:


with open ('results_base.json', 'w') as f:
    json.dump(results_base, f)


# In[416]:


with open ('results_base_entact.json', 'w') as f:
    json.dump(results_base_entact, f)


# In[25]:


with open ('results_base_ent.json', 'w') as f:
    json.dump(results_base_ent, f)


# ### **Extract results - visualisations etc. - Binary**

# In[229]:


with open('results_transfer.json', 'r') as f:
    results_transfer = json.load(f)
with open('results_transfer_entact.json', 'r') as f:
    results_transfer_entact = json.load(f)
with open('results_base.json', 'r') as f:
    results_base = json.load(f)
with open('results_base_entact.json', 'r') as f:
    results_base_entact = json.load(f)


# In[266]:


def evaluation_binary (results, score_file, normal = 'no', write_to_file = 'no', output_only_conf = 'no'):
    true = []
    pred = []
    for text in results:
        if results[text]['label'] == 1:
            true.append('non-conspiratorial')
        else:
            true.append('conspiratorial')
        if results[text]['binary'] == 1:
            pred.append('non-conspiratorial')
        else:
            pred.append('conspiratorial')
    if normal == 'yes':        
        conf = confusion_matrix(true, pred, normalize = 'true')
    elif normal == 'no':
        conf = confusion_matrix(true, pred)
    
    f1 = f1_score(true, pred, pos_label = 'conspiratorial')
    acc = accuracy_score(true, pred) 
    prec = precision_score(true, pred, pos_label = 'conspiratorial')
    rec = recall_score(true, pred, pos_label = 'conspiratorial')
    scores = {'precision': prec, 'recall': rec, 'accuracy': acc, 'f1': f1}
    if write_to_file == 'yes':
        with open (score_file, 'w') as f:
            json.dump(scores, f)
    
    if output_only_conf == 'no':
        return scores, conf
    if output_only_conf == 'yes':
        return conf


# In[53]:


scores_transfer, conf_transfer = evaluation_binary(results_transfer, 'scores_transfer.json', write_to_file = 'yes')


# In[54]:


scores_transfer_entact, conf_transfer_entact = evaluation_binary(results_transfer_entact, 'scores_transfer_entact.json', write_to_file = 'yes')


# In[155]:


scores_transfer_ent, conf_transfer_ent = evaluation_binary(results_transfer_ent, 'scores_transfer_ent.json', write_to_file = 'yes')


# In[267]:


scores_base, conf_base = evaluation_binary(results_base,'scores_base.json', write_to_file = 'yes')


# In[57]:


scores_base_entact, conf_base_entact = evaluation_binary(results_base_entact, 'scores_base_entact.json', write_to_file = 'yes')


# In[156]:


scores_base_ent, conf_base_ent = evaluation_binary(results_base_ent, 'scores_base_ent.json', write_to_file = 'yes')


# In[234]:


match_trip = {'1': [], '2': []}
for text in results_transfer:
    if results_transfer[text]['binary'] == 2 and results_transfer[text]['label'] == 1:
        for x in results_transfer[text]['triplets']:
            match_trip['1'].append(x)
    if results_transfer[text]['binary'] == 2 and results_transfer[text]['label'] == 2:
        for x in results_transfer[text]['triplets']:
            match_trip['2'].append(x)
print(match_trip)


# In[237]:


for text in results_transfer:
    if results_transfer[text]['binary'] == 2 and results_transfer[text]['label'] == 2:
        print(text)


# In[11]:


with open('results_transfer_entact.json', 'r') as f:
    results_entact = json.load(f)
with open('results_transfer_ent.json', 'r') as f:
    results_ent = json.load(f)


# In[9]:


matches = 0
for text in results_entact:
    matches += results_entact[text]['matches']
print(matches)
print(matches/len(results_entact))


# In[12]:


matches = 0
for text in results_ent:
    matches += results_ent[text]['matches']
print(matches)
print(matches/len(results_ent))


# #### **Printed results**

# In[59]:


print('Scores transfer triplets: precision: {0}, recall: {1}, accuracy: {2}, f1-score: {3}'.format(round(scores_transfer['precision'],3), round(scores_transfer['recall'],3), round(scores_transfer['accuracy'],3), round(scores_transfer['f1']),))


# In[60]:


print('Scores transfer pair: precision: {0}, recall: {1}, accuracy: {2}, f1-score: {3}'.format(round(scores_transfer_entact['precision'],3), round(scores_transfer_entact['recall'],3), round(scores_transfer_entact['accuracy'],3), round(scores_transfer_entact['f1']),3))


# In[157]:


print('Scores transfer entities: precision: {0}, recall: {1}, accuracy: {2}, f1-score: {3}'.format(round(scores_transfer_ent['precision'],3), round(scores_transfer_ent['recall'],3), round(scores_transfer_ent['accuracy'],3), round(scores_transfer_ent['f1']),3))


# In[62]:


print('Scores base triplets: precision: {0}, recall: {1}, accuracy: {2}, f1-score: {3}'.format(round(scores_base['precision'],3), round(scores_base['recall'],3), round(scores_base['accuracy'],3), round(scores_base['f1'],3)))


# In[63]:


print('Scores base pair: precision: {0}, recall: {1}, accuracy: {2}, f1-score: {3}'.format(round(scores_base_entact['precision'],3), round(scores_base_entact['recall'],3), round(scores_base_entact['accuracy'],3), round(scores_base_entact['f1'],3)))


# In[158]:


print('Scores base entities: precision: {0}, recall: {1}, accuracy: {2}, f1-score: {3}'.format(round(scores_base_ent['precision'],3), round(scores_base_ent['recall'],3), round(scores_base_ent['accuracy'],3), round(scores_base_ent['f1'],3)))


# #### **Confusion matrices**

# In[164]:


x_labels = ['conspiratorial', 'non-conspiratorial']
y_labels = ['conspiratorial', 'non-conspiratorial']

plt.figure(figsize=(16, 6))
plt.subplot(131)
sns.heatmap(conf_transfer_ent, annot = True, fmt=".0f", linewidths=1, square = True, cmap = "Blues", cbar = False, xticklabels = x_labels, yticklabels = y_labels);
plt.ylabel('True label', labelpad = 10);
plt.xlabel('Predicted label', labelpad = 10)
plt.title('Confusion matrix - entities transfer learning', pad = 15)
plt.subplot(132)
sns.heatmap(conf_transfer_entact, annot = True, fmt=".0f", linewidths=1, square = True, cmap = "Blues", cbar = False, xticklabels = x_labels, yticklabels = y_labels);
plt.ylabel('True label', labelpad = 10)
plt.xlabel('Predicted label', labelpad = 10)
plt.title('Confusion matrix - pairs transfer learning', pad = 15),
plt.subplot(133)
sns.heatmap(conf_transfer, annot = True, fmt=".0f", linewidths=1, square = True, cmap = "Blues", cbar = False, xticklabels = x_labels, yticklabels = y_labels);
plt.ylabel('True label', labelpad = 10)
plt.xlabel('Predicted label', labelpad = 10)
plt.title('Confusion matrix - triplets transfer learning', pad = 15),
plt.savefig('Confusion matrix transfer')
plt.show()


# In[109]:


8+3+250+340


# In[110]:


217+204+41+139


# In[111]:


254+4+30+313


# In[165]:


x_labels = ['conspiratorial', 'non-conspiratorial']
y_labels = ['conspiratorial', 'non-conspiratorial']

plt.figure(figsize=(16, 6))
plt.subplot(131)
sns.heatmap(conf_base_ent, annot = True, fmt=".0f", linewidths=1, square = True, cmap = "Blues", cbar = False, xticklabels = x_labels, yticklabels = y_labels);
plt.ylabel('True label', labelpad = 10);
plt.xlabel('Predicted label', labelpad = 10)
plt.title('Confusion matrix - entities baseline', pad = 15)
plt.subplot(132)
sns.heatmap(conf_base_entact, annot = True, fmt=".0f", linewidths=1, square = True, cmap = "Blues", cbar = False, xticklabels = x_labels, yticklabels = y_labels);
plt.ylabel('True label', labelpad = 10)
plt.xlabel('Predicted label', labelpad = 10)
plt.title('Confusion matrix - pairs baseline', pad = 15),
plt.subplot(133)
sns.heatmap(conf_base, annot = True, fmt=".0f", linewidths=1, square = True, cmap = "Blues", cbar = False, xticklabels = x_labels, yticklabels = y_labels);
plt.ylabel('True label', labelpad = 10)
plt.xlabel('Predicted label', labelpad = 10)
plt.title('Confusion matrix - triplets baseline', pad = 15),
plt.savefig('Confusion matrix base')
plt.show()


# In[112]:


### one text without triplet additionally


# In[106]:


53+65+31+47


# In[ ]:


### two texts without entity


# In[107]:


71+13+99+14


# In[108]:


82+104+2+9


# In[168]:


c = 0
n = 0
for text in results_base:
    if results_base[text]['label'] == 1:
        n += 1
    else:
        c += 1
print(n)
print(c)


# In[169]:


c = 0
n = 0
for text in results_base_entact:
    if results_base_entact[text]['label'] == 1:
        n += 1
    else:
        c += 1
print(n)
print(c)


# In[170]:


c = 0
n = 0
for text in results_base_ent:
    if results_base_ent[text]['label'] == 1:
        n += 1
    else:
        c += 1
print(n)
print(c)


# #### **Test**

# In[241]:


with open('entities_youtrain_lem.json') as f:
    ent_youtrain_lem = json.load(f)
with open('entities_youtest_lem.json') as f:
    ent_youtest_lem = json.load(f)
with open('entities_action_youtrain_lem.json') as f:
    entact_youtrain_lem = json.load(f)
with open('entities_action_youtest_lem.json') as f:
    entact_youtest_lem = json.load(f)
with open('triplets_youtrain_lem02.json') as f:
    trip_youtrain_lem = json.load(f)
with open('triplets_youtest_lem02.json') as f:
    trip_youtest_lem = json.load(f)


# In[245]:


def classification_base (train, test, conf_trip, data):
    results = {}
    labels = data['label'].tolist()
    texts = list(test.keys())
    for tex in texts:
        results[str(tex)] = {'label': labels[int(tex)], 'total': 0, 'matches': 0, 'triplets': [], 'binary': -1, 'confidence': -1}
        
    knowledge_base = []
    for text in train:
        for trip in train[text]:
            if trip not in knowledge_base:
                knowledge_base.append(trip)
                
    for text in test:
        for trip in test[text]:
            results[text]['total'] = results[text]['total'] + 1
            if trip in knowledge_base:
                results[text]['triplets'].append(trip)
                results[text]['matches'] = results[text]['matches'] + 1
        
    for text in results:
        if results[text]['total'] == 0:
            results[text]['binary'] = 0
        else:
            co = 0
            for trip in results[text]['triplets']:
                co += conf_trip[trip]
            co_ = co/results[text]['total']
            
        if co >= 0.5:
            results[text]['binary'] = 2
        else:
            results[text]['binary'] = 1
    
    return results


# In[262]:


results_base = classification_base (trip_youtrain_lem, trip_youtest_lem, conf_trip, you_data)


# In[263]:


results_base_entact = classification_base (entact_youtrain_lem, entact_youtest_lem, conf_pair, you_data)


# In[264]:


results_base_ent = classification_base (ent_youtrain_lem, ent_youtest_lem, conf_ent, you_data)


# In[268]:


scores_base, conf_base = evaluation_binary(results_base,'scores_base.json', write_to_file = 'no')


# In[269]:


scores_base_entact, conf_base_entact = evaluation_binary(results_base_entact, 'scores_base_entact.json', write_to_file = 'no')


# In[270]:


scores_base_ent, conf_base_ent = evaluation_binary(results_base_ent, 'scores_base_ent.json', write_to_file = 'no')


# In[271]:


print('Scores base triplets: precision: {0}, recall: {1}, accuracy: {2}, f1-score: {3}'.format(round(scores_base['precision'],3), round(scores_base['recall'],3), round(scores_base['accuracy'],3), round(scores_base['f1'],3)))


# In[272]:


print('Scores base pair: precision: {0}, recall: {1}, accuracy: {2}, f1-score: {3}'.format(round(scores_base_entact['precision'],3), round(scores_base_entact['recall'],3), round(scores_base_entact['accuracy'],3), round(scores_base_entact['f1'],3)))


# In[273]:


print('Scores base entities: precision: {0}, recall: {1}, accuracy: {2}, f1-score: {3}'.format(round(scores_base_ent['precision'],3), round(scores_base_ent['recall'],3), round(scores_base_ent['accuracy'],3), round(scores_base_ent['f1'],3)))


# In[275]:


x_labels = ['conspiratorial', 'non-conspiratorial']
y_labels = ['conspiratorial', 'non-conspiratorial']

plt.figure(figsize=(16, 6))
plt.subplot(131)
sns.heatmap(conf_base_ent, annot = True, fmt=".0f", linewidths=1, square = True, cmap = "Blues", cbar = False, xticklabels = x_labels, yticklabels = y_labels);
plt.ylabel('True label', labelpad = 10);
plt.xlabel('Predicted label', labelpad = 10)
plt.title('Confusion matrix - entities baseline', pad = 15)
plt.subplot(132)
sns.heatmap(conf_base_entact, annot = True, fmt=".0f", linewidths=1, square = True, cmap = "Blues", cbar = False, xticklabels = x_labels, yticklabels = y_labels);
plt.ylabel('True label', labelpad = 10)
plt.xlabel('Predicted label', labelpad = 10)
plt.title('Confusion matrix - pairs baseline', pad = 15),
plt.subplot(133)
sns.heatmap(conf_base, annot = True, fmt=".0f", linewidths=1, square = True, cmap = "Blues", cbar = False, xticklabels = x_labels, yticklabels = y_labels);
plt.ylabel('True label', labelpad = 10)
plt.xlabel('Predicted label', labelpad = 10)
plt.title('Confusion matrix - triplets baseline', pad = 15),
plt.savefig('Confusion matrix base')
plt.show()


# #### **Counts for entities, pairs and triplets**

# In[171]:


with open('entities_red_lem.json', 'r') as f:
    entities_red = json.load(f)
with open('entities_you_lem.json', 'r') as f:
    entities_you = json.load(f)
with open('entities_youtrain_lem.json', 'r') as f:
    entities_youtrain = json.load(f)
with open('entities_youtest_lem.json', 'r') as f:
    entities_youtest = json.load(f)


# In[172]:


ents_red = []
ents_you = []
ents_youtrain = []
ents_youtest = []
for text in entities_red:
    for ent in entities_red[text]:
        if ent not in ents_red:
            ents_red.append(ent)
for text in entities_you:
    for ent in entities_you[text]:
        if ent not in ents_you:
            ents_you.append(ent)
for text in entities_youtrain:
    for ent in entities_youtrain[text]:
        if ent not in ents_youtrain:
            ents_youtrain.append(ent)
for text in entities_youtest:
    for ent in entities_youtest[text]:
        if ent not in ents_youtest:
            ents_youtest.append(ent)


# In[173]:


print(len(ents_red))
print(len(ents_you))
print(len(ents_youtrain))
print(len(ents_youtest))


# In[174]:


with open('entities_action_red_lem.json', 'r') as f:
    entact_red = json.load(f)
with open('entities_action_you_lem.json', 'r') as f:
    entact_you = json.load(f)
with open('entities_action_youtrain_lem.json', 'r') as f:
    entact_youtrain = json.load(f)
with open('entities_action_youtest_lem.json', 'r') as f:
    entact_youtest = json.load(f)


# In[175]:


enta_red = []
enta_you = []
enta_youtrain = []
enta_youtest = []
for text in entact_red:
    for ent in entact_red[text]:
        if ent not in enta_red:
            enta_red.append(ent)
for text in entact_you:
    for ent in entact_you[text]:
        if ent not in enta_you:
            enta_you.append(ent)
for text in entact_youtrain:
    for ent in entact_youtrain[text]:
        if ent not in enta_youtrain:
            enta_youtrain.append(ent)
for text in entact_youtest:
    for ent in entact_youtest[text]:
        if ent not in enta_youtest:
            enta_youtest.append(ent)


# In[176]:


print(len(enta_red))
print(len(enta_you))
print(len(enta_youtrain))
print(len(enta_youtest))


# In[179]:


with open('triplets_red_lem02.json', 'r') as f:
    triplets_red = json.load(f)
with open('triplets_you_lem02.json', 'r') as f:
    triplets_you = json.load(f)
with open('triplets_youtrain_lem02.json', 'r') as f:
    triplets_youtrain = json.load(f)
with open('triplets_youtest_lem02.json', 'r') as f:
    triplets_youtest = json.load(f)


# In[180]:


trips_red = []
trips_you = []
trips_youtrain = []
trips_youtest = []
for text in triplets_red:
    for ent in triplets_red[text]:
        if ent not in trips_red:
            trips_red.append(ent)
for text in triplets_you:
    for ent in triplets_you[text]:
        if ent not in trips_you:
            trips_you.append(ent)
for text in triplets_youtrain:
    for ent in triplets_youtrain[text]:
        if ent not in trips_youtrain:
            trips_youtrain.append(ent)
for text in triplets_youtest:
    for ent in triplets_youtest[text]:
        if ent not in trips_youtest:
            trips_youtest.append(ent)


# In[181]:


print(len(enta_red))
print(len(enta_you))
print(len(enta_youtrain))
print(len(enta_youtest))


# ### **Evaluation confidence**

# In[213]:


def evaluation_conf (results, score_file, normal = 'no', write_to_file = 'no', output_only_conf = 'no'):
    true = []
    pred = []
    for text in results:
        if results[text]['label'] == 1:
            true.append('non-conspiratorial')
        else:
            true.append('conspiratorial')
        if results[text]['confidence'] < 0.5:
            pred.append('non-conspiratorial')
        else:
            pred.append('conspiratorial')
    if normal == 'yes':        
        conf = confusion_matrix(true, pred, normalize = 'true')
    elif normal == 'no':
        conf = confusion_matrix(true, pred)
    
    f1 = f1_score(true, pred, pos_label = 'conspiratorial')
    prec = precision_score(true, pred, pos_label = 'conspiratorial')
    rec = recall_score(true, pred, pos_label = 'conspiratorial')
    scores = {'precision': prec, 'recall': rec, 'f1': f1}
    if write_to_file == 'yes':
        with open (score_file, 'w') as f:
            json.dump(scores, f)
    
    if output_only_conf == 'no':
        return scores, conf
    if output_only_conf == 'yes':
        return conf


# In[214]:


scores_transfer_conf, conf_transfer_conf = evaluation_conf(results_transfer, 'scores_transfer_conf.json', write_to_file = 'yes')


# In[215]:


scores_transfer_entact_conf, conf_transfer_entact_conf = evaluation_conf(results_transfer_entact, 'scores_transfer_entact_conf.json', write_to_file = 'yes')


# In[216]:


scores_transfer_ent_conf, conf_transfer_ent_conf = evaluation_conf(results_transfer_ent, 'scores_transfer_ent_conf.json', write_to_file = 'yes')


# In[217]:


scores_base_conf, conf_base_conf = evaluation_conf(results_base,'scores_base_conf.json', write_to_file = 'yes')


# In[218]:


scores_base_entact_conf, conf_base_entact_conf = evaluation_conf(results_base_entact, 'scores_base_entact_conf.json', write_to_file = 'yes')


# In[219]:


scores_base_ent_conf, conf_base_ent_conf = evaluation_conf(results_base_ent, 'scores_base_ent.json_conf', write_to_file = 'yes')


# #### **Printed results**

# In[220]:


print('Scores transfer triplets: precision: {0}, recall: {1}, f1-score: {2}'.format(round(scores_transfer_conf['precision'],3), round(scores_transfer_conf['recall'],3), round(scores_transfer_conf['f1']),))


# In[221]:


print('Scores transfer pair: precision: {0}, recall: {1}, f1-score: {2}'.format(round(scores_transfer_entact_conf['precision'],3), round(scores_transfer_entact_conf['recall'],3), round(scores_transfer_entact_conf['f1']),3))


# In[222]:


print('Scores transfer entities: precision: {0}, recall: {1}, f1-score: {2}'.format(round(scores_transfer_ent_conf['precision'],3), round(scores_transfer_ent_conf['recall'],3), round(scores_transfer_ent_conf['f1']),3))


# In[223]:


print('Scores base triplets: precision: {0}, recall: {1}, f1-score: {2}'.format(round(scores_base_conf['precision'],3), round(scores_base_conf['recall'],3), round(scores_base_conf['f1'],3)))


# In[224]:


print('Scores base pair: precision: {0}, recall: {1}, f1-score: {2}'.format(round(scores_base_entact_conf['precision'],3), round(scores_base_entact_conf['recall'],3), round(scores_base_entact_conf['f1'],3)))


# In[225]:


print('Scores base entities: precision: {0}, recall: {1}, f1-score: {2}'.format(round(scores_base_ent_conf['precision'],3), round(scores_base_ent_conf['recall'],3), round(scores_base_ent_conf['f1'],3)))


# #### **Confusion matrices**

# In[226]:


x_labels = ['conspiratorial', 'non-conspiratorial']
y_labels = ['conspiratorial', 'non-conspiratorial']

plt.figure(figsize=(16, 6))
plt.subplot(131)
sns.heatmap(conf_transfer_ent_conf, annot = True, fmt=".0f", linewidths=1, square = True, cmap = "Blues", cbar = False, xticklabels = x_labels, yticklabels = y_labels);
plt.ylabel('True label', labelpad = 10);
plt.xlabel('Predicted label', labelpad = 10)
plt.title('Confusion matrix - entities transfer learning', pad = 15)
plt.subplot(132)
sns.heatmap(conf_transfer_entact_conf, annot = True, fmt=".0f", linewidths=1, square = True, cmap = "Blues", cbar = False, xticklabels = x_labels, yticklabels = y_labels);
plt.ylabel('True label', labelpad = 10)
plt.xlabel('Predicted label', labelpad = 10)
plt.title('Confusion matrix - pairs transfer learning', pad = 15),
plt.subplot(133)
sns.heatmap(conf_transfer_conf, annot = True, fmt=".0f", linewidths=1, square = True, cmap = "Blues", cbar = False, xticklabels = x_labels, yticklabels = y_labels);
plt.ylabel('True label', labelpad = 10)
plt.xlabel('Predicted label', labelpad = 10)
plt.title('Confusion matrix - triplets transfer learning', pad = 15),
#plt.savefig('Confusion matrix transfer')
plt.show()


# In[228]:


#x_labels = ['conspiratorial', 'non-conspiratorial']
#y_labels = ['conspiratorial', 'non-conspiratorial']

plt.figure(figsize=(16, 6))
plt.subplot(131)
sns.heatmap(conf_base_ent_conf, annot = True, fmt=".0f", linewidths=1, square = True, cmap = "Blues", cbar = False);
plt.ylabel('True label', labelpad = 10);
plt.xlabel('Predicted label', labelpad = 10)
plt.title('Confusion matrix - entities baseline', pad = 15)
plt.subplot(132)
sns.heatmap(conf_base_entact_conf, annot = True, fmt=".0f", linewidths=1, square = True, cmap = "Blues", cbar = False);
plt.ylabel('True label', labelpad = 10)
plt.xlabel('Predicted label', labelpad = 10)
plt.title('Confusion matrix - pairs baseline', pad = 15),
plt.subplot(133)
sns.heatmap(conf_base_conf, annot = True, fmt=".0f", linewidths=1, square = True, cmap = "Blues", cbar = False);
plt.ylabel('True label', labelpad = 10)
plt.xlabel('Predicted label', labelpad = 10)
plt.title('Confusion matrix - triplets baseline', pad = 15),
#plt.savefig('Confusion matrix base')
plt.show()


# In[ ]:




