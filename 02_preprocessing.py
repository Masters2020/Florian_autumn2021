#!/usr/bin/env python
# coding: utf-8

# ## **Setup**

# In[1]:


import os
import re
from nltk import tokenize
import emoji
from sklearn.model_selection import train_test_split
import pandas as pd


# In[2]:


path = os.path.abspath('02_preprocessing.ipynb')  #this saves the path to this notebook and uses it as a base to access the required csvs
dname = os.path.dirname(path)
os.chdir(dname)


# ## **Preprocessing**

# ### **Reddit**

# In[7]:


#this code requires the csvs to be placed in a subfolder called data within the folder in which this notebook is placed
reddit = dname + '\\data\\reddit.csv'


# In[8]:


red_df = pd.read_csv(reddit)


# In[19]:


def preprocessing_reddit (file, column, texttokenize = 'no'):
    text = list(file[column])
    dic = {}
    
    for index, value in enumerate(text):
        a = re.sub(r'\.+', '.', value)
        b = re.sub(r'\?+', '?', a)
        c = re.sub(r'!+', '!', b)
        d = re.sub(r'\?+\.+', '?', c)
        e = re.sub(r'\?+!+', '?', d)
        f = re.sub(r'!+\.+', '!', e)
        g = re.sub(r'!+\?+', '!', f)
        h = re.sub(r'\.+\?+', '.', g)
        i = re.sub(r'\.+!+', '.', h)
        j = re.sub(r'http\S+', '', i) #source: https://gist.github.com/MrEliptik/b3f16179aa2f530781ef8ca9a16499af
        k = re.sub(r'\(\s{1}', ' ', j)
        l = k.replace('\n', ' ')
        m = l.replace('“', ' ') #needs to specify the exact doublequote used (had to copy and paste it into here)
        n = m.replace('”', ' ')
        o = n.replace('"', ' ')
        p = re.sub(r'\s{1}\.{1}', '\.', o)
        q = emoji.demojize(p)
        r = re.sub(r':\w+:', ' ', q)
        s = r.replace('\\', ' ')
        t = s.replace('*', ' ')
        u = t.replace('[', ' ')
        v = u.replace('(', ' ')
        w = v.replace(']', ' ')
        x = w.replace(')', ' ')
        y = x.replace('~', ' ')
        clean = re.sub(r'\s+', ' ', y)
        if texttokenize == 'yes':
            sen = tokenize.sent_tokenize(clean) #results in list of sentences in the text and the list gets appended
            dic[index] = sen
        if texttokenize == 'no':
            dic[index] = clean #appends the whole text as one string
            
    return dic
#to be aware: using the sentence mode or the text mode without tokenizing results in a dictionary with key and strings as values, tokenizing the text into sentences results in a dictionary with keys and a list of the sentences as values


# In[20]:


red_dic = preprocessing(red_df, 'text')


# ### **Youtube**

# In[3]:


#this code requires the csvs to be placed in a subfolder called data within the folder in which this notebook is placed
youtube = dname + '\\data\\videos.csv'


# In[4]:


you_df = pd.read_csv(youtube)


# In[5]:


print(you_df)


# In[6]:


def preprocessing_youtube (file, column, column2):
    text = list(file[column])
    label = list(file[column2])
    dic = {}
    
    for index, value in enumerate(text):
        a = re.sub(r'\.+', '.', value)
        b = re.sub(r'\?+', '?', a)
        c = re.sub(r'!+', '!', b)
        d = re.sub(r'\?+\.+', '?', c)
        e = re.sub(r'\?+!+', '?', d)
        f = re.sub(r'!+\.+', '!', e)
        g = re.sub(r'!+\?+', '!', f)
        h = re.sub(r'\.+\?+', '.', g)
        i = re.sub(r'\.+!+', '.', h)
        j = re.sub(r'http\S+', '', i) #source: https://gist.github.com/MrEliptik/b3f16179aa2f530781ef8ca9a16499af
        k = re.sub(r'\(\s{1}', ' ', j)
        l = k.replace('\n', ' ')
        m = l.replace('“', ' ') #needs to specify the exact doublequote used (had to copy and paste it into here)
        n = m.replace('”', ' ')
        o = n.replace('"', ' ')
        p = re.sub(r'\s{1}\.{1}', '\.', o)
        q = p.replace('[Music]', '')
        r = q.replace('NARRATOR:', '')
        s = r.replace('\\', ' ')
        t = s.replace('*', ' ')
        u = t.replace('[', ' ')
        v = u.replace('(', ' ')
        w = v.replace(']', ' ')
        x = w.replace(')', ' ')
        y = x.replace('~', ' ')
        clean = re.sub(r'\s+', ' ', y)
        
        dic[index] = {}
        dic[index]['text'] = clean
        if label[index] == 3:
            dic[index]['label'] = 2
        else:
            dic[index]['label'] = label[index]
            
    return diC


# In[7]:


you_dic = preprocessing_youtube(you_df, 'transcripts', 'label')


# In[8]:


you_dic2 = {}
it = 0
for key in you_dic:
    if len(you_dic[key]['text']) > 10:
        you_dic2[it] = you_dic[key]
        it += 1


# In[9]:


print(len(you_dic2))


# In[10]:


print(you_dic2[0])


# ## **Create csv**

# In[16]:


#function to create datasets from the cleaned data to be loaded for the BERT Trainer API
#arguments include the dictionary created by the preprocessing, if a split of train and validation data should occur and if yes which size the validation split should be
#Also a name in string form should be supplied for the csv file to be named, the prefix of train or val will be included automatically, same for the csv appendix
def modeldata (dic, name, column, size = 0.2, split = 'yes'):
    data = pd.DataFrame.from_dict(dic, orient = 'index', columns = column)
    if split == 'yes':
        train, val = train_test_split(data, test_size = size, stratify = data['label']) #stratify only added for splitting youtube data into test and train
        print('train split size:', len(train))
        print('validation split size:', len(val))
        train.to_csv('train_' + name + '.csv', index_label = 'index')
        val.to_csv('val_' + name + '.csv', index_label = 'index')
        print('\nfiles created')
    if split == 'no':
        data.to_csv(name + '.csv', index_label = 'index')
        print('file created')


# In[22]:


modeldata(red_dic, 'red_text', ['text'], split = 'no')


# In[23]:


modeldata(red_dic, 'red_text', ['text'])


# In[80]:


modeldata(you_dic2, 'you_text', ['text', 'label'], split = 'no')


# In[18]:


modeldata(you_dic2, 'you_text', ['text', 'label'], size = 0.33)


# ### **Youtube without labels - fine-tuning BERT**

# In[3]:


you_t = dname + '\\data\\train_you_text.csv'
you_v = dname + '\\data\\val_you_text.csv'


# In[5]:


yout_df = pd.read_csv(you_t, skipinitialspace=True)
youv_df = pd.read_csv(you_v, skipinitialspace=True)


# In[6]:


yout_dict = yout_df.to_dict(orient = 'index')
youv_dict = youv_df.to_dict(orient = 'index')


# In[7]:


datat = pd.DataFrame.from_dict(yout_dict, orient = 'index', columns = ['text'])
datav = pd.DataFrame.from_dict(youv_dict, orient = 'index', columns = ['text'])


# In[8]:


datat.to_csv('train_you.csv', index = False),
datav.to_csv('val_you.csv', index = False)

