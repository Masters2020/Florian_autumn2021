#!/usr/bin/env python
# coding: utf-8

# ## **Set-Up**

# In[1]:


import os
import pandas as pd
import torch
import scipy
import numpy as np
import tensorflow as tf
import keras
import transformers
import time
import json


# In[2]:


from transformers import set_seed
set_seed(42)


# In[3]:


from datasets import load_dataset
from datasets import Dataset
from transformers import AutoTokenizer


# In[4]:


path = os.path.abspath('06_embeddings.ipynb')  #this saves the path to this notebook and uses it as a base to access the required csvs
dname = os.path.dirname(path)
os.chdir(dname)


# ## **Import text**

# ### **Reddit**

# In[5]:


from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('flboehm/reddit-bert-text_10', model_max_length = 512, padding_side = 'right')


# In[6]:


print(len(tokenizer))


# In[7]:


from transformers import AutoModelForMaskedLM
model = AutoModelForMaskedLM.from_pretrained('flboehm/reddit-bert-text_10', output_hidden_states = True)


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


# In[17]:


print(len(inputs[141]))


# ### **Youtube**

# In[16]:


from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('flboehm/youtube-bert', model_max_length = 512, padding_side = 'right')


# In[17]:


print(len(tokenizer))


# In[18]:


from transformers import AutoModelForMaskedLM
model = AutoModelForMaskedLM.from_pretrained('flboehm/youtube-bert', output_hidden_states = True)


# In[19]:


you_text = dname + '\\data\\you_text_label.csv' #exported as one from the notebook preprocessing


# In[20]:


youtext_df = pd.read_csv(you_text, skipinitialspace=True)


# In[21]:


youtext_dict = youtext_df.to_dict(orient = 'index')
print(youtext_dict[0])


# In[22]:


inputs_you = []
for i in youtext_dict:
    inputs_you.append(youtext_dict[i]['text'])


# In[23]:


print(inputs_you[600])


# ## **Create Embeddings file**

# In[20]:


import os
import zipfile
from nltk import tokenize

def getembeddings (filename, inputs):
    with zipfile.ZipFile(filename, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        for i in range(len(inputs)):
            if len(inputs[i]) <= 512:
                input_ = tokenizer(inputs[i], add_special_tokens=False, return_tensors = 'pt')
                output = model(**input_)
                embeddings = output['hidden_states']
                embeddings = torch.stack(embeddings, dim=0)
                embeddings = torch.squeeze(embeddings, dim=1)
                embeddings = embeddings[11]
                final = embeddings.detach().numpy()

                tmpfilename = "text{}.npy".format(i)
                np.save(tmpfilename, final)
                zf.write(tmpfilename)
                os.remove(tmpfilename)
            
            else:
                sentences = tokenize.sent_tokenize(inputs[i])
                sent_clean = []
                indeces = []
                ind = []
                length = 0

                for a in sentences:
                    if len(a) > 512:
                        ind_ = []
                        indeces_ = []
                        length_ = 0
                        new = a.split(' ')
                        for b in range(len(new)):
                            if b != len(new)-1:
                                if length_ + len(new[b]) < 512:
                                    length_ += len(new[b])
                                    ind_.append(b)
                                else:
                                    indeces_.append(ind_)
                                    length_ = len(new[b])
                                    ind_ = [b]
                            else:
                                ind_.append(b)
                                indeces_.append(ind_)

                        for c in indeces_:
                            s = ''
                            for d in c:
                                add = ' ' + new[d]
                                s += add
                            sent_clean.append(s)
                    else:
                        sent_clean.append(a)
                for j in range(len(sent_clean)):
                    if j != len(sent_clean)-1:
                        if length + len(sent_clean[j]) < 512:
                            length += len(sent_clean[j])
                            ind.append(j)
                        else:
                            indeces.append(ind)
                            length = len(sent_clean[j])
                            ind = [j]
                    else:
                        ind.append(j)
                        indeces.append(ind)

                collect = []
                for k in indeces:
                    if not k:
                        pass
                    else:
                        strin = ''
                        for l in k:
                            strin += sent_clean[l]
                        collect.append(strin)
                    
                for m in range(len(collect)):
                    if m == 0:
                        input_ = tokenizer(collect[m], add_special_tokens=False, return_tensors = 'pt')
                        output = model(**input_)
                        embeddings = output['hidden_states']
                        embeddings = torch.stack(embeddings, dim=0)
                        embeddings = torch.squeeze(embeddings, dim=1)
                        embeddings = embeddings[11]
                        inter = embeddings.detach().numpy()
                        final = np.expand_dims(inter, axis = 0)

                    else:
                        input_ = tokenizer(collect[m], add_special_tokens=False, return_tensors = 'pt')
                        output = model(**input_)
                        embeddings = output['hidden_states']
                        embeddings = torch.stack(embeddings, dim=0)
                        embeddings = torch.squeeze(embeddings, dim=1)
                        embeddings = embeddings[11]
                        inter = embeddings.detach().numpy()
                        final_stack = np.expand_dims(inter, axis = 0)
                        final = np.hstack((final, final_stack))

                final_append = np.squeeze(final, axis = 0)

                tmpfilename = "text{}.npy".format(i)
                np.save(tmpfilename, final_append)
                zf.write(tmpfilename)
                os.remove(tmpfilename)


# In[21]:


getembeddings ('E:\\embeddings_red.npz', inputs)


# In[24]:


getembeddings ('E:\\embeddings_you.npz', inputs_you)

