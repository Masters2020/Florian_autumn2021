#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#personal use script: fSVRf6__Hn7ty1Y5DP-aow
#secret key: WwC1P-p1gpdmmVwVvWYQk4gHnWof1g
#source: https://www.storybench.org/how-to-scrape-reddit-with-python/


# In[16]:


import praw, prawcore
import pandas as pd
import time
import datetime


# In[2]:


#set up the scrapper app in python


# In[3]:


reddit = praw.Reddit(client_id='fSVRf6__Hn7ty1Y5DP-aow', client_secret='WwC1P-p1gpdmmVwVvWYQk4gHnWof1g', user_agent='Scrapping', username='ENTER HERE', password='ENTER HERE')


# In[4]:


#scrap the subreddits


# In[5]:


subs = ['conspiracy', 'conspiracytheories', 'conspiracyNOPOL']
organizer = ['top', 'hot', 'new', 'controversial'] #these are all legal values for the organizer 


# In[17]:


#
#          Florian Böhm 2021 v.01 - 01.12.2021
#
#
def scrap (subs, organizer, limit, mode): #subs and organizer have to be lists, limit an integer in the range 1 to 1000, mode a string either 'update' or 'new'
    #either creates a new dictionary or imports a csv file (in the same folder as this notebook!) and converts it to a dictionary which will be expanded
    if mode == 'update':
        oldlen = len(data['id'])
        data_df = pd.read_csv('reddit.csv')
        data = data_df.to_dict()
    if mode == 'new':
        data = {'id': [], 'text': [], 'selftext': [], 'subreddit': [], 'organizer': [], 'selftext': [], 'link': [], 'content url': [], 'created': []}
    
    #includes the submissions into the dictionary (new or update)
    for sub in subs:
        subr = reddit.subreddit(sub)
        for orga in organizer:
            if orga == 'top':
                subr_ = subr.top(limit = limit)
            elif orga == 'hot':
                subr_ = subr.hot(limit = limit)
            elif orga == 'new':
                subr_ = subr.new(limit = limit)
            elif orga == 'controversial':
                subr_ = subr.controversial(limit = limit)
            else:
                print('\nThe selected organizer does not exist and will therefore be skipped. Please only use the four valid organizers top, hot, new and controversial.')
                print('\tsub:', sub)
                print('\torganizer:', orga)
                break
            start = time.time()
            try:
                for submission in subr_:
                    if submission.id not in data['id']:
                        data['text'].append(str(submission.title) + '. ' + str(submission.selftext))
                        
                        data['id'].append(submission.id)
                        data['subreddit'].append(sub)
                        data['organizer'].append(orga)
                        data['selftext'].append(submission.is_self)
                        data['link'].append('reddit.com' + str(submission.permalink))
                        data['content url'].append(submission.url)
                        data['created'].append(datetime.datetime.fromtimestamp(submission.created_utc).strftime('%Y-%m-%d %H:%M:%S'))
                    if submission.id in data['id']:
                        pass
                stop = time.time()
                print('\nScrapping for', sub, orga, 'done with', stop-start, 'seconds')
            except UnboundLocalError: #because of how the function is build, this has to be included so the wrong organizer if included is actually skipped
                pass
            except prawcore.exceptions.Redirect:
                print('\nThe selected subreddit could not be found and will therefore be skipped. This can happen if either the subreddit does not exist, is not public or there is a typo.')
                print('\tsub:', sub)
                print('\torganizer:', orga)
                pass
    
    #some info on how many submission were included for an update or new scrapping
    if mode == 'update':
        newlen = len(data['id'])
        print('\nNew submissions included:', newlen - oldlen)
    if mode == 'new':
        print('\nSubmissions scrapped:', len(data['id']))
    
    #create dataframe to be returned
    reddit_df = pd.DataFrame(data)
    reddit_df.drop_duplicates(subset = ['text'], inplace = True) #to make sure that posts that were just copied and pasted by other users or into different subreddits are not included twice
    
    
    #check if some duplicates slipped through
    dupli = reddit_df[reddit_df.duplicated(['id'])]
    if dupli.empty == True:
        print('\nDuplicate check passed')
    if dupli.empty == False:
        print('\nDuplicate check failed. Either check if code correct and run again or remove manually')
    
    #export dataframe to csv
    reddit_df.to_csv('reddit.csv')
    
    return reddit_df


# In[18]:


reddit_df = scrap(subs, organizer, 1000, 'new')

