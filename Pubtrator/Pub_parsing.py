#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This is a sample Python script.
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
import requests
from bs4 import BeautifulSoup as bs


# In[10]:


search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=COVID-19%20AND%20Merck%20AND%202020%202021[pdat]"
resp = requests.get(search_url).text
bs_obj = bs(resp,features="lxml")
id_list = bs_obj.find_all("id")   # idlist
print(id_list)


# In[41]:


# print(id_list)
id_temp_list = []
for iden in id_list:
    id_temp_list.append(iden.text)
id_string = ",".join(id_temp_list)
print(id_string)

fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id="+id_string+"&rettype=xml"
fetch_resp = requests.get(fetch_url).text
print(type(fetch_resp))
fetch_obj = bs(fetch_resp,features="lxml")


# In[125]:


iden = fetch_obj.find_all("pmid")
id_list=[i.contents[0] for i in iden]
# print(id_list)
titles = fetch_obj.find_all("articletitle")
title_list=[i.contents[0] for i in titles]
# print(title_list)

pub_date = fetch_obj.find_all("daterevised")
pub_date_list=[i.contents[5].next_element+'-'+i.contents[3].next_element+"-"+i.contents[1].next_element for i in pub_date]
print(pub_date_list)

Data_dict={'PMID':id_list,'Title':title_list,'Date':pub_date_list}
print(Data_dict)
pub_df=pd.DataFrame.from_dict(data=Data_dict,orient='index').transpose()


# In[126]:

pub_df

