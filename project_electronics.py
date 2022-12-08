#!/usr/bin/env python
# coding: utf-8

# In[2]:


#get_ipython().system('pip install elasticsearch==7.17')

#pip install elasticsearch==7.17
# In[3]:


from elasticsearch import Elasticsearch


# In[4]:


#get_ipython().system('pip freeze|grep elasticsearch')


# In[5]:


es_client = Elasticsearch('https://my-deployment-ccde32.es.us-east4.gcp.elastic-cloud.com', http_auth=('elastic','ibi6dHbvjbMem8xbqxMknZgA'))

# In[6]:


es_client


# In[7]:


from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark import SparkContext 

spark = SparkSession.builder.config("spark.sql.caseSensitive", "true").getOrCreate()


# In[8]:


#spark.conf.set("spark.sql.caseSensitive", "true")


# In[9]:


data = spark.read.json("/user/sa6142_nyu_edu/project/electronics/Electronics.json")
temp1 = data.select("asin", "reviewerID", "summary", "overall")


# In[12]:


meta_data = spark.read.json("/user/sa6142_nyu_edu/project/electronics/meta_Electronics.json")
temp2 = meta_data.select("asin", "title", "brand")


# In[13]:


joined_data = temp1.join(temp2, "asin", "inner")


# In[21]:


joined_data.show(5)


# In[23]:


index_name = "amazon_electronics"
try:
    es_client.indices.delete(index=index_name)
except Exception as e:
    print(e)
index_body = {
      'settings': {
        'number_of_shards': 1,
        'number_of_replicas': 0,
        'analysis': {
          "filter":{  
            "english_stop":{
              "type":"stop",
              "stopwords":"english"
            },
            "english_stemmer":{
              "type":"stemmer",
              "language":"english"
            }
          },  
          "analyzer": {
            "stem_english": { 
              "type":"custom",
              "tokenizer":"standard",
              "filter":[
                "lowercase",
                "english_stop",
                "english_stemmer"
              ]
            }
        }
      }},
      'mappings': {
          'properties': {
              'asin' : {'type': 'text'},
              'reviewerID' : {'type': 'text'},
            'summary': {
                'type': 'text',
                'analyzer': 'stem_english'
            },
            'overall':  {'type': 'double'},
            'title': {
                'type': 'text',
                'analyzer': 'stem_english'
            },
            'brand': {
                'type': 'text',
                'analyzer': 'stem_english'
            },
            "profile_vector": {
              "type": "dense_vector",
              "dims": 48
            }
          }
      }
    }
es_client.indices.create(index=index_name,body=index_body)


# In[30]:


joined_data.count()


# In[29]:


from elasticsearch import helpers
import uuid
#items_frame = itemfactors.select('id','features').toPandas().rename(columns={"id": "movie_id", "features": "features"})
# join this with the original dataframe
#db_movies = movies.merge(items_frame, left_on='movieId', right_on='movie_id')
# create a dataset for Elasticsearch
small_df = joined_data.limit(5)
es_dataset = [{"_index": index_name, "_id": uuid.uuid4(), "_source" : {"title": doc[1]["title"], "summary": doc[1]["summary"],"asin": doc[1]["asin"], "reviewerID": doc[1]["reviewerID"], "overall": doc[1]["overall"] }} for doc in joined_data.toPandas().iterrows()]
#es_dataset = joined_data.map(x=> {"_index": index_name, "_id": uuid.uuid4(), "_source" : {"title": x["title"], "summary": x["summary"], "asin": x["asin"], "reviewerID": x["reviewerID"], "overall": x["overall"] }}).collect();
#bulk insert them
helpers.bulk(es_client, es_dataset)


# In[ ]:




