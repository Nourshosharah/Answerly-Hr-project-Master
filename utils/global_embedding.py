#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from langchain_huggingface import HuggingFaceEmbeddings
# from utils.load_config import LoadConfig

# APPCFG = LoadConfig()
GLOBAL_EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="/home/rangpt/Documents/LLM_Models/multilingual-e5-large",
    model_kwargs={"device": "cpu","trust_remote_code": True},  # use GPU
    encode_kwargs={"normalize_embeddings": True}, # E5 models require normalized embeddings
)


