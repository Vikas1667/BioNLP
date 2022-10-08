### Biomedical Word2Vec with Gensim and Fastapi
### Data Sources 

https://bio.nlplab.org/#word-vectors
https://www.kaggle.com/code/bonhart/pubmed-abstracts-word-embeddings
https://www.nlm.nih.gov/databases/download/pubmed_medline.html
https://www.nlm.nih.gov/databases/download/mesh.html


## Fastapi References 
https://medium.com/mlearning-ai/fastapi-and-machine-learning-b75ac9c60412


#### Commands 

```
$ pip install fastapi
$ pip install "uvicorn[standard]"
```

```
uvicorn main:app --reload
```


```
http://127.0.0.1:8000/similarity?word1=diabetes&word2=diabetic
```
