# User and Product-aware Sentiment Classification

### Introduction
Sentiment classification is a fundamental problem which aims to extract users’ overall sentiments about products from their reviews. The sentiment classification problem has been exhaustively studied since last decade where the proposed solutions primarily focus on local text of reviews to capture sentiments. However, the sentiments of a review depend not only on the context captured in local text, but also on other exogenous factors like who wrote the review and what product the review is written for. In this paper, we consider user preferences and product characteristics in addition to local text to improve sentiment classification. The challenge is how to incorporate user and product information along with the local text of a review for sentiment analysis. We propose a novel approach of enhancing Glove word embedding with user and product information  and applied the user-product specific embedding to train a hierarchical Bi-LSTM network with attention. The proposed solution is compared with a baseline model where user and product information is ignored from the hierarchical Bi-LSTM network. The performance of the proposed model is evaluated on the IMDB reviews while considering only user or product information. The proposed model improves the prediction accuracy compared to the baseline approach by 20% when only user information is considered and 10% when only product information is considered at the embedding layer. The performance can be further improved if both user and product information is considered at the embedding layer.



This repository contains the following files.
* req.txt: A requirement file containing dependent python libraries
* user_prod_text_classifier.py: Implementation of hierarchical bidirectional LSTM networks with attention and enhanced embedding layer
* user_text_classifier.py: Implementation of the model while considering only user information
* prod_text_classifier.py: Implementation of the model while considering only prod information
* custom_embedding.py: Implementation of two layer neural network to obtain user and product embedding
* create_prod_embedding.py: A python script to create product embedding
* create_user_embedding.py: A python script to create user embedding
* user<1-4>: A list of users
* prod<1-5>: A list of products
* prod_embedding: A folder containing the product embeddings of the products from IMDB database.
* user_embedding: A folder containing the user embeddings of the users from IMDB database.


### Process to train a model
```
# clone the repo
git clone {repo address}

# install Dependent library
cd textClassifier
pip install -r req.txt

# download imdb train from Kaggle in the below link and keep the files in the working directory
https://www.kaggle.com/c/word2vec-nlp-tutorial/download/labeledTrainData.tsv
# download glove word vector
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip

# install nltk 'punkt' using the following code in python interpretor
>>>import nltk
>>>nltk.download('punkt')

# train the model using both user and product information
python user_prod_text_classifier.py

# train the model using only user information
python user_text_classifier.py

# train the model using only product information
python prod_text_classifier.py

# note if in case while installing word2vec, cython error occurs then 
pip install --upgrade cython
```

### To create user and product embedding, apply the following steps

Run the below command to create custom embedding for products. Below process takes serial approach. It creates embedding for one prod and then moves to next one. We created 5 sets on products (from prod1 to prod5) files so that we can run this process on different machines in parallel to save time.

To run for prod2 to prod5 just change the argument in the below command.

```
python create_prod_embedding.py prod1
```

Similarly, run the below command to create custom embedding for users. Below process also takes serial approach. It creates embedding for one user and then moves to next one. We created 4 sets on users (from user1 to user4) files so that we can run this process on different machines in parallel to save time.

to run for user2 to user4 just change the argument in the below command.

```
python create_user_embedding.py user1
```


