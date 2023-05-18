![versions](https://img.shields.io/badge/python-3.above-blue.svg)
# Recommendation

The objective of this repository is to address the challenges of **upselling** and **cross-selling** in the context of personalized recommendations.

## Data

The data used in this project is a sample from the H&M Personalized Fashion Recommendations Kaggle competition. The dataset can be accessed at the following link: [Kaggle Competition - H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data)

## Files

* `transactions_train.csv`: This file contains user purchase history.
* `articles.csv`: This file contains detailed metadata for each available article_id for purchase.
* `customers.csv`: This file contains metadata for each customer_id in the dataset.

## Approach

The solution to this problem is divided into two main components:

1. `Relevance`: In this stage, a pool of products that are relevant to the Product Display Page (PDP), Cart, or previously bought products needs to be identified.

2. `Ranking`: Once the relevant products have been shortlisted, they need to be ranked to provide personalized recommendations to users.

## Relevance

### Upselling 
The goal is to find similar products to the one in question. To achieve this, product embeddings are constructed and cosine similarity is utilized to identify the N Nearest Neighbors. Two approaches for creating embeddings are explored in the provided notebooks: using a pre-trained transformer model and training a transformer model from scratch. Although any Language Model (LM) or custom-built transformer could be used, BERT was chosen for this task. A result comparison between the two approaches is available in the repository.

### Cross-Selling
Multiple approaches were employed and then combined in an ensemble to address this aspect.

- `Co-occurrence`: This approach considers the frequency at which two products are purchased together, capturing their association.
- `Collaborative Filtering`: Similar to co-occurrence, this approach also takes into account the individual occurrence of the two products to enhance the recommendations.
- `User-User Similarity`: By utilizing product embeddings, embeddings for users are generated. Once the embedding representation for each user is obtained, the most similar users to a given user can be identified, and recommendations can be made based on the items favored by those similar users.

## Ranking

To rank the products for each user, a LambdaRank Model is trained using the relevance scores as features, along with carefully designed user/item features. This approach aims to provide personalized ranking for optimal recommendations.

Please refer to the repository for further details and code implementation.