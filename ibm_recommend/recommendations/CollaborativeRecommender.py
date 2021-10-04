import pandas as pd
import numpy as np

from ibm_recommend.recommendations import BaseRecommender
from ibm_recommend.matrix import create_user_item_matrix, get_user_articles

class CollaborativeRecommender(BaseRecommender):

    def __init__(self):
        super().__init__("collaborative")

    def _compute_similarities(self, user1, user2):
        user1_articles = self.matrix.loc[user1][self.matrix.loc[user1] == 1].index.values
        user2_articles = self.matrix.loc[user2][self.matrix.loc[user2] == 1].index.values
    
        common_articles = np.intersect1d(user1_articles, user2_articles, assume_unique=True)

        return len(common_articles)

    def _get_top_sorted_users(self, user_id):

        neighbors_df = pd.DataFrame(
            [[i, self._compute_similarities(user_id, i)] for i in self.matrix.index.values if i != user_id], 
            columns=["neighbor_id", "similarity"]
        ) \
        .sort_values(by="similarity", ascending=False) 

        neighbors_df["num_interactions"]  = neighbors_df.neighbor_id.apply(lambda x: len(self.interactions[self.interactions.user_id == x]))
        neighbors_df = neighbors_df.sort_values(by=["similarity", "num_interactions"], ascending=False)
    
        return neighbors_df


    def fit(self, X):
        self.interactions = X
        self.matrix = create_user_item_matrix(X)

    def can_recommend(self, user_id) -> bool:
        return user_id in self.interactions.user_id.values

    def recommend(self, user_id, rec_num = 10) -> list:
        recs = []
        user_articles = get_user_articles(self.interactions, user_id)[0]
        similar_users = self._get_top_sorted_users(user_id)
        for user in similar_users.neighbor_id:
            articles = get_user_articles(self.interactions, user)[0]

            for article in articles:
                if article not in user_articles and article not in recs:
                    recs.append(article)
                    if len(recs) == rec_num:
                        break
        
            if len(recs) == rec_num:
                break
    
        return recs

