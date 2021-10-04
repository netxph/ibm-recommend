from ibm_recommend.recommendations import BaseRecommender

class RankBasedRecommender(BaseRecommender):
    """Rank Based Recommender
    """

    def __init__(self):
        super().__init__("rank_based")

    def fit(self, X):
        '''
        Fit the model

        INPUT:
        X: pandas.DataFrame, the dataframe of interactions
        '''

        self.interactions = X
        self.interactions_summary = X \
            .groupby(by=["article_id", "title"]) \
            .agg(interacts = ("title", "count")) \
            .reset_index() \
            .sort_values(by="interacts", ascending = False)

    def can_recommend(self, user_id) -> bool:
        '''
        Checks if the model can recommend for the user

        INPUT:
        user_id: int, the user id

        OUTPUT:
        bool, True if the model can recommend for the user, False otherwise
        '''
        return user_id not in self.interactions.user_id.values

    def recommend(self, user_id, rec_num = 10) -> list:
        '''
        Recommend articles for the user

        INPUT:
        user_id: int, the user id

        OUTPUT:
        list, the list of recommended articles
        '''

        return self.interactions_summary.article_id.head(rec_num).tolist()
