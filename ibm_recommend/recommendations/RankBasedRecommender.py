from ibm_recommend.recommendations import BaseRecommender

class RankBasedRecommender(BaseRecommender):

    def __init__(self):
        super().__init__("rank_based")

    def fit(self, X):
        self.interactions = X
        self.interactions_summary = X \
            .groupby(by=["article_id", "title"]) \
            .agg(interacts = ("title", "count")) \
            .reset_index() \
            .sort_values(by="interacts", ascending = False)

    def can_recommend(self, user_id) -> bool:
        return user_id not in self.interactions.user_id.values

    def recommend(self, user_id, rec_num = 10) -> list:
        return self.interactions_summary.article_id.head(rec_num).tolist()
