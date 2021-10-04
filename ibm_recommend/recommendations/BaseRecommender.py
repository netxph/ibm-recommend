class BaseRecommender():

    def __init__(self, name):
        self.name = name

    def fit(self, X):
        pass

    def can_recommend(self, user_id) -> bool:
        pass

    def recommend(self, user_id, rec_num = 10) -> list:
        pass