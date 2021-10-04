class Recommender():

    def __init__(self, recommenders):
        self.recommenders = recommenders

    def fit(self, X):
        for rec in self.recommenders:
            rec.fit(X)

    def recommend(self, user_id, rec_num = 10):
        recs = {}
        for rec in self.recommenders:


            if rec.can_recommend(user_id):
                recs[rec.name] = rec.recommend(user_id, rec_num)

        return recs