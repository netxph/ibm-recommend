class Recommender():
    '''Main recommender class
    '''

    def __init__(self, recommenders):
        self.recommenders = recommenders

    def fit(self, X):
        '''
        Train all recommenders

        INPUT:
        X: pandas dataframe
        '''

        for rec in self.recommenders:
            rec.fit(X)

    def recommend(self, user_id, rec_num = 10):
        '''
        Run to all recommenders and gather recommendations

        INPUT:
        user_id: int, the user id
        rec_num: int, the number of recommendations

        OUTPUT:
        recommendations: list, the recommendations
        '''

        recs = {}
        for rec in self.recommenders:


            if rec.can_recommend(user_id):
                recs[rec.name] = rec.recommend(user_id, rec_num)

        return recs