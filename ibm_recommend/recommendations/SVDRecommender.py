import numpy as np

from ibm_recommend.recommendations import BaseRecommender
from ibm_recommend.matrix import fit_transform, create_user_item_matrix, get_user_articles

class SVDRecommender(BaseRecommender):
    '''Matrix Factorization with SVD Recommender
    '''

    def __init__(self, k = 4):
        super().__init__("svd")

        self.k = k

    def fit(self, X):
        '''
        Fits the model to the data.

        INPUT:
            X: pandas.DataFrame
        '''

        self.interactions = X
        self.matrix = create_user_item_matrix(X)

        u, s, vt = np.linalg.svd(self.matrix)
        u_k, s_k, vt_k = u[:, :self.k], np.diag(s[:self.k]), vt[:self.k, :]

        self.preds = fit_transform(s_k, u_k, vt_k, use_round=False)

    def can_recommend(self, user_id) -> bool:
        '''
        Checks if the model can recommend for the user.

        INPUT:
            user_id: int
        '''

        return user_id in self.interactions.user_id.values

    def recommend(self, user_id, rec_num = 10) -> list:
        '''
        Provides recommendations for the user.

        INPUT:
            user_id: int, the user id
            rec_num: int, the number of recommendations to return
        '''
        idx = self.matrix.index.get_loc(user_id)
        article_ids, _ = get_user_articles(self.interactions, user_id)
        col_idxs =[self.matrix.columns.get_loc(article_id) for article_id in article_ids]

        preds = self.preds[idx]
        preds_flt = np.delete(preds, col_idxs)

        sorted_idxs = preds_flt.argsort()[-rec_num:][::-1]

        return self.matrix.columns[sorted_idxs].tolist()
 