import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_user_articles(data, user_id):
    '''
    INPUT:
    data - (pandas dataframe) matrix of users by articles: 
                1's when a user has interacted with an article, 0 otherwise
    user_id - (int) a user id
    
    OUTPUT:
    article_ids - (list) a list of the article ids seen by the user
    article_names - (list) a list of article names associated with the list of article ids 
                    (this is identified by the doc_full_name column in df_content)
    
    Description:
    Provides a list of the article_ids and article titles that have been seen by a user
    '''
    num_interactions = data[["user_id", "article_id", "title"]].groupby("article_id").agg(count = ("user_id", "count")).reset_index()

    user_articles = data[["user_id", "article_id", "title"]][data.user_id == user_id].drop_duplicates()

    sorted_articles = pd.merge(user_articles, num_interactions, on="article_id").sort_values(by="count", ascending=False)
    

    return sorted_articles.article_id.tolist(), sorted_articles.title.tolist() # return the ids and names

def create_user_item_matrix(data):
    '''
    INPUT:
    data - pandas dataframe with article_id, title, user_id columns
    
    OUTPUT:
    user_item - user item matrix 
    
    Description:
    Return a matrix with user ids as rows and article ids on the columns with 1 values where a user interacted with 
    an article and a 0 otherwise
    '''

    user_item = data[["user_id", "article_id"]].groupby(by=["user_id", "article_id"]).agg(lambda x: 1).unstack(fill_value = 0)
    
    return user_item

def fit_transform(s, u, vt, use_round = True):
    '''
    INPUT
    s - singular values
    u - left singular vectors
    vt - transpose of right singular vectors

    OUTPUT
    user_item_est - a user-item matrix of the estimated ratings
    '''

    result = np.dot(np.dot(u, s), vt)

    return np.around(result) if use_round else result

def calculate_error(df_true, df_pred):
    '''
    INPUT
    df_true - a user-item matrix of the true ratings
    df_pred - a user-item matrix of the predicted ratings

    OUTPUT
    err - total error of the predicted ratings
    '''

    diffs = np.subtract(df_true, df_pred)

    return np.sum(np.sum(np.abs(diffs)))

def calculate_accuracy(df_true, df_pred):
    '''
    INPUT
    df_true - a user-item matrix of the true ratings
    df_pred - a user-item matrix of the predicted ratings

    OUTPUT
    acc - accuracy of the predicted ratings
    '''

    errors = calculate_error(df_true, df_pred)

    return 1 - np.array(errors)/(df_true.shape[0] * df_true.shape[1])


def intersect_data(data1, data2):
    '''
    INPUT
    data1 - a user-item matrix
    data2 - a user-item matrix

    OUTPUT
    data_intersect1 - a user-item matrix of the intersection of data1
    data_intersect2 - a user-item matrix of the intersection of data2
    '''
    
    idx = np.intersect1d(data1.index, data2.index)
    cols = np.intersect1d(data1.columns, data2.columns)

    return data1.loc[idx][cols], \
        data2.loc[idx][cols]


def plot_accuracy(num_latent_feats, accuracy, legend):
    '''
    INPUT
    num_latent_feats - a list of number of latent features to use
    accuracy - a list of accuracies corresponding to the number of latent features

    OUTPUT
    None - This function outputs plots of the accuracy of the predicted ratings
    '''
    plt.plot(num_latent_feats, accuracy, label=legend);
    plt.xlabel('Number of Latent Features');
    plt.ylabel('Accuracy');
    plt.title('Accuracy vs. Number of Latent Features');
    ax = plt.gca()
    ax.legend()