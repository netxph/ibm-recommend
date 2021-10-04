import numpy as np
import matplotlib.pyplot as plt

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