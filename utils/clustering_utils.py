import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import root_mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
import constants as c


def perform_clustering(video_df, data_cols, cluster_target, corr_list):
    """
    Perform KMeans clustering and print cluster compositions and correlations.
    :param video_df: DataFrame with video data
    :param data_cols: list of columns to use for clustering
    :param cluster_target: name of the new cluster column
    :param corr_list: list of columns to compute correlations with
    :return: DataFrame with cluster assignments
    """
    X = video_df[data_cols].copy()
    kmeans = KMeans(n_clusters=2, random_state=42)
    video_df[cluster_target] = kmeans.fit_predict(X)

    # Cluster video composition
    cluster_videos = video_df.groupby(cluster_target)[c.VIDEO_ID_COL].apply(list)
    for cluster_label, videos in cluster_videos.items():
        print(f"\nCluster {cluster_label} contains {len(videos)} videos:")
        print(videos)

    # Create separate sorted correlation DataFrames
    for i in corr_list:
        print(video_df[data_cols].corrwith(video_df[i]).sort_values(ascending=False).to_frame(name=f'{i} corr'))
        print()
    return video_df


def compute_k_rmse(X, video_df, target_col=c.VALENCE):
    """
    Compute RMSE for k-NN predictions across different k values to find the optimal k.
    :param X: feature DataFrame
    :param video_df: DataFrame with video data
    :param target_col: name of the target column
    :return: DataFrame with RMSE values for each k, best RMSE, and best k
    """
    rmses = []
    for k in range(1, len(X) - 1):
        preds = []
        gt = []
        for vid in X.index:
            pred = get_neighbor_prediction(X, vid, video_df, k, target_col)
            preds.append(pred)
            gt.append(video_df.loc[vid, target_col])
        rmse_k = root_mean_squared_error(gt, preds)
        rmses.append(rmse_k)

    rmses_df = (pd.DataFrame({'k': range(1, len(X) - 1), 'rms': rmses})
                .set_index('k'))

    best_k = rmses_df['rms'].idxmin()
    best_rmse = rmses_df.loc[best_k, 'rms']
    print(f"Global best k = {best_k}")
    print(f"Global best RMSE = {best_rmse}")

    return rmses_df, best_rmse, best_k



def predict_candidates(X, video_df, candidate_df, feature_cols, target_col, k, id_col=c.VIDEO_ID_COL, negate=False):
    """
    Predict valence for candidate videos using k-NN trained on existing video data.
    :param X: feature DataFrame for training videos
    :param video_df: DataFrame with training video data
    :param candidate_df: DataFrame with candidate video data
    :param feature_cols: list of feature columns to use
    :param target_col: name of the target column
    :param k: number of neighbors
    :param id_col: name of the ID column
    :param negate: whether to negate predictions
    :return: DataFrame with candidate IDs and predicted valence
    """
    knn = KNeighborsRegressor(n_neighbors=k, weights='distance', metric='manhattan')
    knn.fit(X, video_df.loc[X.index, target_col])

    rows = []
    for vid in candidate_df.index:
        pred = knn.predict(candidate_df.loc[vid, feature_cols].to_frame().T)[0]
        if negate:
            pred = -pred
        rows.append({id_col: candidate_df.loc[vid, id_col], 'valence_prediction': pred})

    results_df = (pd.DataFrame(rows).sort_values(by='valence_prediction', ascending=False).reset_index(drop=True)
                  )
    return results_df


def get_neighbor_prediction(
        X: pd.DataFrame,
        video: str,
        video_df: pd.DataFrame,
        k: int,
        target_col: str = c.VALENCE
) -> float:
    """
    Get k-NN prediction for a specific video.
    :param X: feature DataFrame
    :param video: video ID to predict
    :param video_df: DataFrame with video data
    :param k: number of neighbors
    :param target_col: name of the target column
    :return: predicted value
    """
    curr_X = X.drop(index=[video])
    ratings_X = video_df.loc[curr_X.index, target_col]
    knn = KNeighborsRegressor(n_neighbors=k, weights='distance', metric='manhattan')
    knn.fit(curr_X, ratings_X)
    prediction = knn.predict(X.loc[video, :].to_frame().T)[0]

    return prediction
