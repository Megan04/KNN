import numpy as np
import matplotlib.pyplot as plt
import operator
import pandas as pd
import os

from sklearn.metrics.pairwise import cosine_similarity
from IPython.display import display, Markdown, clear_output
from ipywidgets import widgets, interactive, Layout

REPO_PATH = "/home/asruser/unsupervised-learning-lab" #set REPO_PATH to empty string "" if running from local

def show_imports():
    shown = False

    str_val = '''
        import os<br/>
        import operator<br/>
        import tqdm<br/>
        import sys<br/>
        <br/>
        import pandas as pd<br/>
        import numpy as np<br/>
        import matplotlib<br/>
        import matplotlib.pyplot as plt<br/>
        import seaborn as sns<br/>
        <br/>
        from mpl_toolkits.mplot3d import axes3d<br/>
        from datetime import datetime<br/>
        <br/>
        from IPython.display import display, Markdown<br/>
        from ipywidgets import widgets, interactive, Layout<br/>
        <br/>
        from sklearn.cluster import KMeans<br/>
        from sklearn.decomposition import PCA<br/>
        from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances<br/>
        <br/>
        from scipy.sparse.linalg import svds<br/>
        '''
    out = widgets.Output()

    def si_text(b):
        nonlocal shown
        if shown:
            with out:
                clear_output()
            shown = False
        else:
            with out:
                display(widgets.HTML(value=str_val))
            shown = True

    btn = widgets.Button(description='Show/hide what we imported',
                        layout=Layout(width='20%'))
    display(btn)
    display(out)
    btn.on_click(si_text)


def show_join_description():
    shown = False

    str_val = '''
        Left join returns all records from the left dataframe 
        and the matched records from the right dataframe.
        '''

    image_file = open(os.path.join(REPO_PATH, "images/leftjoin.gif"), "rb")
    image = image_file.read()
    out = widgets.Output()

    def si_text(b):
        nonlocal shown
        if shown:
            with out:
                clear_output()
            shown = False
        else:
            with out:
                display(widgets.HTML(value=str_val))
                display(widgets.Image(
                    value=image,
                    format='png',
                    width=300,
                    height=300,
                ))
            shown = True

    btn = widgets.Button(description='What is left join?',
                         layout=Layout(width='20%'))
    display(btn)
    display(out)
    btn.on_click(si_text)


def show_genre_popularity_interaction(ratings_per_genre_per_year, genre_all):
    # Create lists of options for each dropdown
    options = genre_all
    color_idx = np.linspace(0, 1, len(genre_all)+1)

    def show_score(genre, show_conf):
        '''
        Display the popularity trend of the selected genre.

        Parameters
        ----------
        genre : str
            selected movie genre
        show_conf: bool
            whether to display the one standard deviation confidence
        '''
        plt.figure(figsize=(10, 5))
        for curr_genre in genre:
            # Get ratings across years for selected genre
            ratings_for_genre = ratings_per_genre_per_year[ratings_per_genre_per_year.genre == curr_genre]

            # Get a unique color and plot
            color = color=plt.cm.tab20_r(options.index(curr_genre))
            plt.plot(ratings_for_genre['timestamp'], ratings_for_genre['mean'], '-', color=color, label=curr_genre)
            plt.xticks(range(1996, 2019, 2))
            plt.ylim(0, 1)

            if show_conf:
                plt.fill_between(ratings_for_genre['timestamp'],
                                 ratings_for_genre['mean'] - ratings_for_genre['std'],
                                 ratings_for_genre['mean'] + ratings_for_genre['std'],
                                 color=color, alpha=0.2)
        plt.xlabel('Year')
        plt.ylabel('Mean rating ± 1 std deviation' if show_conf else 'Mean rating')
        plt.legend()
        plt.show()
        display(Markdown('#### {}'.format(list(genre))))

    # IPython widgets to allow interactive selection of genre and confidence options
    m = interactive(show_score, genre=widgets.SelectMultiple(options=options,
                                                             rows=10,
                                                             value=[options[0]],
                                                             description='Genre:'),
                    show_conf = widgets.Checkbox(
                        value=False,
                        description='Show confidence',
                        disabled=False
                    ))
    display(Markdown('#### Chose genre(s) from the list and use "Show confidence" to toggle confidence display. \
    Use shift or ctrl (cmd) key to select multiple genres.'))
    display(m)


def show_cluster_interaction(u_m_rating_df):
    cluster_options = range(8)
    num_movies_options = range(1, 21)

    def show_movies(cluster, num_movies):
        temp_df = u_m_rating_df[u_m_rating_df.cluster == cluster]
        display(temp_df.head(num_movies)[['title', 'genres', 'cluster']])

    dm = interactive(show_movies, cluster=widgets.Dropdown(options=cluster_options,
                                                           value=cluster_options[0],
                                                           description='Cluster:'),
                     num_movies = widgets.Dropdown(options=num_movies_options,
                                                   value=5,
                                                   description='# movies:'))

    display(Markdown('#### Chose a cluster and number of movies from that cluster you want to see.'))
    display(dm)

genre_columns_local = ['Action', 'Adventure', 'Animation', 'Children',
                       'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
                       'Horror', 'IMAX', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                       'Thriller', 'War', 'Western']

rating_feature_columns_local = list(range(1, 611))

movie_feature_columns_local = list(range(50))


def get_similarity_genre_jaccard(query_feats, data_feats):
    '''
    Returns similarity based using jacccard index

    Parameters
    ----------
    query_feats : int
        features of the query
    data_feats : pandas datarame
        features of all data points

    Returns
    -------
    int, np.array
        jacccard similarity of query with each item in the data
    '''
    feats_intersection = (query_feats & data_feats).sum(axis=1)
    feats_union = (query_feats | data_feats).sum(axis=1)

    similarities = (feats_intersection * 1.0) / feats_union

    return similarities


def get_similarity_cosine(query_feats, data_feats):
    '''
    Returns similarity based using cosine similarity

    Parameters
    ----------
    query_feats : int
        features of the query
    data_feats : pandas datarame
        features of all data points

    Returns
    -------
    int, np.array
        cosine similarity of query with each item in the data
    '''
    similarities = cosine_similarity(query_feats, data_feats)

    return similarities.flatten()


def get_similar_movies_fast(movie_id, data_frame, feature_columns, sim_function, number_of_recommendations=5):
    '''
    Returns movies similar to the movie with id movie_id.
    Vectorized implementation.

    Parameters
    ----------
    movie_id : int
        query movie id
    data_frame : pandas datarame
        dataframe with information about all movies
    feature_columns: column names of features
    sim_function : function that returns similarity between query features
                and all data points
    number_of_recommendations : int, default 5
        number of recommendations to return

    Returns
    -------
    dataframe : most similar movies
    '''
    # Get query movie features
    query_row = data_frame.loc[data_frame.movieId==movie_id]
    query_feats = query_row[feature_columns].values

    # Get features of all other movies in a m*n array
    # where m is the number of movies and n is the number of features
    data_feats = data_frame[feature_columns].values

    # apply the similarity function on the 2 features sets
    similarities = sim_function(data_feats, query_feats)

    # Sort by similarity and return n most similar movies
    movie_id_similarity = zip(data_frame.movieId.values, similarities)
    movie_id_similarity = sorted(movie_id_similarity, key=operator.itemgetter(1), reverse=True)

    movie_id_similarity = pd.DataFrame(movie_id_similarity[:number_of_recommendations+1],
                                       columns=['movieId', 'similarity'])

    movie_id_similarity = movie_id_similarity[movie_id_similarity.movieId!=movie_id]

    movie_id_similarity = movie_id_similarity.merge(data_frame,
                                                    left_on='movieId',
                                                    right_on='movieId')[:number_of_recommendations]

    return movie_id_similarity[['movieId', 'title', 'genres', 'similarity']]


def show_compare_interactive(pop_movs,
                             mov_idx,
                             movies_df,
                             methods,
                             movies_df_with_indicators=None,
                             user_movie_rating_df=None,
                             movies_df_with_svd=None):

    num_movies_options = range(1, 21)

    def show_movies(title, method, num_movies=5):
        mov_id = mov_idx[title]

        display(Markdown(f'##### {title} using "{method}"'))

        display(movies_df[movies_df.title==title])

        try:
            # Call appropriate function based on type of method
            if method=='jaccard similarity of genre':
                display(get_similar_movies_fast(mov_id,
                                                movies_df_with_indicators,
                                                genre_columns_local,
                                                get_similarity_genre_jaccard,
                                                num_movies))
            elif method=='item-item collaborative filtering':
                display(get_similar_movies_fast(mov_id,
                                                user_movie_rating_df,
                                                rating_feature_columns_local,
                                                get_similarity_cosine,
                                                num_movies))
            elif method=='matrix factorization':
                display(get_similar_movies_fast(mov_id,
                                                movies_df_with_svd,
                                                movie_feature_columns_local,
                                                get_similarity_cosine,
                                                num_movies))
        except Exception as e:
            raise e
            display(Markdown('Something went wrong, please try another combination'))

    dm_compare = interactive(show_movies, title=widgets.Dropdown(options=pop_movs,
                                                                 value=pop_movs[0],
                                                                 description='Title:'),
                             method = widgets.Dropdown(options=methods,
                                                       value=methods[0],
                                                       description='Method:'),
                             num_movies = widgets.Dropdown(options=num_movies_options,
                                                           value=5,
                                                           description='# movies:'))

    display(Markdown('### Chose a movie and method to see recommendations.'))
    display(dm_compare)


