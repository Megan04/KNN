import os
import operator
import tqdm
import sys

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from mpl_toolkits.mplot3d import axes3d
from datetime import datetime

from IPython.display import display, Markdown, clear_output
from ipywidgets import widgets, interactive, Layout

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances

from scipy.sparse.linalg import svds
