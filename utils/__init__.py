

import os
import sys
import ast
import mne
import pytz
import time
import logging
import datetime
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.patches as mpatches
import pickle
import librosa
import xgboost as xgb
import statistics
from tqdm import tqdm
from random import shuffle
from statistics import mean
from numpy import linalg as LA
from matplotlib import pyplot as plt
from numpy import mean, sqrt, square
from collections import OrderedDict
from scipy import interpolate, signal
from scipy.stats import skew, kurtosis
from scipy.fftpack import fft
from sampen import sampen2
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve

matplotlib.rcParams['font.size'] = 8
matplotlib.rcParams['legend.fontsize'] = 8
matplotlib.rcParams['axes.labelsize'] = 9
matplotlib.rcParams['axes.titlesize'] = 9
matplotlib.rcParams['xtick.labelsize'] = 8
matplotlib.rcParams['ytick.labelsize'] = 8




