# --- Imports ---

import re
import os
import gc
import dill
import random
import pickle

import numpy as np
import polars as pl
import pandas as pd

import lightgbm as lgb

from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedGroupKFold

from typing import Tuple, Union

import sys
sys.path.append('/home/toefl/K/MCTS/dataset/')
sys.path.append('/kaggle/input/openfe-modified')

from openfe import transform
import kaggle_evaluation.mcts_inference_server

import warnings
warnings.filterwarnings('ignore')


# --- Run mode ---

IS_TRAIN = True
LOCAL = True
IS_RERUN = False


# --- Config ---

class Config:
    """Config."""
    
    # Seed
    
    seed = 0xFACED
    
    # Mode (Train / Inference)
    
    is_train = IS_TRAIN
    
    # Paths
    
    path_to_train_dataset = '/home/toefl/K/MCTS/dataset/train.csv' if LOCAL else '/kaggle/input/um-game-playing-strength-of-mcts-variants/train.csv' 
    path_to_save_data_checkpoint = 'data_checkpoint.pickle'     # Drop columns, categorical columns, etc.
    path_to_load_data_checkpoint = 'data_checkpoint.pickle' if LOCAL else '/kaggle/input/mcts-solution-checkpoint/data_checkpoint.pickle'
    path_to_save_solver_checkpoint = 'solver_checkpoint.pickle' # Models, weights, etc.
    path_to_load_solver_checkpoint = 'solver_checkpoint.pickle' if LOCAL else '/kaggle/input/mcts-solution-checkpoint/solver_checkpoint.pickle'
    path_to_load_features = 'feature.pickle' if LOCAL else '/kaggle/input/mcts-solution-checkpoint/feature.pickle'
    path_to_tfidf = '/home/toefl/K/MCTS/dataset/tf_idf' if LOCAL else '/kaggle/input/mcts-solution-checkpoint/tf_idf'
    
    # Training

    task = "regression"
    
    n_splits = 5

    n_openfe_features = (0, 0)    # (All, numerical)
    n_tf_ids_features = 0
    
    catboost_params = {
        'iterations': 30000,
        'learning_rate': 0.01,
        'depth': 10,
        'early_stopping_rounds': 200,
        
        'loss_function': 'RMSE',
        'task_type': 'GPU',
        'verbose': 1000,
        'thread_count': 14,
        
        'use_best_model': True,
        'random_seed': 0xFACED,
    }
    
    lgbm_params = {
        "boosting_type": "gbdt",
        'objective': 'regression',
        
        'num_iterations': 30000,
        'learning_rate': 0.01,
        'max_depth': 10,
        'extra_trees': True,
        
        'metric': 'rmse',
        'device': 'gpu',
        'verbose': -1,
        'seed': 0xFACED,
    }
    
    to_train = {
        "catboost": True,
        "lgbm": False,
    }
    
    weights = {
        "catboost": 1,
        "lgbm": 0,
    }


# --- Utils ---

def set_seed(SEED):
    """Set random seeds."""
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)


# --- Data pipeline ---

class Dataset:
    """The data pipeline."""
    
    def __init__(self, config: Config) -> None:
        """Initialization."""
        self.config = config
        
        if self.config.is_train:
            self.data_checkpoint = {
                "dropcols": [],
                "catcols": [],
            }
        else:
            with open(config.path_to_load_data_checkpoint, "rb") as f:
                self.data_checkpoint = pickle.load(f)
        
    def preprocessing(self, df: pl.DataFrame) -> pl.DataFrame:
        """The basic preprocessing."""
        
        # Mirror the dataset.
        
        if self.config.is_train:
            df = df.with_columns(pl.lit("original").alias("data_mode"))

            df_mirror = df.clone()

            df_mirror = df_mirror.with_columns(
                pl.col("agent1").alias("agent2"),
                pl.col("agent2").alias("agent1"),
                (pl.col("utility_agent1") * -1).alias("utility_agent1"),
                (1 - pl.col("AdvantageP1")).alias("AdvantageP1"),
                pl.lit("mirror").alias("data_mode")
            )

            df = pl.concat([df, df_mirror])
        
        # Initial data shape.
        
        print("Initial shape", df.shape)
        
        # Drop constant columns.
        
        if self.config.is_train:
            constant_columns = np.array(df.columns)[df.select(pl.all().n_unique() == 1).to_numpy().ravel()]
            drop_columns = list(constant_columns) + ['Id']
            self.data_checkpoint["dropcols"] += drop_columns
        else:
            drop_columns = self.data_checkpoint["dropcols"]
            
        df = df.drop(drop_columns)
        
        print('Shape after dropping constant columns:', df.shape)
        
        # Basic information.
        
        print('There are', df.null_count().to_numpy().sum(), 'missing values.')
        print('There are', df.select(pl.all().n_unique() == 2).to_numpy().sum(), 'binary columns.')
        
        return df
        
    def feature_generation(self, df: pl.DataFrame) -> pl.DataFrame:
        """Generate the new features."""

        def ARI(txt):
            characters = len(txt)
            words = len(re.split(' |\\n|\\.|\\?|\\!|\,', txt))
            sentence = len(re.split('\\.|\\?|\\!', txt))
            ari_score = 4.71*(characters/words)+0.5*(words/sentence)-21.43
            return ari_score

        def McAlpine_EFLAW(txt):
            W = len(re.split(' |\\n|\\.|\\?|\\!|\,', txt))
            S = len(re.split('\\.|\\?|\\!', txt))
            mcalpine_eflaw_score = (W+S*W)/S
            return mcalpine_eflaw_score
        
        def CLRI(txt):
            characters = len(txt)
            words = len(re.split(' |\\n|\\.|\\?|\\!|\,', txt))
            sentence = len(re.split('\\.|\\?|\\!', txt))
            L = 100*characters/words
            S = 100*sentence/words
            clri_score = 0.0588*L-0.296*S-15.8
            return clri_score
        
        def drop_gamename(rule):
            rule = rule[len('(game "'):]
            for i in range(len(rule)):
                if rule[i] == '"':
                    return rule[i+1:]
                
        def get_player(rule):
            player = ''
            stack = []
            for i in range(len(rule)):
                player += rule[i]
                if rule[i] in ['(', '{']:
                    stack.append(rule[i])
                elif rule[i] in [')', '}']:
                    stack = stack[:-1]
                    if len(stack) == 0:
                        return player
        
        # Split the agent string.

        df = df.with_columns(
            pl.col('agent1').str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', 1).alias('src_p1_selection'),
            pl.col('agent1').str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', 2).alias('src_p1_exploration'),
            pl.col('agent1').str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', 3).alias('src_p1_playout'),
            pl.col('agent1').str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', 4).alias('src_p1_bounds'),
            pl.col('agent2').str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', 1).alias('src_p2_selection'),
            pl.col('agent2').str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', 2).alias('src_p2_exploration'),
            pl.col('agent2').str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', 3).alias('src_p2_playout'),
            pl.col('agent2').str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', 4).alias('src_p2_bounds'),

            pl.col('agent2').str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', 1).alias('tta_p1_selection'),
            pl.col('agent2').str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', 2).alias('tta_p1_exploration'),
            pl.col('agent2').str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', 3).alias('tta_p1_playout'),
            pl.col('agent2').str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', 4).alias('tta_p1_bounds'),
            pl.col('agent1').str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', 1).alias('tta_p2_selection'),
            pl.col('agent1').str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', 2).alias('tta_p2_exploration'),
            pl.col('agent1').str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', 3).alias('tta_p2_playout'),
            pl.col('agent1').str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', 4).alias('tta_p2_bounds')
        )

        # TTA fundament.

        df = df.with_columns(
            pl.col('agent1').alias('src_agent1'),
            pl.col('agent2').alias('src_agent2'),
            pl.col('agent1').alias('tta_agent2'),
            pl.col('agent2').alias('tta_agent1'),
            pl.col('AdvantageP1').alias('src_AdvantageP1'),
            (1 - pl.col('AdvantageP1')).alias('tta_AdvantageP1'),
        )

        df = df.drop(['AdvantageP1'], strict=False)

        # Feature engineering.
        
        df = df.to_pandas()

        # Player position (positive / negative).

        total_agent = ['MCTS-ProgressiveHistory-0.1-MAST-false', 'MCTS-ProgressiveHistory-0.1-MAST-true', 'MCTS-ProgressiveHistory-0.1-NST-false', 'MCTS-ProgressiveHistory-0.1-NST-true', 'MCTS-ProgressiveHistory-0.1-Random200-false', 'MCTS-ProgressiveHistory-0.1-Random200-true', 'MCTS-ProgressiveHistory-0.6-MAST-false', 'MCTS-ProgressiveHistory-0.6-MAST-true', 'MCTS-ProgressiveHistory-0.6-NST-false', 'MCTS-ProgressiveHistory-0.6-NST-true', 'MCTS-ProgressiveHistory-0.6-Random200-false', 'MCTS-ProgressiveHistory-0.6-Random200-true', 'MCTS-ProgressiveHistory-1.41421356237-MAST-false', 'MCTS-ProgressiveHistory-1.41421356237-MAST-true', 'MCTS-ProgressiveHistory-1.41421356237-NST-false', 'MCTS-ProgressiveHistory-1.41421356237-NST-true', 'MCTS-ProgressiveHistory-1.41421356237-Random200-false', 'MCTS-ProgressiveHistory-1.41421356237-Random200-true', 'MCTS-UCB1-0.1-MAST-false', 'MCTS-UCB1-0.1-MAST-true', 'MCTS-UCB1-0.1-NST-false', 'MCTS-UCB1-0.1-NST-true', 'MCTS-UCB1-0.1-Random200-false', 'MCTS-UCB1-0.1-Random200-true', 'MCTS-UCB1-0.6-MAST-false', 'MCTS-UCB1-0.6-MAST-true', 'MCTS-UCB1-0.6-NST-false', 'MCTS-UCB1-0.6-NST-true', 'MCTS-UCB1-0.6-Random200-false', 'MCTS-UCB1-0.6-Random200-true', 'MCTS-UCB1-1.41421356237-MAST-false', 'MCTS-UCB1-1.41421356237-MAST-true', 'MCTS-UCB1-1.41421356237-NST-false', 'MCTS-UCB1-1.41421356237-NST-true', 'MCTS-UCB1-1.41421356237-Random200-false', 'MCTS-UCB1-1.41421356237-Random200-true', 'MCTS-UCB1GRAVE-0.1-MAST-false', 'MCTS-UCB1GRAVE-0.1-MAST-true', 'MCTS-UCB1GRAVE-0.1-NST-false', 'MCTS-UCB1GRAVE-0.1-NST-true', 'MCTS-UCB1GRAVE-0.1-Random200-false', 'MCTS-UCB1GRAVE-0.1-Random200-true', 'MCTS-UCB1GRAVE-0.6-MAST-false', 'MCTS-UCB1GRAVE-0.6-MAST-true', 'MCTS-UCB1GRAVE-0.6-NST-false', 'MCTS-UCB1GRAVE-0.6-NST-true', 'MCTS-UCB1GRAVE-0.6-Random200-false', 'MCTS-UCB1GRAVE-0.6-Random200-true', 'MCTS-UCB1GRAVE-1.41421356237-MAST-false', 'MCTS-UCB1GRAVE-1.41421356237-MAST-true', 'MCTS-UCB1GRAVE-1.41421356237-NST-false', 'MCTS-UCB1GRAVE-1.41421356237-NST-true', 'MCTS-UCB1GRAVE-1.41421356237-Random200-false', 'MCTS-UCB1GRAVE-1.41421356237-Random200-true', 'MCTS-UCB1Tuned-0.1-MAST-false', 'MCTS-UCB1Tuned-0.1-MAST-true', 'MCTS-UCB1Tuned-0.1-NST-false', 'MCTS-UCB1Tuned-0.1-NST-true', 'MCTS-UCB1Tuned-0.1-Random200-false', 'MCTS-UCB1Tuned-0.1-Random200-true', 'MCTS-UCB1Tuned-0.6-MAST-false', 'MCTS-UCB1Tuned-0.6-MAST-true', 'MCTS-UCB1Tuned-0.6-NST-false', 'MCTS-UCB1Tuned-0.6-NST-true', 'MCTS-UCB1Tuned-0.6-Random200-false', 'MCTS-UCB1Tuned-0.6-Random200-true', 'MCTS-UCB1Tuned-1.41421356237-MAST-false', 'MCTS-UCB1Tuned-1.41421356237-MAST-true', 'MCTS-UCB1Tuned-1.41421356237-NST-false', 'MCTS-UCB1Tuned-1.41421356237-NST-true', 'MCTS-UCB1Tuned-1.41421356237-Random200-false', 'MCTS-UCB1Tuned-1.41421356237-Random200-true']
        
        agent1, agent2 = df['src_agent1'].values, df['src_agent2'].values
        for i in range(len(total_agent)):
            value = np.zeros(len(df))
            for j in range(len(df)):
                if agent1[j] == total_agent[i]:
                    value[j] += 1
                elif agent2[j] == total_agent[i]:
                    value[j] -= 1
            df[f'src_agent_{total_agent[i]}'] = value
        
        agent1, agent2 = df['tta_agent1'].values, df['tta_agent2'].values
        for i in range(len(total_agent)):
            value = np.zeros(len(df))
            for j in range(len(df)):
                if agent1[j] == total_agent[i]:
                    value[j] += 1
                elif agent2[j] == total_agent[i]:
                    value[j] -= 1
            df[f'tta_agent_{total_agent[i]}'] = value

        # One-hot encoding.

        onehot_cols = [['NumOffDiagonalDirections', [0.0, 4.82, 2.0, 5.18, 3.08, 0.06]], ['NumLayers', [1, 0, 4, 5]], ['NumPhasesBoard', [3, 2, 1, 5, 4]], ['NumContainers', [1, 4, 3, 2]], ['NumDice', [0, 2, 1, 4, 6, 3, 5, 7]], ['ProposeDecisionFrequency', [0.0, 0.05, 0.01]], ['PromotionDecisionFrequency', [0.0, 0.01, 0.03, 0.02, 0.11, 0.05, 0.04]], ['SlideDecisionToFriendFrequency', [0.0, 0.19, 0.06]], ['LeapDecisionToEnemyFrequency', [0.0, 0.04, 0.01, 0.02, 0.07, 0.03, 0.14, 0.08]], ['HopDecisionFriendToFriendFrequency', [0.0, 0.13, 0.09]], ['HopDecisionEnemyToEnemyFrequency', [0.0, 0.01, 0.2, 0.03]], ['HopDecisionFriendToEnemyFrequency', [0.0, 0.01, 0.09, 0.25, 0.02]], ['FromToDecisionFrequency', [0.0, 0.38, 1.0, 0.31, 0.94, 0.67]], ['ProposeEffectFrequency', [0.0, 0.01, 0.03]], ['PushEffectFrequency', [0.0, 0.5, 0.96, 0.25]], ['FlipFrequency', [0.0, 0.87, 1.0, 0.96]], ['SetCountFrequency', [0.0, 0.62, 0.54, 0.02]], ['DirectionCaptureFrequency', [0.0, 0.55, 0.54]], ['EncloseCaptureFrequency', [0.0, 0.08, 0.1, 0.07, 0.12, 0.02, 0.09]], ['InterveneCaptureFrequency', [0.0, 0.01, 0.14, 0.04]], ['SurroundCaptureFrequency', [0.0, 0.01, 0.03, 0.02]], ['NumPlayPhase', [1, 2, 3, 4, 5, 6, 7, 8]], ['LineLossFrequency', [0.0, 0.96, 0.87, 0.46, 0.26, 0.88, 0.94]], ['ConnectionEndFrequency', [0.0, 0.19, 1.0, 0.23, 0.94, 0.35, 0.97]], ['ConnectionLossFrequency', [0.0, 0.54, 0.78]], ['GroupEndFrequency', [0.0, 1.0, 0.11, 0.79]], ['GroupWinFrequency', [0.0, 0.11, 1.0]], ['LoopEndFrequency', [0.0, 0.14, 0.66]], ['LoopWinFrequency', [0.0, 0.14, 0.66]], ['PatternEndFrequency', [0.0, 0.63, 0.35]], ['PatternWinFrequency', [0.0, 0.63, 0.35]], ['NoTargetPieceWinFrequency', [0.0, 0.72, 0.77, 0.95, 0.32, 1.0]], ['EliminatePiecesLossFrequency', [0.0, 0.85, 0.96, 0.68]], ['EliminatePiecesDrawFrequency', [0.0, 0.03, 0.91, 1.0, 0.36, 0.86]], ['NoOwnPiecesLossFrequency', [0.0, 1.0, 0.68]], ['FillEndFrequency', [0.0, 1.0, 0.04, 0.01, 0.99, 0.72]], ['FillWinFrequency', [0.0, 1.0, 0.04, 0.01, 0.99]], ['ReachDrawFrequency', [0.0, 0.9, 0.98]], ['ScoringLossFrequency', [0.0, 0.6, 0.62]], ['NoMovesLossFrequency', [0.0, 1.0, 0.13, 0.06]], ['NoMovesDrawFrequency', [0.0, 0.01, 0.04, 0.03, 0.22]], ['BoardSitesOccupiedChangeNumTimes', [0.0, 0.06, 0.42, 0.12, 0.14, 0.94]], ['BranchingFactorChangeNumTimesn', [0.0, 0.3, 0.02, 0.07, 0.04, 0.13, 0.01, 0.21, 0.03]], ['PieceNumberChangeNumTimes', [0.0, 0.06, 0.42, 0.12, 0.14, 1.0]], ['src_p1_selection', ['ProgressiveHistory', 'UCB1', 'UCB1GRAVE', 'UCB1Tuned']], ['src_p2_selection', ['ProgressiveHistory', 'UCB1GRAVE', 'UCB1', 'UCB1Tuned']], ['src_p1_exploration', ['0.1', '0.6', '1.41421356237']], ['src_p2_exploration', ['0.6', '0.1', '1.41421356237']], ['src_p1_playout', ['MAST', 'NST', 'Random200']], ['src_p2_playout', ['Random200', 'NST', 'MAST']]]
        for col, unique in onehot_cols:
            for u in unique:
                df[f'{col}_{u}'] = (df[col] == u).astype(np.int8)
                if 'src' in col:
                    tta_col = col.replace('src', 'tta')
                    df[f'{tta_col}_{u}'] = (df[col] == u).astype(np.int8)

        # Drop game's name from the rules.

        df['LudRules'] = df['LudRules'].apply(lambda x: drop_gamename(x))

        # Get player.

        df['player'] = df['LudRules'].apply(lambda rule: get_player(rule))
        df['player_len'] = df['player'].apply(len)
        df['LudRules'] = [rule[len(player):] for player, rule in zip(
            df['player'], df['LudRules'])]
        df = df.drop(['player'], axis=1)

        # Rules parcing.

        for rule in ['EnglishRules', 'LudRules']:
            df[rule + "_ARI"] = df[rule].apply(lambda x: ARI(x))
            df[rule + "CLRI"] = df[rule].apply(lambda x: CLRI(x))
            df[rule + "McAlpine_EFLAW"] = df[rule].apply(lambda x: McAlpine_EFLAW(x))

        # External features.

        df['PlayoutsPerSecond/MovesPerSecond'] = df['PlayoutsPerSecond'] / df['MovesPerSecond']

        df = pl.from_pandas(df)

        return df
    
    def build_validation_and_cv_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Build the validation and CV features."""
        
        if self.config.is_train:

            # Build the validation based on the original data and then assign the same folds to the mirror data to avoid the leak.

            src_df = df.clone()
            
            df = df.filter(pl.col('data_mode') == 'original')
            
            df = df.with_columns(pl.lit(0).alias("fold"))
            df = df.with_row_index('index')

            cv = StratifiedGroupKFold(n_splits=self.config.n_splits, shuffle=True, random_state=self.config.seed)

            for fold, (_, index) in enumerate(cv.split(
                    df,
                    df["utility_agent1"].alias("utility_agent1").cast(pl.Utf8) + "_" + df["agent1"],
                    df["GameRulesetName"]
                )):
                df = df.with_columns(
                    pl.when(pl.col('index').is_in(index))
                    .then(pl.lit(fold))
                    .otherwise(pl.col('fold'))
                    .alias('fold')
                )

            src_df = src_df.with_columns(pl.Series("fold", np.concatenate((df["fold"].to_numpy(), df["fold"].to_numpy()))))

            src_df = src_df.drop(['index', 'agent1', 'agent2'], strict=False)

            # Drop duplicates.

            columns_for_duplicates = [column for column in src_df.columns if column != "data_mode"]

            src_df = src_df.unique(subset=columns_for_duplicates)

            print("Data shape after dropping duplicates", src_df.shape)

            return src_df
        
        else:
            df = df.drop(['index', 'agent1', 'agent2'], strict=False)

            return df
    
    
    def drop_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Drop the certain columns."""
        
        columns_to_drop = [
            'GameRulesetName',
        ]
        
        if self.config.is_train:
            columns_to_drop += [
                'num_wins_agent1',
                'num_draws_agent1',
                'num_losses_agent1',
            ]
        
        df = df.drop(columns_to_drop, strict=False)
        
        print('Shape after dropping specific columns:', df.shape)
        
        if self.config.is_train:
            print('\n', df["fold"].value_counts())
        
        return df
    
    def postprocessing(self, df: pl.DataFrame) -> pd.DataFrame:
        """Adjust data types."""
        
        df = df.to_pandas()
        
        if self.config.is_train:
            cat_mapping = {feature: pd.CategoricalDtype(categories=list(set(df[feature]))) for feature in df.columns[df.dtypes == object]}
            catcols = list(cat_mapping.keys())
            catcols = [column for column in catcols if 'tta' not in column]
            self.data_checkpoint["catcols"] += catcols

        if self.config.task == "classification":
            df["utility_agent1"] = df["utility_agent1"].astype(str)
        
        return df
    
    def save_data_checkpoint(self) -> None:
        """Save data checkpoint."""
        
        if self.config.is_train:
            with open(self.config.path_to_save_data_checkpoint, "wb") as f:
                pickle.dump(self.data_checkpoint, f)
        
    
    def get_dataset(self, df: pl.DataFrame) -> Tuple[pd.DataFrame, list]:
        """Get the dataset."""
        
        df = self.preprocessing(df)
        df = self.feature_generation(df)
        df = self.build_validation_and_cv_features(df)
        df = self.drop_columns(df)
        df = self.postprocessing(df)
        self.save_data_checkpoint()
        
        return df, self.data_checkpoint
    

# --- Solver ---

class Solver:
    """Solution."""

    def __init__(self, config: Config, rerun: bool) -> None:
        """Initialiaztion."""
        
        self.config = config
        self.rerun = rerun
        
        try:
            with open(config.path_to_load_solver_checkpoint, "rb") as f:
                self.models = pickle.load(f)
        except FileNotFoundError:
            self.models = {}

    def generate_TF_IDF(self, df, mode, fold, n_tf_ids_features):
        """Generate TF-IDF features."""

        def pickle_dump(obj, path):
            with open(path, mode="wb") as f:
                dill.dump(obj, f, protocol=4)
                
        def pickle_load(path):
            with open(path, mode="rb") as f:
                data = dill.load(f)
                return data

        str_cols = ['EnglishRules', 'LudRules']

        if mode == 'train': self.models["tfidf_paths"] = []

        for col in str_cols:
            df[f'{col}_len'] = df[col].apply(len)

            if mode == 'train':
                tfidf = TfidfVectorizer(max_features=n_tf_ids_features, ngram_range=(2, 3))
                tfidf_feats = tfidf.fit_transform(df[col]).toarray()
                for i in range(tfidf_feats.shape[1]):
                    df[f"{col}_tfidf_{i}"] = tfidf_feats[:, i]
                pickle_dump(tfidf, os.path.join(self.config.path_to_tfidf, f'tfidf_{fold}_{col}.model'))
                self.models["tfidf_paths"].append(os.path.join(self.config.path_to_tfidf, f'tfidf_{fold}_{col}.model'))

            else:
                for i in range(len(self.models["tfidf_paths"])):
                    if f'tfidf_{fold}_{col}.model' == self.models["tfidf_paths"][i].split('/')[-1]:
                        tfidf = pickle_load(os.path.join(self.config.path_to_tfidf, f'tfidf_{fold}_{col}.model'))
                        tfidf_feats = tfidf.transform(df[col]).toarray()
                        for j in range(tfidf_feats.shape[1]):
                            df[f"{col}_tfidf_{j}"] = tfidf_feats[:, j]

        df = df.drop(str_cols, axis=1)
        return df
    
    def train_one_model(self, df, X, Y, catcols, model_name) -> Tuple[np.array, np.array, Union[pl.DataFrame, None]]:
        """Train N folds of a certain model."""
        
        # Initialize.
        
        if model_name not in self.models:
            self.models[model_name] = {
                "features": None,
                "oof_score": None,
                "models": [],
            }
            
        if len(self.models[model_name]["models"]) != self.config.n_splits:
            self.models[model_name]["models"] = [None] * self.config.n_splits
        
        # Define the instances for the metrics.

        scores = []
        oof_preds, oof_labels = np.array([]), np.array([])

        if model_name == "catboost":
            feature_importances = None

        for fold in range(self.config.n_splits):    
            
            # Get the fold's data.
            
            print(f"\n{model_name} | Fold {fold}")
            
            train_index = df[df["fold"] != fold].index
            valid_index = df[df["fold"] == fold].index

            X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
            Y_train, Y_valid = Y.iloc[train_index], Y.iloc[valid_index]

            mask = X_valid["data_mode"] == "original"
            X_valid = X_valid[mask]
            Y_valid = Y_valid[mask]

            X_train = X_train.drop(["data_mode"], axis=1)
            X_valid = X_valid.drop(["data_mode"], axis=1)

            # TF-IDF.

            if self.config.n_tf_ids_features != 0:
                X_train = self.generate_TF_IDF(X_train, mode='train', fold=fold, n_tf_ids_features=self.config.n_tf_ids_features)
                X_valid = self.generate_TF_IDF(X_valid, mode='test', fold=fold, n_tf_ids_features=self.config.n_tf_ids_features)
                print('Shape with TF-IDF features', X_train.shape, X_valid.shape)
            else:
                X_train = X_train.drop(['EnglishRules', 'LudRules'], axis=1)
                X_valid = X_valid.drop(['EnglishRules', 'LudRules'], axis=1)

            # TTA processing.

            src_columns = [column for column in X_train.columns.tolist() if 'tta' not in column and column != 'data_mode'] 
            tta_columns = [column.replace('src', 'tta') for column in src_columns if column != 'data_mode']

            X_train = X_train[src_columns]
            X_valid_src = X_valid[src_columns]
            X_valid_tta = X_valid[tta_columns].rename(columns={column: column.replace('tta', 'src') for column in tta_columns})

            print('Original features shape', X_train.shape)
            
            # Apply the new features.

            with open(self.config.path_to_load_features, 'rb') as file:
                ofe_features = pickle.load(file)

                operators = ["abs", "log", "sqrt", "square", "sigmoid", "round", "residual", "min", "max", "+", "-", "*", "/"]

                ofe_features_basic = ofe_features[:self.config.n_openfe_features[0]]
                ofe_features_num = [feature for feature in ofe_features[self.config.n_openfe_features[0]:] if feature.name in operators]
                basic_num_count = len([f for f in ofe_features_basic if f.name in operators]) 
                ofe_features_num = ofe_features_num[:max(self.config.n_openfe_features[1], basic_num_count) - basic_num_count]

                ofe_features = ofe_features_basic + ofe_features_num

            if self.config.n_openfe_features != (0, 0):
                _, X_valid_src = transform(X_train[:10], X_valid_src, ofe_features, n_jobs=1)
                X_train, X_valid_tta = transform(X_train, X_valid_tta, ofe_features, n_jobs=1)

            X_train = X_train.drop([column for column in X_train.columns if 'index' in column], axis=1)
            X_valid_src = X_valid_src.drop([column for column in X_valid_src.columns if 'index' in column], axis=1)
            X_valid_tta = X_valid_tta.drop([column for column in X_valid_tta.columns if 'index' in column], axis=1)

            print('Shape with OpenFE features', X_train.shape)

            # Categorical mapping.

            cat_mapping, catcols = {}, []
            for feature in X_train.columns:
                if X_train[feature].dtype == object:
                    cat_mapping[feature] = pd.CategoricalDtype(categories=list(set(X_train[feature])))
                    catcols.append(feature)
                else:
                    cat_mapping[feature] = float
    
            X_train = X_train.astype(cat_mapping)
            X_valid_src = X_valid_src.astype(cat_mapping)
            X_valid_tta = X_valid_tta.astype(cat_mapping)
            
            # Create and fit the model.
            
            if not self.rerun:
                
                if model_name == "catboost":
                    if self.config.task == "classification":
                        model = CatBoostClassifier(**self.config.catboost_params, cat_features=catcols)
                    else:
                        model = CatBoostRegressor(**self.config.catboost_params, cat_features=catcols)
                    model.fit(X_train, Y_train, eval_set=(X_valid_src, Y_valid))
                    
                elif model_name == "lgbm":
                    model = lgb.LGBMRegressor(**self.config.lgbm_params)
                    model.fit(X_train, Y_train,
                      eval_set=[(X_valid_src, Y_valid)],
                      eval_metric='rmse',
                      callbacks=[
                          lgb.early_stopping(self.config.catboost_params["early_stopping_rounds"]),
                           lgb.log_evaluation(self.config.catboost_params["verbose"])
                      ])
            else:
                model = self.models[model_name]["models"][fold]
            
            # Prediction (with TTA).
            
            if self.config.task == "classification":
                Y_valid = Y_valid.astype(np.float)
                preds_original = model.predict(X_valid_src).astype(float)
                preds_tta = model.predict(X_valid_tta).astype(float) * -1
                preds = (preds_original + preds_tta) / 2
                preds = preds[0]
            else:
                preds_original = model.predict(X_valid_src)
                preds_tta = model.predict(X_valid_tta) * -1
                preds = (preds_original + preds_tta) / 2
            
            # Save the scores and the metrics.
            
            oof_preds =  np.concatenate((oof_preds, preds_original))
            oof_labels =  np.concatenate((oof_labels, Y_valid))

            score_original = mean_squared_error(Y_valid, preds_original, squared=False)
            score_tta = mean_squared_error(Y_valid, preds_tta, squared=False)
            score = mean_squared_error(Y_valid, preds, squared=False)

            scores.append(score_original)

            print(round(score_original, 4))
            print(round(score_tta, 4))
            print(round(score, 4))    
            
            if not self.rerun:
                self.models[model_name]["models"][fold] = model
                
            if model_name == "catboost":
                if feature_importances is None:
                    feature_importances = model.get_feature_importance() / self.config.n_splits
                else:
                    feature_importances += model.get_feature_importance() / self.config.n_splits
                fi = pl.DataFrame({
                    "feature": X_train.columns,
                    "importance": feature_importances
                }).sort(by='importance', descending=True)
                print(fi.head(10))
            
            gc.collect()
        
        # Clip final predictions. 
        
        oof_preds = np.clip(oof_preds, -1, 1)
        
        # Count and display the scores.
            
        oof_score = mean_squared_error(oof_labels, oof_preds, squared=False)
        
        if not self.rerun:
            self.models[model_name]["features"] = src_columns
            self.models[model_name]["oof_score"] = oof_score
        
        print(f'\nCV scores {model_name}')
        for fold in range(len(scores)):
            print(f'Fold {fold} | {round(scores[fold], 4)}')
        
        print("AVG", round(np.mean(scores), 4))
        print("STD", round(np.std(scores), 4))
        print("OOF", round(oof_score, 4))
        
        if model_name == "catboost":
            feature_importances = pd.DataFrame({
                "feature": X_train.columns,
                "importance": feature_importances
            }).sort_values(by='importance', ascending=False)
            print(feature_importances.head(10))
            feature_importances.to_csv('dataset/feature_importance.csv')
        else:
            feature_importances = None
        
        return oof_labels, oof_preds, feature_importances
            
    def fit(self, df: pl.DataFrame, data_checkpoint: dict, oof_features=None) -> dict:
        """Training."""
        
        # Select the feature and the targets.
        
        X = df.drop(['utility_agent1', 'fold'], axis=1).rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x)).reset_index()
        Y = df['utility_agent1']
        catcols = data_checkpoint["catcols"]
        
        if oof_features is not None:
            X["oof"] = oof_features
            
        # Train models.
        
        artifacts = {}
        
        for model_name, to_train in self.config.to_train.items():
            if to_train:
                artifacts[model_name] = {}

                oof_labels, oof_preds, feature_importance = self.train_one_model(df, X, Y, catcols, model_name)

                artifacts[model_name]["oof_preds"] = oof_preds
                artifacts[model_name]["oof_labels"] = oof_labels
                artifacts[model_name]["feature_importance"] = feature_importance
        
        # Save solution checkpoint for the inference.
        
        if self.config.is_train and not self.rerun:
            with open(self.config.path_to_save_solver_checkpoint, "wb") as f:
                pickle.dump(self.models, f)
                
        return artifacts
            
    def predict(self, X: pd.DataFrame, df_train: pd.DataFrame, data_checkpoint: dict) -> np.array:
        """Inference."""
        
        prediction = np.zeros(len(X))

        src_columns = [column for column in X.columns.tolist() if 'tta' not in column]
        tta_columns = [column.replace('src', 'tta') for column in src_columns]

        # Apply OpenFE features.

        with open(self.config.path_to_load_features, 'rb') as file:
            ofe_features = pickle.load(file)

            operators = ["abs", "log", "sqrt", "square", "sigmoid", "round", "residual", "min", "max", "+", "-", "*", "/"]
            ofe_features_basic = ofe_features[:self.config.n_openfe_features[0]]
            ofe_features_num = [feature for feature in ofe_features[self.config.n_openfe_features[0]:] if feature.name in operators]
            ofe_features_num = ofe_features_num[:self.config.n_openfe_features[1] - len([f for f in ofe_features_basic if f.name in operators])]

            ofe_features = ofe_features_basic + ofe_features_num

        X = X.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x)).reset_index()
        X_valid_src = X[src_columns].reset_index()
        X_valid_tta = X[tta_columns].rename(columns={column: column.replace('tta', 'src') for column in tta_columns}).reset_index()

        if self.config.n_openfe_features != (0, 0):
            _, X_valid_src = transform(X_valid_src[:1], X_valid_src, ofe_features, n_jobs=1)
            _, X_valid_tta = transform(X_valid_tta[:1], X_valid_tta, ofe_features, n_jobs=1)
        
        X_valid_src = X_valid_src.drop([column for column in X_valid_src.columns if 'index' in column], axis=1)
        X_valid_tta = X_valid_tta.drop([column for column in X_valid_tta.columns if 'index' in column], axis=1)

        # Categorical mapping.

        catcols = ['src_p1_selection', 'src_p1_exploration', 'src_p1_playout', 'src_p1_bounds', 
                   'src_p2_selection', 'src_p2_exploration', 'src_p2_playout', 'src_p2_bounds', 'src_agent1', 'src_agent2']

        cat_mapping = {column: float for column in X_valid_src.columns}
        for column in catcols: cat_mapping[column] = pd.CategoricalDtype(categories=list(set(X_valid_src[column])))

        X_valid_src = X_valid_src.astype(cat_mapping)
        X_valid_tta = X_valid_tta.astype(cat_mapping)

        # Inference.
        
        for model_name, weight in self.config.weights.items():
            
            if model_name not in self.models: continue
            
            preds = np.zeros(len(X))

            for fold in range(self.config.n_splits):    

                model = self.models[model_name]["models"][fold]
                    
                preds_original = model.predict(X_valid_src)
                # preds_tta = model.predict(X_valid_tta) * -1
                preds += (preds_original + preds_original) / 2 / self.config.n_splits

            prediction += np.clip(preds, -1, 1) * weight
            
        return prediction
    

# --- Train and inference ---

def train(rerun: bool, oof_features=None) -> dict:
    """Training function."""
    
    config = Config()
    
    set_seed(config.seed)
    dataset = Dataset(config)
    solver = Solver(config, rerun)
    
    df = pl.read_csv(config.path_to_train_dataset)
    df, data_checkpoint = dataset.get_dataset(df)
    artifacts = solver.fit(df, data_checkpoint, oof_features)
    
    return artifacts


if not IS_TRAIN:
    
    config = Config()   
    
    set_seed(config.seed)
    dataset = Dataset(config)
    solver = Solver(config, rerun=False)

    df_train = pl.read_csv(config.path_to_train_dataset)
    df_train, _ = dataset.get_dataset(df_train)

    def predict(test: pl.DataFrame, sample_sub: pl.DataFrame) -> pl.DataFrame:
        """Inference function."""
        
        df, data_checkpoint = dataset.get_dataset(test)
        preds = solver.predict(df, df_train, data_checkpoint)

        # preds[df["GameTreeComplexity"] == 0] = 2 * df[df["GameTreeComplexity"] == 0]["AdvantageP1"] - 1

        return sample_sub.with_columns(pl.Series('utility_agent1', preds))
    

# --- Run ---

if IS_TRAIN:
    artifacts = train(rerun=IS_RERUN, oof_features=None)
else:
    inference_server = kaggle_evaluation.mcts_inference_server.MCTSInferenceServer(predict)
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        inference_server.serve()
    else:
        inference_server.run_local_gateway(
            (
                '/home/toefl/K/MCTS/dataset/test.csv' if LOCAL else '/kaggle/input/um-game-playing-strength-of-mcts-variants/test.csv',
                '/home/toefl/K/MCTS/dataset/sample_submission.csv' if LOCAL else '/kaggle/input/um-game-playing-strength-of-mcts-variants/sample_submission.csv'
            )
        )