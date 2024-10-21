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
from tqdm import tqdm

import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier

from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
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

    n_openfe_features = (0, 500)    # (All, numerical)
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
        'objective': 'regression',
        'min_child_samples': 24,
        'num_iterations': 30000,
        'learning_rate': 0.01,
        'extra_trees': True,
        'reg_lambda': 0.8,
        'reg_alpha': 0.1,
        'num_leaves': 64,
        'metric': 'rmse',
        'device': 'cpu',
        'max_depth': 9,
        'max_bin': 128,
        'verbose': -1,
        'seed': 42
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
            pl.col('agent1').alias('src_p1_agent'),
            pl.col('agent2').alias('src_p2_agent'),
            pl.col('agent1').alias('tta_p2_agent'),
            pl.col('agent2').alias('tta_p1_agent'),
            pl.col('AdvantageP1').alias('src_AdvantageP1'),
            (1 - pl.col('AdvantageP1')).alias('tta_AdvantageP1'),
        )

        df = df.drop(['AdvantageP1'], strict=False)

        # Feature engineering.

        df = df.with_columns([
            (pl.col('PlayoutsPerSecond') / (pl.col('MovesPerSecond') + 1e-15)).alias('Playouts/Moves'),
            (pl.col('MovesPerSecond') / (pl.col('PlayoutsPerSecond') + 1e-15)).alias('EfficiencyPerPlayout'),
            (pl.col('DurationActions') / (pl.col('DurationTurnsStdDev') + 1e-15)).alias('TurnsDurationEfficiency'),
            (pl.col('DurationActions') / (pl.col('MovesPerSecond') + 1e-15)).alias('ActionTimeEfficiency'),
            (pl.col('DurationTurnsStdDev') / (pl.col('DurationActions') + 1e-15)).alias('StandardizedTurnsEfficiency'),
            (pl.col('DurationActions') / (pl.col('StateTreeComplexity') + 1e-15)).alias('DurationToComplexityRatio'),
            (pl.col('GameTreeComplexity') / (pl.col('StateTreeComplexity') + 1e-15)).alias('NormalizedGameTreeComplexity'),
            (pl.col('Balance') * pl.col('GameTreeComplexity')).alias('ComplexityBalanceInteraction'),
            (pl.col('StateTreeComplexity') + pl.col('GameTreeComplexity')).alias('OverallComplexity'),
            (pl.col('GameTreeComplexity') / (pl.col('PlayoutsPerSecond') + 1e-15)).alias('ComplexityPerPlayout'),
            (pl.col('DurationTurnsNotTimeouts') / (pl.col('MovesPerSecond') + 1e-15)).alias('TurnsNotTimeouts/Moves'),
            (pl.col('Timeouts') / (pl.col('DurationActions') + 1e-15)).alias('Timeouts/DurationActions'),
            (pl.col('StepDecisionToEnemy') + pl.col('SlideDecisionToEnemy') + pl.col('HopDecisionMoreThanOne')).alias('ComplexDecisionRatio'),
            (pl.col('StepDecisionToEnemy') + 
             pl.col('HopDecisionEnemyToEnemy') + 
             pl.col('HopDecisionFriendToEnemy') + 
             pl.col('SlideDecisionToEnemy')).alias('AggressiveActionsRatio'),

            (pl.col('src_AdvantageP1') / (pl.col('Balance') + 1e-15)).alias('src_AdvantageBalanceRatio'),
            (pl.col('src_AdvantageP1') / (pl.col('DurationActions') + 1e-15)).alias('src_AdvantageTimeImpact'),
            (pl.col('OutcomeUniformity') / (pl.col('src_AdvantageP1') + 1e-15)).alias('src_OutcomeUniformity/AdvantageP1'),

            (pl.col('tta_AdvantageP1') / (pl.col('Balance') + 1e-15)).alias('tta_AdvantageBalanceRatio'),
            (pl.col('tta_AdvantageP1') / (pl.col('DurationActions') + 1e-15)).alias('tta_AdvantageTimeImpact'),
            (pl.col('OutcomeUniformity') / (pl.col('tta_AdvantageP1') + 1e-15)).alias('tta_OutcomeUniformity/AdvantageP1'),
        ])
        
        df = df.to_pandas()

        # Cross-features.

        cols = ["agent", "selection", "exploration", "playout", "bounds"]
        for c1 in cols:
            for c2 in cols:
                df[f"src_{c1}_{c2}"] = df[f"src_p1_{c1}"].astype(str) + df[f"src_p1_{c2}"].astype(str)
                df[f"tta_{c1}_{c2}"] = df[f"tta_p1_{c1}"].astype(str) + df[f"tta_p1_{c2}"].astype(str)

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

                valid_columns = ['autoFE_f_112', 'autoFE_f_468', 'autoFE_f_386', 'autoFE_f_6', 'autoFE_f_333', 'autoFE_f_357', 'autoFE_f_353', 'autoFE_f_194', 'autoFE_f_59', 'autoFE_f_174', 'autoFE_f_191', 'autoFE_f_182', 'autoFE_f_436', 'autoFE_f_261', 'autoFE_f_328', 'autoFE_f_189', 'autoFE_f_8', 'autoFE_f_275', 'autoFE_f_279', 'autoFE_f_223', 'autoFE_f_154', 'autoFE_f_319', 'autoFE_f_221', 'autoFE_f_218', 'autoFE_f_380', 'autoFE_f_402', 'autoFE_f_276', 'autoFE_f_1', 'autoFE_f_253', 'autoFE_f_362', 'autoFE_f_294', 'autoFE_f_108', 'autoFE_f_484', 'autoFE_f_11', 'autoFE_f_200', 'autoFE_f_356', 'autoFE_f_491', 'autoFE_f_2', 'autoFE_f_248', 'autoFE_f_176', 'autoFE_f_449', 'autoFE_f_335', 'autoFE_f_310', 'autoFE_f_479', 'autoFE_f_322', 'autoFE_f_446', 'autoFE_f_198', 'autoFE_f_116', 'autoFE_f_206', 'autoFE_f_214']

                X_train = X_train.drop([column for column in X_train.columns if ('autoFE' in column) & (column not in valid_columns)], axis=1)
                X_valid_src = X_valid_src.drop([column for column in X_valid_src.columns if ('autoFE' in column) & (column not in valid_columns)], axis=1)
                X_valid_tta = X_valid_tta.drop([column for column in X_valid_tta.columns if ('autoFE' in column) & (column not in valid_columns)], axis=1)

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
                fi.write_csv('dataset/feature_importance.csv')
            
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

        # TF-IDF.

        if self.config.n_tf_ids_features != 0:
            X = self.generate_TF_IDF(X, mode='test', fold=0, n_tf_ids_features=self.config.n_tf_ids_features)
            print('Shape with TF-IDF features', X.shape)
        else:
            X = X.drop(['EnglishRules', 'LudRules'], axis=1)

        # TTA.
        
        X = X.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x)).reset_index()

        src_columns = [column for column in X.columns.tolist() if 'tta' not in column]
        tta_columns = [column.replace('src', 'tta') for column in src_columns]

        X_valid_src = X[src_columns]
        X_valid_tta = X[tta_columns].rename(columns={column: column.replace('tta', 'src') for column in tta_columns})

        # Apply OpenFE features.

        with open(self.config.path_to_load_features, 'rb') as file:
            ofe_features = pickle.load(file)

            operators = ["abs", "log", "sqrt", "square", "sigmoid", "round", "residual", "min", "max", "+", "-", "*", "/"]

            ofe_features_basic = ofe_features[:self.config.n_openfe_features[0]]
            ofe_features_num = [feature for feature in ofe_features[self.config.n_openfe_features[0]:] if feature.name in operators]
            basic_num_count = len([f for f in ofe_features_basic if f.name in operators]) 
            ofe_features_num = ofe_features_num[:max(self.config.n_openfe_features[1], basic_num_count) - basic_num_count]

            ofe_features = ofe_features_basic + ofe_features_num

        if self.config.n_openfe_features != (0, 0):
            _, X_valid_src = transform(X_valid_src[:1], X_valid_src, ofe_features, n_jobs=1)
            _, X_valid_tta = transform(X_valid_tta[:1], X_valid_tta, ofe_features, n_jobs=1)
        
        X_valid_src = X_valid_src.drop([column for column in X_valid_src.columns if 'index' in column], axis=1)
        X_valid_tta = X_valid_tta.drop([column for column in X_valid_tta.columns if 'index' in column], axis=1)

        # Categorical mapping.

        cat_mapping = {}
        for feature in X_valid_src.columns:
            if X_valid_src[feature].dtype == object:
                cat_mapping[feature] = pd.CategoricalDtype(categories=list(set(X_valid_src[feature])))
            else:
                cat_mapping[feature] = float

        X_valid_src = X_valid_src.astype(cat_mapping)
        
        cat_mapping = {}
        for feature in X_valid_tta.columns:
            if X_valid_tta[feature].dtype == object:
                cat_mapping[feature] = pd.CategoricalDtype(categories=list(set(X_valid_tta[feature])))
            else:
                cat_mapping[feature] = float
                
        X_valid_tta = X_valid_tta.astype(cat_mapping)

        # Inference.

        prediction = np.zeros(len(X))
        
        for model_name, weight in self.config.weights.items():
            
            if model_name not in self.models: continue
            
            preds = np.zeros(len(X))

            for fold in range(self.config.n_splits):    

                model = self.models[model_name]["models"][fold]
                    
                preds_original = model.predict(X_valid_src)
                preds_tta = model.predict(X_valid_tta) * -1
                preds += (preds_original + preds_tta) / 2 / self.config.n_splits

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