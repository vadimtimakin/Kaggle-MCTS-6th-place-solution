# --- Imports ---

import re
import os
import gc
import dill
import random
import pickle

import shap
import numpy as np
import polars as pl
import pandas as pd

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier, Pool

import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset

from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, QuantileTransformer

import joblib
from tqdm import tqdm
from typing import Tuple, Union
from itertools import permutations, combinations

import sys
sys.path.append('/home/toefl/K/MCTS/openfe')
sys.path.append('/home/toefl/K/MCTS/dataset/')
sys.path.append('/kaggle/input/openfe-modified')

try:
    from optimizer import Ranger, MADGRAD
except ImportError:
    pass

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

    path_to_load_features = 'feature.pickle' if LOCAL else '/kaggle/input/mcts-solution-checkpoint/feature.pickle'
    path_to_tfidf = '/home/toefl/K/MCTS/dataset/tf_idf' if LOCAL else '/kaggle/input/mcts-solution-checkpoint/tf_idf'

    path_to_save_data_checkpoint = 'checkpoints/data_checkpoint_xgboost_5fold'     # Drop columns, categorical columns, etc.
    path_to_save_solver_checkpoint = 'checkpoints/solver_checkpoint_xgboost_5fold' # Models, weights, etc.

    path_to_load_data_checkpoint = 'checkpoints/data_checkpoint_xgboost_5fold' if LOCAL else '/kaggle/input/mcts-solution-checkpoint/data_checkpoint.pickle'
    path_to_load_solver_checkpoint = {
        "main": 'checkpoints/solver_checkpoint_stacked' if LOCAL else '/kaggle/input/mcts-solution-checkpoint/solver_checkpoint_stacked',
        "baseline": 'checkpoints/solver_checkpoint_stacked' if LOCAL else '/kaggle/input/mcts-solution-checkpoint/solver_checkpoint_stacked',
    }

    path_to_save_dataset = 'checkpoints/dataset.csv'
    load_dataset_checkpoint = True
    
    task = "regression"
    
    n_splits = 5

    pl_power = 0
    n_openfe_features = (0, 500, 0)
    n_tf_ids_features = 0

    use_oof = True
    use_baseline_scores = False
    show_shap = False
    mask_filter = False
    stacked = False
    
    catboost_params = {
        'iterations': 30000,
        'learning_rate': 0.01,
        'depth': 10,
        'early_stopping_rounds': 200,
        
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',

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

    xgb_params = {
        'n_estimators': 30000,
        'learning_rate': 0.01,
        'max_depth': 10,
        'objective': 'reg:squarederror',
        'enable_categorical': True,
        'eval_metric': 'rmse',
        'tree_method': 'gpu_hist',
        'verbosity': 1,
        'n_jobs': 14,
        'early_stopping_rounds': 200,
        'seed': 0xFACED,
    }

    dnn_params = {
        "output_size": 1,
        "epochs": 20,
        "batch_size": 1024,
        "learning_rate": 0.003,
        "device": "cuda",
    }
    
    to_train = {
        "catboost": False,
        "lgbm": False,
        "xgboost": False,
        "DNN": True,
    }
    
    weights = {
        "catboost": 1,
        "lgbm": 0,
        "xgboost": 0,
        "DNN": 0,
    }


# --- Utils ---

def set_seed(SEED):
    """Set random seeds."""
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --- Custom DNN model ---

class DNN(nn.Module):

    def __init__(self, output_size=1, epochs=100, batch_size=32, learning_rate=0.001, 
                 device='cpu', early_stopping_patience=10):
        super(DNN, self).__init__()

        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience

        self.output_size = output_size
        self.embeddings = None
        self.scaler = None
        self.cat_scaler = None
        self.input_size = 0
        self.cat_column_names = []

    def _build_model(self):
        num_numerical_features = len(self.num_columns)

        self.embedding_mlp = nn.Embedding(sum(self.field_dims), 128)

        input_dim = (len(self.field_dims) * 128) + num_numerical_features  
        print("DNN input_dim |", input_dim)

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.Hardswish(),

            nn.Dropout(0.9),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.Hardswish(),

            nn.Dropout(0.9),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.Hardswish(),

            nn.Dropout(0.9),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Hardswish(),

            nn.Dropout(0.9),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.Hardswish(),

            nn.Linear(128, 1),
        )

    def forward(self, x_num, x_cat):
       
        mlp_embed_x = self.embedding_mlp(x_cat)
        mlp_x = torch.cat([mlp_embed_x.view(mlp_embed_x.size(0), -1), x_num], dim=1)
        mlp_out = self.mlp(mlp_x)

        return mlp_out

    def fit(self, X_train, Y_train, X_val, Y_val):
        if isinstance(X_train, pd.DataFrame) and isinstance(X_val, pd.DataFrame):
           
            self.cat_column_names = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
            self.num_columns = X_train.select_dtypes(exclude=['object', 'category']).columns.tolist()
            self.field_dims = [X_train[col].nunique() for col in self.cat_column_names]

            self._build_model() 
            self.to(self.device)

            X_train_num = X_train[self.num_columns]
            X_val_num = X_val[self.num_columns]

            self.scaler = QuantileTransformer()
            self.scaler.fit(X_train_num.drop(["oof"], axis=1).values)

            X_train_num_scaled = torch.tensor(np.concatenate([self.scaler.transform(X_train_num.drop(["oof"], axis=1).values), np.expand_dims(X_train_num["oof"].values, axis=1)], axis=1)).float().to(self.device)
            X_val_num_scaled = torch.tensor(np.concatenate([self.scaler.transform(X_val_num.drop(["oof"], axis=1).values), np.expand_dims(X_val_num["oof"].values, axis=1)], axis=1)).float().to(self.device)

            X_train_cat = X_train[self.cat_column_names].astype('category').apply(lambda x: x.cat.codes)
            X_val_cat = X_val[self.cat_column_names].astype('category').apply(lambda x: x.cat.codes)

            X_train_cat_tensor = torch.tensor(X_train_cat.values).long().to(self.device)
            X_val_cat_tensor = torch.tensor(X_val_cat.values).long().to(self.device)

            Y_train_tensor = torch.tensor(Y_train.values).float().to(self.device).unsqueeze(-1)
            Y_val_tensor = torch.tensor(Y_val.values).float().to(self.device).unsqueeze(-1)

            train_dataset = TensorDataset(X_train_num_scaled, X_train_cat_tensor, Y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

            val_dataset = TensorDataset(X_val_num_scaled, X_val_cat_tensor, Y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

            criterion = lambda outputs, targets: torch.sqrt(nn.MSELoss()(outputs, targets))
            optimizer = MADGRAD(self.parameters(), lr=self.learning_rate)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-8)

            scaler = GradScaler()

            best_val_loss = np.inf
            patience_counter = 0

            for epoch in range(self.epochs):
                print(f"\nEpoch {epoch + 1}/{self.epochs}")
                
                self.train()
                train_loss = 0.0
                train_batches = tqdm(train_loader, desc="Train Batches", leave=False) 

                for batch_idx, (batch_X_num, batch_X_cat, batch_Y) in enumerate(train_batches):
                    batch_X_num, batch_X_cat, batch_Y = batch_X_num.to(self.device), batch_X_cat.to(self.device), batch_Y.to(self.device)
                    
                    optimizer.zero_grad()
                    
                    with autocast():
                        outputs = self.forward(batch_X_num, batch_X_cat)
                        loss = criterion(outputs, batch_Y)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    train_loss += loss.item()
                    train_batches.set_postfix({'Current Train Loss': loss.item()})
                
                train_loss /= len(train_loader)
                print(f"Training Loss: {train_loss:.4f}", end=' ')
                
                self.eval()
                val_loss = 0.0
                all_preds = []
                all_labels = []
                val_batches = tqdm(val_loader, desc="Val Batches", leave=False)
                
                with torch.no_grad():
                    for batch_idx, (batch_X_num, batch_X_cat, batch_Y) in enumerate(val_batches):
                        batch_X_num, batch_X_cat, batch_Y = batch_X_num.to(self.device), batch_X_cat.to(self.device), batch_Y.to(self.device)
                        
                        with autocast():
                            outputs = self.forward(batch_X_num, batch_X_cat)
                            loss = criterion(outputs, batch_Y)
                        
                        val_loss += loss.item()
                        all_preds.append(outputs.cpu().numpy())
                        all_labels.append(batch_Y.cpu().numpy())
                        val_batches.set_postfix({'Current Val Loss': loss.item()})
                
                val_loss /= len(val_loader)
                all_preds = np.concatenate(all_preds)
                all_labels = np.concatenate(all_labels)
                val_rmse = mean_squared_error(all_labels, all_preds, squared=False)
                print(f"Validation Loss: {val_loss:.4f}, Validation RMSE: {val_rmse:.4f}")
                
                if val_rmse < best_val_loss:
                    best_val_loss = val_rmse 
                    patience_counter = 0
                    self.save('checkpoints/tmp.pt')
                else:
                    patience_counter += 1
                
                if patience_counter >= self.early_stopping_patience:
                    break
                
                scheduler.step()

        else:
            raise ValueError("X_train and X_val must be pandas DataFrames")

    def predict(self, X):
        self.eval()
        self.load('checkpoints/tmp.pt')
        with torch.no_grad():
            X_num = X.select_dtypes(exclude=['object', 'category'])
            X_num_scaled = torch.tensor(np.concatenate([self.scaler.transform(X_num.drop(["oof"], axis=1).values), np.expand_dims(X_num["oof"].values, axis=1)], axis=1)).float().to(self.device)

            X_cat = X[self.cat_column_names].astype('category').apply(lambda x: x.cat.codes).values
            X_cat_tensor = torch.tensor(X_cat).long().to(self.device)

            dataset = TensorDataset(X_num_scaled, X_cat_tensor)
            data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

            predictions = []
            print('Prediction')
            for X_num_batch, X_cat_batch in tqdm(data_loader):
                preds = self.forward(X_num_batch, X_cat_batch)
                predictions.append(preds.cpu().numpy())
            return np.squeeze(np.vstack(predictions))

    def save(self, file_path):
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'scaler': self.scaler,
        }
        joblib.dump(checkpoint, file_path)

    def load(self, file_path):
        checkpoint = joblib.load(file_path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = checkpoint['scaler']


# --- Data pipeline ---

class Dataset:
    """The data pipeline."""
    
    def __init__(self, config: Config, rerun: bool) -> None:
        """Initialization."""
        self.config = config
        self.rerun = rerun
        
        if self.config.is_train:
            self.data_checkpoint = {}
        else:
            with open(config.path_to_load_data_checkpoint, "rb") as f:
                self.data_checkpoint = pickle.load(f)
        
    def preprocessing(self, df: pl.DataFrame) -> pl.DataFrame:
        """The basic preprocessing."""

        # Initial data shape.
        
        print("Initial shape", df.shape)

        # Mirror the dataset.
        
        if self.config.is_train:

            df = df.with_columns(pl.lit("original").alias("data_mode"))

            df_mirror = df.clone()

            df_mirror = df_mirror.with_columns(
                pl.col("agent1").alias("agent2"),
                pl.col("agent2").alias("agent1"),
                (pl.col("utility_agent1") * -1).alias("utility_agent1"),
                (1 - pl.col("AdvantageP1")).alias("AdvantageP1"),
                (1 - pl.col("Balance")).alias("Balance"),
                pl.lit("mirror").alias("data_mode")
            )

            df = pl.concat([df, df_mirror])

            with open('checkpoints/rmse_mask_full.pickle', 'rb') as file:
                rmse_mask = pickle.load(file)
                df = df.with_columns(pl.Series('mask', (rmse_mask < np.quantile(rmse_mask, 0.9))))
        
            print("Shape after data generation", df.shape)
        
        # Drop constant columns.
        
        if self.config.is_train:
            constant_columns = np.array(df.columns)[df.select(pl.all().n_unique() == 1).to_numpy().ravel()]
            drop_columns = list(constant_columns) + ['Id']
            self.data_checkpoint["dropcols"] = drop_columns
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
            pl.col('agent1').str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', 1).alias('p1_selection'),
            pl.col('agent1').str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', 2).alias('p1_exploration'),
            pl.col('agent1').str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', 3).alias('p1_playout'),
            pl.col('agent1').str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', 4).alias('p1_bounds'),
            pl.col('agent2').str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', 1).alias('p2_selection'),
            pl.col('agent2').str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', 2).alias('p2_exploration'),
            pl.col('agent2').str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', 3).alias('p2_playout'),
            pl.col('agent2').str.extract(r'MCTS-(.*)-(.*)-(.*)-(.*)', 4).alias('p2_bounds'),
        )

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
            (pl.col('AdvantageP1') / (pl.col('Balance') + 1e-15)).alias('AdvantageBalanceRatio'),
            (pl.col('AdvantageP1') / (pl.col('DurationActions') + 1e-15)).alias('AdvantageTimeImpact'),
            (pl.col('OutcomeUniformity') / (pl.col('AdvantageP1') + 1e-15)).alias('OutcomeUniformity/AdvantageP1'),
        ])
        
        df = df.to_pandas()

        # Cross-features.

        df["p1_agent"] = df["agent1"]
        df["p2_agent"] = df["agent2"]

        cols = ["agent", "selection", "exploration", "playout", "bounds"]
        for c1 in cols:
            for c2 in cols:
                df[f"{c1}_{c2}"] = df[f"p1_{c1}"].astype(str) + df[f"p1_{c2}"].astype(str)

        # Apply the OpenFE features.

        df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x)).reset_index()

        with open(self.config.path_to_load_features, 'rb') as file:
            ofe_features = pickle.load(file)

            operators = ["abs", "log", "sqrt", "square", "sigmoid", "round", "residual", "min", "max", "+", "-", "*", "/"]
            
            ofe_features_basic = ofe_features[:self.config.n_openfe_features[0]]
            ofe_features_num = [feature for feature in ofe_features[self.config.n_openfe_features[0]:] if feature.name in operators]
            ofe_features_num = ofe_features_num[:self.config.n_openfe_features[1]]

            ofe_features_combine = [feature for feature in ofe_features if feature.name == "Combine"][:self.config.n_openfe_features[2]]

            ofe_features = ofe_features_basic + ofe_features_num

        if self.config.n_openfe_features != (0, 0):
            valid_features = sorted([112, 468, 386, 6, 333, 357, 353, 194, 59, 174, 191, 182, 436, 261, 328, 189, 8, 275, 279, 223, 154, 319, 221, 218, 380, 402, 276, 1, 253, 362, 294, 108, 484, 11, 200, 356, 491, 2, 248, 176, 449, 335, 310, 479, 322, 446, 198, 116, 206, 214])
            df, _ = transform(df, df[:10], ofe_features, valid_features, name="num", n_jobs=1)
            if self.config.n_openfe_features[2] != 0:
                df, _ = transform(df, df[:10], ofe_features_combine, None, name="combine", n_jobs=1)
        
            del ofe_features

        df = df.drop([column for column in df.columns if 'index' in column], axis=1)

        df = df.fillna(-100)

        print('Shape with OpenFE features', df.shape)

        # Convert back to polars.

        df = pl.from_pandas(df)

        print("Shape after feature generation", df.shape)

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

            split = cv.split(
                df,
                df["utility_agent1"].alias("utility_agent1").cast(pl.Utf8) + "_" + df["agent1"],
                df["GameRulesetName"]
            )

            pl_folds = []

            for fold, (_, index) in enumerate(split):
                print("Generating features for fold", fold)

                # Set fold index.

                df = df.with_columns(
                    pl.when(pl.col('index').is_in(index))
                    .then(pl.lit(fold))
                    .otherwise(pl.col('fold'))
                    .alias('fold')
                )

                # Pseudo labeling part.

                if self.config.pl_power != 0:

                    df_pl = df[index]

                    # Group normalization for PL.

                    columns_to_average = ['Behaviour', 'StateRepetition', 'Duration', 'Complexity', 'BoardCoverage', 'GameOutcome', 'StateEvaluation', 'Clarity', 'Decisiveness', 'Drama', 'MoveEvaluation', 'StateEvaluationDifference', 'BoardSitesOccupied', 'BranchingFactor', 'DecisionFactor', 'MoveDistance', 'PieceNumber', 'ScoreDifference']

                    columns_to_average = list(set(columns_to_average) & set(df.columns))

                    avg = df_pl.group_by("GameRulesetName").agg([
                        pl.col(column).mean().alias(f"avg_{column}") for column in columns_to_average
                    ])

                    df_pl = df_pl.join(avg, on="GameRulesetName")

                    for column in columns_to_average:
                        df_pl = df_pl.with_columns(pl.col(f"avg_{column}").alias(column))

                    df_pl = df_pl.drop([f"avg_{column}" for column in columns_to_average])

                    # New data generation for PL.

                    def generate_unique_combinations(group_df):
                        
                        existing_combinations = set(
                            tuple(pair)
                            for pair in zip(group_df['p1_agent'].to_list(), group_df['p2_agent'].to_list())
                        )

                        unique_agents = set(group_df['p1_agent'].to_list() + group_df['p2_agent'].to_list())

                        all_combinations = set(combinations(unique_agents, 2))

                        new_combinations = all_combinations - existing_combinations

                        combined_permutations = [(a, b) for comb in new_combinations for a, b in permutations(comb, 2)]

                        original_group_size = group_df.height * self.config.pl_power
                        if len(combined_permutations) >= original_group_size:
                            sampled_permutations = random.sample(combined_permutations, original_group_size)
                        else:
                            sampled_permutations = combined_permutations

                        if sampled_permutations:
                            p1_agents_new, p2_agents_new = zip(*sampled_permutations)
                        else:
                            p1_agents_new, p2_agents_new = [], []

                        group_df = pl.concat([group_df] * self.config.pl_power)

                        new_group_df = group_df.drop(['p1_agent', 'p2_agent']).with_columns([
                            pl.Series("p1_agent", p1_agents_new),
                            pl.Series("p2_agent", p2_agents_new)
                        ])

                        return new_group_df
                    
                    columns = df_pl.columns

                    df_pl = df_pl.group_by("GameRulesetName", maintain_order=True).map_groups(generate_unique_combinations)

                    df_pl = df_pl[columns]

                    # Feature encoding for PL.

                    mean_values = df_pl.group_by('GameRulesetName').mean()

                    ruleset_dict = mean_values.to_dict(as_series=False)

                    X = df_pl.with_columns(
                        df_pl['GameRulesetName'].map_elements(lambda x: ruleset_dict["utility_agent1"][ruleset_dict['GameRulesetName'].index(x)], return_dtype=pl.Float64)
                    )

                    X = X.drop(['fold', 'utility_agent1', 'num_wins_agent1','num_draws_agent1', 'num_losses_agent1', 'EnglishRules', 'LudRules', 'data_mode', 'agent1', 'mask', 'index', 'agent2']).to_pandas()
                    
                    # PL predict.
                    
                    preds = np.zeros(len(X))
                    for idx in range(self.config.n_splits):
                        with open(self.config.path_to_load_solver_checkpoint["pl"], 'rb') as file:
                            model = pickle.load(file)["catboost"]["models"][idx]

                        model.set_feature_names([f.replace('src_', '') for f in model.feature_names_])

                        cat_feature_indices = model.get_cat_feature_indices()
                        cat_feature_names = [model.feature_names_[i] for i in cat_feature_indices]
                        cat_mapping = {f: "category" if f in cat_feature_names else float for f in model.feature_names_}
                        X = X.astype(cat_mapping)[model.feature_names_]

                        preds += model.predict(X) / self.config.n_splits
                        
                        del model
                        gc.collect()

                    preds = np.clip(preds, -1, 1)

                    labels = np.array([-0.4666666666666667, -0.3333333333333333, -0.0666666666666666, 0.0666666666666666, 0.2, -0.2, -0.7333333333333333, 0.6, 0.4666666666666667, 0.3333333333333333, -0.6, 0.4, 0.1333333333333333, 0.2666666666666666, 0.5333333333333333, 0.7333333333333333, 1.0, 0.8, 0.9333333333333332, 0.8666666666666667, -0.1333333333333333, -1.0, 0.6666666666666666, -0.2666666666666666, -0.5333333333333333, 0.0, -0.4, 0.3666666666666666, -0.8666666666666667, -0.9333333333333332, -0.6666666666666666, -0.8, 0.2888888888888888, 0.0333333333333333, -0.9666666666666668, 0.5666666666666667, 0.9, -0.8333333333333334, -0.4333333333333333, -0.1666666666666666, 0.1, 0.3, -0.5, -0.5666666666666667, 0.7666666666666667, -0.7, 0.2333333333333333, -0.7666666666666667, 0.1666666666666666, -0.1, -0.2333333333333333, 0.9666666666666668, 0.8333333333333334, -0.6333333333333333, -0.0333333333333333, 0.9555555555555556, -0.9555555555555556, -0.3666666666666666, -0.3, 0.5, 0.7, 0.6333333333333333, 0.4333333333333333, -0.9, 0.0222222222222222, 0.2444444444444444, 0.4444444444444444, -0.4444444444444444])

                    preds = np.array([labels[np.abs(labels - v).argmin()] for v in preds])

                    src_df = pl.concat([src_df, df_pl.with_columns(pl.Series('utility_agent1', preds), pl.lit("pl").alias("data_mode")).drop(['index', 'fold'])])

                    pl_folds += [fold] * len(df_pl)

            if self.config.pl_power != 0: del X, df_pl

            src_df = src_df.with_columns(pl.Series("fold", np.concatenate([df["fold"].to_numpy(), df["fold"].to_numpy(), np.array(pl_folds)])))

            del df

            print("Shape after pseudo labelilng", src_df.shape)

            # Filter by RMSE mask.
            
            if self.config.mask_filter:
                src_df = src_df.filter((pl.col('mask') == True) | (pl.col('data_mode') == "original")) 
            
            src_df = src_df.drop(['mask', 'index', 'agent1', 'agent2'], strict=False)
            print("Data shape after filtering by mask", src_df.shape)

            # Drop duplicates.

            columns_for_duplicates = [column for column in src_df.columns if column != "data_mode"]
            src_df = src_df.unique(subset=columns_for_duplicates, maintain_order=True)
            print("Data shape after dropping duplicates", src_df.shape)

            return src_df
        
        else:
            df = df.drop(['index', 'agent1', 'agent2'], strict=False)

            return df
    
    
    def drop_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Drop the certain columns."""

        # Leak check first.

        if self.config.is_train:

            df = df.to_pandas()

            tmpd = df[["GameRulesetName", "fold"]].copy()
            flag = True
            for i in range(self.config.n_splits):
                for j in range(self.config.n_splits):
                    if i == j: continue
                    intersection = set(tmpd[tmpd["fold"] == i]["GameRulesetName"].unique()) & set(tmpd[tmpd["fold"] == j]["GameRulesetName"].unique())
                    if len(intersection) != 0:
                        print(i, j)
                        flag = False

            if flag: print('No groups intersections detected.')

            df = pl.from_pandas(df)

        columns_to_drop = [
            'PieceState', 'GraphStyle', 'MovesOperators', 'SowCCW', 'ScoreDifferenceMedian', 'AbsoluteDirections', 'PushEffectFrequency', 'LineWin', 'LeapDecisionToEmptyFrequency', 'AlquerqueBoardWithOneTriangle', 'TaflStyle', 'Capture', 'Even', 'RegularShape', 'SlideDecisionToFriendFrequency', 'SwapPiecesDecisionFrequency', 'AddDecision', 'LineLossFrequency', 'CheckmateFrequency', 'Multiplication', 'MoveAgain', 'TriangleTiling', 'SetSiteState', 'SwapPlayersDecision', 'RemoveDecision', 'LineOfSight', 'CaptureEnd', 'SquareTiling', 'ForwardsDirection', 'NoProgressEndFrequency', 'Draw', 'Odd', 'Parity', 'ConnectionLossFrequency', 'NoMovesWin', 'SurakartaStyle', 'Checkmate', 'TrackLoop', 'StepEffect', 'StepDecisionToFriend', 'Maximum', 'HopEffect', 'NineMensMorrisBoard', 'TriangleShape', 'FillWinFrequency', 'Style', 'FlipFrequency', 'VoteEffect', 'NoMoves', 'Meta', 'GroupEndFrequency', 'Hand', 'NoMovesEnd', 'CountPiecesMoverComparison', 'FromToDecision', 'StackType', 'IsEnemy', 'AlquerqueBoardWithFourTriangles', 'MancalaFourRows', 'Group', 'HopDecisionFriendToEnemyFrequency', 'NoOwnPiecesWinFrequency', 'CountPiecesComparison', 'VoteDecision', 'NoProgressDrawFrequency', 'RaceEnd', 'SetRotation', 'CrossBoard', 'SwapPlayersEffect', 'PieceRotation', 'ReplacementCapture', 'TerritoryWinFrequency', 'HopDecisionFriendToFriendFrequency', 'NoPieceMover', 'LineEnd', 'LeapDecision', 'PolygonShape', 'SemiRegularTiling', 'EliminatePiecesLossFrequency', 'NumDice', 'FromToDecisionFrequency', 'RemoveEffect', 'FillEndFrequency', 'CanMove', 'StarBoard', 'Track', 'PassEffect', 'ProposeDecisionFrequency', 'ConnectionEnd', 'Modulo', 'ChessComponent', 'NoProgressDraw', 'FromToDecisionEmpty', 'Scoring', 'LineLoss', 'PatternEnd', 'NoTargetPieceWinFrequency', 'NoOwnPiecesEnd', 'PatternEndFrequency', 'Efficiency', 'PenAndPaperStyle', 'ForgetValues', 'MancalaTwoRows', 'DiagonalDirection', 'HopDecision', 'PatternWinFrequency', 'StackState', 'Stack', 'StateType', 'ShowPieceState', 'AlquerqueBoardWithTwoTriangles', 'Math', 'TaflComponent', 'HopDecisionFriendToEmptyFrequency', 'InitialScore', 'PatternWin', 'SquarePyramidalShape', 'Directions', 'Pattern', 'SetMove', 'Division', 'PromotionEffect', 'ScoringLossFrequency', 'ShibumiStyle', 'ScoringWin', 'TrackOwned', 'ShowPieceValue', 'DiamondShape', 'GroupWinFrequency', 'LeapDecisionToEnemy', 'BackwardDirection', 'ScoringLoss', 'AddEffect', 'BackgammonStyle', 'ReachWin', 'Absolute', 'PieceValue', 'ScoringEnd', 'NoTargetPiece', 'HopDecisionFriendToEnemy', 'ScoringDraw', 'NoMovesLoss', 'ConnectionLoss', 'HopDecisionEnemyToEnemyFrequency', 'QueenComponent', 'PawnComponent', 'ShootDecision', 'Implementation', 'GroupEnd', 'NoMovesDrawFrequency', 'RememberValues', 'CircleTiling', 'ThreeMensMorrisBoard', 'FairyChessComponent', 'SetInternalCounter', 'BackwardLeftDirection', 'OppositeDirection', 'PromotionDecision', 'LeapEffect', 'Territory', 'Moves', 'FromToEffect', 'SlideEffect', 'SetCountFrequency', 'BishopComponent', 'CircleShape', 'ReachLoss', 'ProposeDecision', 'PloyComponent', 'XiangqiStyle', 'CheckmateWin', 'DiceD6', 'AggressiveActionsRatio', 'FromToDecisionFriend', 'ProgressCheck', 'ForwardDirection', 'LargePiece', 'HopCaptureMoreThanOne', 'DiceD4', 'LeftwardDirection', 'NoProgressEnd', 'InternalCounter', 'ByDieMove', 'FromToDecisionEnemy', 'CanNotMove', 'Minimum', 'Dice', 'Stochastic', 'HexTiling', 'SameDirection', 'PushEffect', 'ForwardLeftDirection', 'EliminatePiecesLoss', 'DirectionCapture', 'SowCapture', 'StepDecisionToEnemy', 'BackwardRightDirection', 'SlideDecision', 'InitialCost', 'LeapDecisionToEmpty', 'AlquerqueBoardWithEightTriangles', 'GroupWin', 'Tile', 'TerritoryEnd', 'DirectionCaptureFrequency', 'NoBoard', 'NoTargetPieceWin', 'ForwardRightDirection', 'ProposeEffectFrequency', 'TurnKo', 'NoOwnPiecesLossFrequency', 'RotationalDirection', 'SowRemove', 'HopDecisionEnemyToEnemy', 'RookComponent', 'TableStyle', 'TerritoryWin', 'MancalaSixRows', 'MaxDistance', 'NoOwnPiecesLoss', 'Threat', 'PositionalSuperko', 'CaptureSequence', 'NumOffDiagonalDirections', 'ProposeEffect', 'Roll', 'SlideDecisionToFriend', 'LineDraw', 'SetValue', 'GroupDraw', 'SumDice', 'ThreeMensMorrisBoardWithTwoTriangles', 'KingComponent', 'Repetition', 'SurroundCapture', 'Loop', 'NoOwnPiecesWin', 'BranchingFactorChangeNumTimesn', 'RotationDecision', 'LoopEndFrequency', 'InterveneCapture', 'HopDecisionFriendToEmpty', 'EliminatePiecesDrawFrequency', 'DiceD2', 'Edge', 'SetCount', 'RightwardDirection', 'LoopEnd', 'ShogiStyle', 'SwapPiecesDecision', 'FortyStonesWithFourGapsBoard', 'StarShape', 'Boardless', 'MancalaCircular', 'XiangqiComponent', 'ReachLossFrequency', 'Fill', 'SlideDecisionToEnemy', 'JanggiComponent', 'KintsBoard', 'ShogiComponent', 'SowBacktracking', 'Piece', 'InitialRandomPlacement', 'LoopWin', 'LoopWinFrequency', 'Flip', 'FillEnd', 'JanggiStyle', 'ShootDecisionFrequency', 'MancalaThreeRows', 'StrategoComponent', 'RotationDecisionFrequency', 'InterveneCaptureFrequency', 'EliminatePiecesDraw', 'AutoMove', 'PachisiBoard', 'GroupLoss', 'PathExtent', 'VisitedSites', 'Cooperation', 'SetRotationFrequency', 'FillWin', 'SpiralTiling', 'PathExtentEnd', 'SpiralShape', 'Team', 'ReachDrawFrequency', 'LeftwardsDirection', 'ReachDraw', 'PathExtentLoss', 'PathExtentWin', 'LoopLoss', 'RightwardsDirection'
        ]

        if self.config.is_train:
            columns_to_drop += [
                'num_wins_agent1',
                'num_draws_agent1',
                'num_losses_agent1',
            ]

        if self.config.n_tf_ids_features == 0:
            columns_to_drop += ['EnglishRules', 'LudRules']

        df = df.drop(columns_to_drop, strict=False)
        
        print('Shape after dropping specific columns:', df.shape)
        
        if self.config.is_train:
            print('\n', df["fold"].value_counts())
        
        return df
    
    def postprocessing(self, df: pl.DataFrame) -> pd.DataFrame:
        """Adjust data types."""
        
        df = df.to_pandas()

        # Categorical mapping.
        
        if self.config.is_train:
            cat_mapping, catcols = {}, []
            for feature in df.columns:
                if feature in ["fold", "data_mode", "utility_agent1", "GameRulesetName"]: continue
                if (df[feature].dtype == object):
                    cat_mapping[feature] = "category"
                    catcols.append(feature)
                else:
                    cat_mapping[feature] = float
            
            self.data_checkpoint["catcols"] = catcols
        
        else:
            catcols = self.data_checkpoint["catcols"]
            cat_mapping = {f: "category" if f in catcols else float for f in df.columns}

        df = df.astype(cat_mapping)

        if self.config.task == "classification":
            df["utility_agent1"] = df["utility_agent1"].astype(str)
        
        return df
    
    def reduce_mem_usage(self, df, float16_as32=True):
        """Reduce memore usage."""

        start_mem = df.memory_usage().sum() / 1024**2
        print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

        for col in df.columns:
            col_type = df[col].dtype

            if col_type != object and str(col_type)!='category':
                c_min,c_max = df[col].min(),df[col].max()

                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)  
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        if float16_as32:
                            df[col] = df[col].astype(np.float32)
                        else:
                            df[col] = df[col].astype(np.float16)  
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)

        end_mem = df.memory_usage().sum() / 1024**2
        print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

        return df
    
    def save_data_checkpoint(self, df: pd.DataFrame) -> None:
        """Save data checkpoint."""
        
        if self.config.is_train:
            with open(self.config.path_to_save_data_checkpoint, "wb") as f:
                pickle.dump(self.data_checkpoint, f)

            df.to_csv(self.config.path_to_save_dataset, index=False)
        
    def get_dataset(self, df: pl.DataFrame) -> Tuple[pd.DataFrame, list]:
        """Get the dataset."""

        if self.config.is_train and self.config.load_dataset_checkpoint:
            print("Loading the dataset from the checkpoint...")
            df = pd.read_csv(self.config.path_to_save_dataset)
            with open(self.config.path_to_load_data_checkpoint, "rb") as f:
                self.data_checkpoint = pickle.load(f)

        else:
            df = self.preprocessing(df)
            df = self.feature_generation(df)
            df = self.build_validation_and_cv_features(df)
            df = self.drop_columns(df)
            df = self.postprocessing(df)
            df = self.reduce_mem_usage(df)
            self.save_data_checkpoint(df)
        
        return df, self.data_checkpoint
    

# --- Solver ---

class Solver:
    """Solution."""

    def __init__(self, config: Config, rerun: bool) -> None:
        """Initialiaztion."""
        
        self.config = config
        self.rerun = rerun
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
    
    def train_one_model(self, df, X, Y, groups, catcols, model_name) -> Tuple[np.array, np.array, Union[pl.DataFrame, None]]:
        """Train N folds of a certain model."""
        
        # Define the instances for the metrics.

        scores = []
        oof_preds, oof_labels, oof_mask = np.zeros([len(df)]), np.zeros([len(df)]), np.zeros([len(df)])
        oof_baseline_preds =  np.zeros([len(df)])

        if self.config.use_baseline_scores:
            with open(self.config.path_to_load_solver_checkpoint["baseline"] + '.pickle', "rb") as f:
                baseline = pickle.load(f)["catboost"]["oof_baseline_preds"]

        if model_name == "catboost":
            feature_importances = None

        for fold in range(self.config.n_splits):    
            
            # Get the fold's data.
            
            print(f"\n{model_name} | Fold {fold}")
            
            train_index = df[df["fold"] != fold].index
            valid_index = df[df["fold"] == fold].index

            X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
            Y_train, Y_valid = Y.iloc[train_index], Y.iloc[valid_index]

            if self.config.use_baseline_scores:
                baseline_train, baseline_valid = baseline[train_index], baseline[valid_index]

            print("X-train and X-valid shapes", X_train.shape, X_valid.shape)

            # TF-IDF.

            if self.config.n_tf_ids_features != 0:
                X_train = self.generate_TF_IDF(X_train, mode='train', fold=fold, n_tf_ids_features=self.config.n_tf_ids_features)
                X_valid = self.generate_TF_IDF(X_valid, mode='test', fold=fold, n_tf_ids_features=self.config.n_tf_ids_features)
                print('Shape with TF-IDF features', X_train.shape, X_valid.shape)

            # Separate original and mirrored data.

            mask = X_valid["data_mode"] == "original"
            X_valid_src = X_valid[mask]
            Y_valid_src = Y_valid[mask]

            if self.config.use_baseline_scores:
                baseline_valid_src = baseline_valid[mask]

            X_train = X_train.drop(["data_mode"], axis=1)
            X_valid = X_valid.drop(["data_mode"], axis=1)
            X_valid_src = X_valid_src.drop(["data_mode"], axis=1)

            X_train = X_train.drop(["index"], axis=1)
            X_valid = X_valid.drop(["index"], axis=1)
            X_valid_src = X_valid_src.drop(["index"], axis=1)

            print(X_train.columns)
            
            if self.config.use_baseline_scores:
                print("Baseline scores")
                print("\tTrain", round(mean_squared_error(Y_train, baseline_train, squared=False), 4))
                print("\tVal", round(mean_squared_error(Y_valid_src, baseline_valid_src, squared=False), 4))
                print("\tVal full", round(mean_squared_error(Y_valid, baseline_valid, squared=False), 4))

            print('Original features shape', X_train.shape, X_valid_src.shape)

            # Create model.

            if model_name == "catboost":
                if self.config.task == "classification":
                    base_model = CatBoostClassifier(**self.config.catboost_params, cat_features=catcols)
                else:
                    base_model = CatBoostRegressor(**self.config.catboost_params, cat_features=catcols)
                
            elif model_name == "lgbm":
                base_model = lgb.LGBMRegressor(**self.config.lgbm_params)

            elif model_name == "xgboost":
                base_model = xgb.XGBRegressor(**self.config.xgb_params)

            elif model_name == "DNN":
                base_model = DNN(**self.config.dnn_params)

            # Build stacked model with nested cross-validation.

            if self.config.stacked:

                inner_cv = StratifiedGroupKFold(n_splits=self.config.n_splits, shuffle=True, random_state=self.config.seed)

                inner_split = inner_cv.split(
                    X_train,
                    Y_train.astype(str) + "_" + X_train["p1_agent"].astype(str),
                    groups.iloc[train_index]
                )
                
                oof_inner = np.zeros_like(Y_train)
                oof_valid = np.zeros_like(Y_valid)
                oof_valid_src = np.zeros_like(Y_valid_src)

                for inner_fold, (idx_tr_inner, idx_va_inner) in enumerate(inner_split):
                    print(f"Inner fold {inner_fold}.")

                    X_tr_inner = X_train.iloc[idx_tr_inner]
                    X_va_inner = X_train.iloc[idx_va_inner]
                    y_tr_inner = Y_train.iloc[idx_tr_inner]
                    y_va_inner = Y_train.iloc[idx_va_inner]

                    if not self.rerun:
                        model = clone(base_model)

                        if model_name == "catboost":
                            model.set_params(random_seed=58)
                            model.fit(X_tr_inner, y_tr_inner, eval_set=(X_va_inner, y_va_inner))
                        elif model_name == "lgbm":
                            model.fit(X_tr_inner, y_tr_inner,
                                eval_set=[(X_va_inner, y_va_inner)],
                                eval_metric='rmse',
                                callbacks=[
                                    lgb.early_stopping(self.config.catboost_params["early_stopping_rounds"]),
                                    lgb.log_evaluation(self.config.catboost_params["verbose"])
                            ])
                        elif model_name == "xgboost":
                            model.fit(X_tr_inner, y_tr_inner, eval_set=[(X_va_inner, y_va_inner)], verbose=1000)
                        elif model_name == "DNN":
                            model.fit(X_tr_inner, y_tr_inner, X_va_inner, y_va_inner)
                    
                        path_to_save = os.path.join(self.config.path_to_save_solver_checkpoint, f"inner_{model_name}_{fold}_{inner_fold}.pickle")
                        with open(path_to_save, "wb") as f:
                            pickle.dump(model, f)
                    else:
                        path_to_load = os.path.join(self.config.path_to_load_solver_checkpoint["main"], f"inner_{model_name}_{fold}_{inner_fold}.pickle")
                        with open(path_to_load, "rb") as f:
                            model = pickle.load(f)

                    y_pred = model.predict(X_va_inner)
                    oof_valid += model.predict(X_valid) / self.config.n_splits
                    oof_valid_src += model.predict(X_valid_src) / self.config.n_splits
                    oof_inner[idx_va_inner] += y_pred

                    del model

                X_train['oof_feature'] = oof_inner
                X_valid['oof_feature'] = oof_valid
                X_valid_src['oof_feature'] = oof_valid_src

                oof_baseline_preds[valid_index] = oof_valid
            
            # Create and fit the main model.
            
            if not self.rerun:
                
                print("\nMain model")

                model = base_model

                if model_name == "catboost":
                    if self.config.use_baseline_scores:
                        train_pool = Pool(X_train, Y_train, baseline=baseline_train, cat_features=catcols)
                        valid_pool = Pool(X_valid_src, Y_valid_src, baseline=baseline_valid_src, cat_features=catcols)
                    else:
                        train_pool = Pool(X_train, Y_train, cat_features=catcols)
                        valid_pool = Pool(X_valid_src, Y_valid_src, cat_features=catcols)
                    model.fit(train_pool, eval_set=valid_pool)
                    
                elif model_name == "xgboost":
                    model.fit(X_train, Y_train, eval_set=[(X_valid_src, Y_valid_src)], verbose=1000)

                elif model_name == "DNN":
                    model.fit(X_train, Y_train, X_valid_src, Y_valid_src)
                
            else:
                path_to_load = os.path.join(self.config.path_to_load_solver_checkpoint["main"], f"main_{model_name}_{fold}.pickle")
                with open(path_to_load, "rb") as f:
                    model = pickle.load(f)
            
            # Prediction (with TTA).
            
            if model_name == "catboost":
                if self.config.task == "classification":
                    Y_valid_src = Y_valid_src.astype(np.float)
                    preds = model.predict(X_valid_src).astype(float)
                    preds = preds[0]
                else:
                    if self.config.use_baseline_scores:
                        preds = model.predict(Pool(X_valid_src, baseline=baseline_valid_src, cat_features=catcols))
                        full_preds = model.predict(Pool(X_valid, baseline=baseline_valid, cat_features=catcols))
                    else:
                        preds = model.predict(Pool(X_valid_src, cat_features=catcols))
                        full_preds = model.predict(Pool(X_valid, cat_features=catcols))
            elif model_name == "lgbm":
                preds = model.predict(X_valid_src)
                full_preds = model.predict(X_valid)
            elif model_name == "xgboost":
                preds = model.predict(X_valid_src)
                full_preds = model.predict(X_valid)
            elif model_name == "DNN":
                preds = model.predict(X_valid_src)
                full_preds = model.predict(X_valid)
            
            # Save the scores and the metrics.
            
            oof_preds[valid_index] = full_preds
            oof_labels[valid_index] = Y_valid
            oof_mask[valid_index] = mask

            score_original = mean_squared_error(Y_valid_src, preds, squared=False)
            score = mean_squared_error(Y_valid, full_preds, squared=False)

            scores.append(score_original)

            print(round(score_original, 4))
            print(round(score, 4))  

            # SHAP.

            if self.config.show_shap:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer(X_valid_src[:1000])
                shap.plots.beeswarm(shap_values, max_display=20)
            
            if not self.rerun:
                path_to_save = os.path.join(self.config.path_to_save_solver_checkpoint, f"main_{model_name}_{fold}.pickle")
                with open(path_to_save, "wb") as f:
                    pickle.dump(model, f)
                
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
        
        self.models[model_name]["oof_baseline_preds"] = oof_baseline_preds
        self.models[model_name]["oof_preds"] = oof_preds
        self.models[model_name]["oof_labels"] = oof_labels
        self.models[model_name]["mode"] = oof_mask
        self.models[model_name]["fold"] = df["fold"].values
        
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

        # Create checkpoint folder.

        if not os.path.exists(self.config.path_to_save_solver_checkpoint):
            os.mkdir(self.config.path_to_save_solver_checkpoint)
        
        # Select the feature and the targets.
        
        X = df.drop(['utility_agent1', 'fold', 'GameRulesetName'], axis=1).rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x)).reset_index()
        Y = df['utility_agent1']
        groups = df['GameRulesetName']

        catcols = data_checkpoint["catcols"]

        if self.config.use_oof:
            with open(self.config.path_to_load_solver_checkpoint["baseline"] + '.pickle', "rb") as f:
                X["oof"] = pickle.load(f)["catboost"]["oof_baseline_preds"]
            
        # Train models.
        
        artifacts = {}
        
        for model_name, to_train in self.config.to_train.items():
            if to_train:
                self.models[model_name] = {}
                artifacts[model_name] = {}

                oof_labels, oof_preds, feature_importance = self.train_one_model(df, X, Y, groups, catcols, model_name)

                artifacts[model_name]["oof_preds"] = oof_preds
                artifacts[model_name]["oof_labels"] = oof_labels
                artifacts[model_name]["feature_importance"] = feature_importance
        
        # Save solution checkpoint for the inference.
        
        if self.config.is_train and not self.rerun:
            with open(self.config.path_to_save_solver_checkpoint + '.pickle', "wb") as f:
                pickle.dump(self.models, f)
        elif self.rerun:
            pass
                
        return artifacts
            
    def predict(self, X: pd.DataFrame, df_train: pd.DataFrame, data_checkpoint: dict) -> np.array:
        """Inference."""

        # TF-IDF.

        if self.config.n_tf_ids_features != 0:
            X = self.generate_TF_IDF(X, mode='test', fold=0, n_tf_ids_features=self.config.n_tf_ids_features)
            print('Shape with TF-IDF features', X.shape)

        # Inference | Main.

        prediction = np.zeros(len(X))
        
        for model_name, weight in self.config.weights.items():
            
            if model_name not in self.models or weight == 0: continue
            
            preds = np.zeros(len(X))

            for fold in range(self.config.n_splits):    

                model = self.models[model_name]["models"][fold]
                    
                preds += model.predict(X) / self.config.n_splits

            prediction += np.clip(preds, -1, 1) * weight
            
        return prediction
   

# --- Train and inference ---

def train(rerun: bool, oof_features=None) -> dict:
    """Training function."""
    
    config = Config()
    
    set_seed(config.seed)
    dataset = Dataset(config, rerun)
    solver = Solver(config, rerun)
    
    df = pl.read_csv(config.path_to_train_dataset)
    df, data_checkpoint = dataset.get_dataset(df)
    artifacts = solver.fit(df, data_checkpoint, oof_features)
    
    return artifacts


if not IS_TRAIN:
    
    config = Config()   
    
    set_seed(config.seed)
    dataset = Dataset(config, rerun=False)
    solver = Solver(config, rerun=False)

    df_train = pl.read_csv(config.path_to_train_dataset)
    df_train, _ = dataset.get_dataset(df_train)

    def predict(test: pl.DataFrame, sample_sub: pl.DataFrame) -> pl.DataFrame:
        """Inference function."""
        
        df, data_checkpoint = dataset.get_dataset(test)
        preds = solver.predict(df, df_train, data_checkpoint)

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
        df = pd.read_parquet('submission.parquet')
        print(df.head())