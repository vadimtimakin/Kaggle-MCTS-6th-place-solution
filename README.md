## Intro

The solution for the [UM - Game-Playing Strength of MCTS Variants](https://www.kaggle.com/competitions/um-game-playing-strength-of-mcts-variants/leaderboard) competition on Kaggle (6 / 1608 place).

## 1. Summary

My solution combines the power of zero-cost data generation, good modeling, both manual and automatic feature generation, ensembling with two-staged models and also has some risky parts that I'll try to justify further.

The general pipeline is presented below.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F5007869%2Fbbdc46b5cdfdf9ae6c1c7a84f640181b%2FMCTS_pipeline.jpg?generation=1733185200230382&alt=media)

## 2. Validation strategy

Starting with simple GroupKFold with GameRulesetName as a group as anyone else in this competition was a good baseline, however wasn't stable enough. First modification that I did was adding a stratification by target, after this I also added agent1 column to the stratify criteria.

Even with this CV approach I didn't get a complete correlation between CV and public LB or filled the gap between CV and LB (which in my opinion shouldn't be a concern in competitions which use a group type of stratification between train and test set), however it still was more stable and showed more expected results locally with different approaches throught this competitions, it also showed consistent results with different seeds on both CV and public LB.

It's also important to keep the split always the same and validate only on source data which become the most actual with the new data generation.

Using more than 5 folds was leading to a minor score boost yet was less stable.

## 3. Data part

### 3.1. Preprocessing
1. Drop constant columns.
2. Drop GameRulesetName, LudRules and EnglishRules.
3. Memory optimization.

I also experimened with one-hot / label / target encoding for categorical features and scaling / binarizing / normalizing numerical ones to improve model perfomance and reduce overfitting, but it was ineffective in all the cases except for training a DNN model.
### 3.2. Zero-cost data generation

This part is based on both logic and some empirical assumptions. 

Initially, the idea to generate more data comes from the fact that we have some game depending and player depending features which also depend on the order. First, I tried to swapping agent strings (here and further I always invert the target as well) and AdvantageP1 column thinking that it represents the probability of winning of a certain agent over another one, which already gave a solid boost for my CV and LB (it's important to validate only on original data). Soon, I realised that this column actually represents the probability of the first player winning the second one in this game where both players play randomly. So this feature logically shouldn't be inverted and I removed invertion for it, but it actually decreased the score! My interpretation of this is that while it's logically correct to only swap agents and target, such transformation doesn't bring any new signal to the data and therefore leads to the worse results.

After this I still had a gap between my original and inverted OOF predictions scores, so I was looking for ways to fill it by trying to invert other features. One of this features was the Balance, which had the highest feature importance after the AdvantageP1.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F5007869%2Fbbbe9e715a7fcec7bd8e3e23f7029c14%2Ffeature_importances.jpg?generation=1733185844833161&alt=media)

So I tried to invert this feature as well and that's what filled the gap almost completely. Besides this, I also tried to invert even more features, but the rest of them had lower feature importance and didn't bring any new signal to the data. Below is the demonstration of the gap for the case without inverting the Balance column and with inverting it as well.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F5007869%2Fe3dd68ac7f9a31cf0cb1de4700914212%2Fbalance.jpg?generation=1733185259940559&alt=media)

The overall data generation pipeline is illustrated below.
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F5007869%2F39645e500c397e14f6e047fa979b61de%2FData_generation.jpg?generation=1733185283692625&alt=media)

Besides this I tried other methods of new data generation, like agents playing with themselves, but it was less valuable. 

When I filled the error gap between original and generated data, it also gave me an opportunity to use TTA which could boost my score on 0.001-0.002 points. However, it's not used in my final submission.

Insiped by the results described above, I also tried doing pseudo-labeling by training another model without group split criteria and then generating the new data with it, but it didn't improve the score.

### 3.3. Manual feature generation
I tried to generate a lot of features manually, most of them didn't work, but here a bunch of successful ones.

1) The first set of features comes from simply splitting the agent strings.
```python
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
```
2) The second set of features is taken from [this notebook](https://www.kaggle.com/code/litsea/mcts-baseline-fe-lgbm) by @litsea.
```python
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
```
3) The third set of features I call cross-agent features, it captures the interactions between different subtypes of agents.
```python
# Cross-agent features.

df["p1_agent"] = df["agent1"]
df["p2_agent"] = df["agent2"]

cols = ["agent", "selection", "exploration", "playout", "bounds"]
for c1 in cols:
    for c2 in cols:
        df[f"{c1}_{c2}"] = df[f"p1_{c1}"].astype(str) + df[f"p1_{c2}"].astype(str)
```

Talking about the features generated from the columns with the rules. For me, they all turned out to be ineffective. I tried using TF-IDF, counting various indexes, but it did not give an increase in score.

### 3.4. Automatic feature generation

I used [OpenFE](https://github.com/IIIS-Li-Group/OpenFE) library for automatic feature generation. It's a great algorithm, however it didn't work out-of-box as I expected, so I had to make a list of necessary changes to make it work in this competition. Basically, I went through the next steps:
1. Fix bugs related to multiprocessing.
2. Fix bugs related to data types.
3. Adjust logging for debug purposes.
4. Replace StratifiedKFold with StratifiedGroupKFold.
5. Sample 10% of by original data (considering both stratify and group criteria) for running the algorithm.
6. Run the algorithm (around 30 hours on 10% subsample and 1 CPU).
7. Select top-500 the best features.
8. Remove features aggregated by groups and freq features as they produce unstable results in public LB.
9. Fill NaNs with "-100" value.
10. Save features and metadata for generating them on inference.

The graph below shows that many of these features turned out to be quite informative.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F5007869%2F3df9b735efc3d7c598e40280a19ee4ee%2Fshap_values.jpg?generation=1733185300980400&alt=media)

### 3.5. Feature selection

After all the feature generations I left only about 400 columns based on the CV importance. It didn't have any implact neither on CV nor on public LB and was done just for saving memory and faster training purposes.

Besides that I tried to drop features not only based on feature importance but also based on other factors: correlation, varience, number of unique values per group, etc. But it didn't work.

## 4. Modeling

### 4.1. Approach

From the very beginning to the end of the competition, Catboost showed the best results for me, while other models were worse, so I decided to use a two-stage approach to build the ensemble.

I considered this problem as a regression task and trained all my models to optimize RMSE metric. I tried to solve this task as a classification one and optimizing QWK, but got worse results. Anyways, since we have discrete labels in training data, it's possible to found a way of rounding predictions that would led to a perfomance boost. I did some experiments with post-processing, where I was looking for some thresholds for rounding my predictions, however the results were unstable so I decided not to use it in my final solution.

I also tried different reguralizations to achieve more balanced feature importances but it didnt' lead to any improvements as well. 

### 4.2. Catboost 
The best single model. The parameters are adjusted manually.

```python
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
```

### 4.3. Two-staged models

LGBM, XGB and DNN models were consistently worse in this competition, so I decided to use catboost OOF predictions as another feature for these models.

1) First, I needed to collect those OOF predictions propeply. Using simple outputs of 5 folds model split didn't work and caused leakage so I used a **Nested CV Catboost model (5*5 folds)** for this. The reason to do so is described well in [this discussion](https://www.kaggle.com/competitions/um-game-playing-strength-of-mcts-variants/discussion/543756) and [this notebook](https://www.kaggle.com/code/martinapreusse/mcts-stacked-catboost) by @martinapreusse.
2) Then I trained **Catboost OOF and LGBM OOF** models. Paramters for catboost are the same as above, parameters for LGBM model presented below, I didn't tune them a lot and there is a huge room for improvement here.
```python
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
```
3) Next I trained another **Catboost** model using OOF predictions, but this time not as a feature but as a **baseline initialization**. An example can be found [here](https://catboost.ai/docs/en/concepts/python-usages-examples#baseline).
4) Finally, I trained a **DNN model**. I tried a lot of different architecture variations, but ended up with a classic MLP with modifications:
```css
    1. Embedding layer (128) for categorical features.
    2. Quantile transformer for numerical features.
    3. OOF feature as it is.
    4. Concat -> MLP (Dropout=0.9, Hardswish, input -> 2048 -> 1024 -> 512 -> 256 -> 128 -> 1).
    5. Use MADGRAD optimizer and RMSE loss for optimization.
```

### 4.4. Other models
Besides that I tried to train another meta-models that would predict number of games played, corner cases and draws, however it didn't lead to any significant improvements. XGBoost models weren't effective as well.

### 4.5. Ensembling
After testing various ensembling approaches I just ended up choosing weighted ensemble with positive weights selected by the scipy minimize function based on CV score.

## 5. Matching the distribution
Since the beginning of the competition, I noticed that at least the training and public test dataset have a different distribution. This was indirectly confirmed by big gap between LB and CV scores and LB probing done by other participants, for example in [this discussion](https://www.kaggle.com/competitions/um-game-playing-strength-of-mcts-variants/discussion/533210) by @tomooinubushi. I also saw a lot of data shifting and scaling in public notebooks. I decided to focus on my CV in this competition and come back to this closer to the end of competition.

At the end I had a stable ensemble with a good CV score, so I decided to use it as my stable submission without any post-processing, while reserving another one for a more risky approach. I took one of my notebooks which had the best public score and did some probing as well. I tried different clipping, shifting and scaling techinique in order to match the public distribution the best.

I was able to improve my best public score from 0.421 to 0.417 using the following adjustment:
```python
predictions = np.clip(predictions * 1.175, -1, 1)
```

What's interesting is that applying the same adjustment for my best CV ensemble improved the public score only from 0.424 to 0.423.

Basically, the idea behind this is that the private test dataset either has completely new games or has games that intersect with games from public test dataset (or similar to them). I tried to cover both of the cases.

## 6. Final results

**The submission aligned the best with the public LB scored the best on the private LB as well.**


| Model                                           | CV         | Public LB | Private LB |
| ----------------------------------------------- | ---------- | --------- | ---------- |
| Catboost Classic                                | 0.3933     | 0.426     | 0.432      |
| Catboost Nested                                 | 0.3946     | 0.426     | 0.433      |
| Catboost with OOF                               | 0.3978     | 0.425     | 0.431      |
| Catboost with OOF as baseline                   | 0.3969     | 0.426     | 0.433      |
| LGBM with OOF                                   | 0.3996     | 0.426     | 0.432      |
| DNN with OOF                                    | 0.4017     | 0.426     | 0.432      |
| *Best CV ensemble*                              | *0.3905*   | *0.424*   | *0.430*    |
| **Best public (Catboost + Distribution Match)** | **0.3995** | **0.417** | **0.422**  |



## 7. What can also be beneficial and what is not
- **Meaningful LB probing** - could be useful here, but I started doing it only closer to the end of the competition, yet I see many people with hundreds of submissions and looking forward to find out their potential approaches.
- **LLM for rules parcing** - it seems like there just aren't enough unique rules for training a language model, so I didn't do it, yet again looking forward to find out if anyone did it successfully.
- **Pseudo-labeling on test data** - could be possible and valuable here if the test dataset wasn't given in batches.
- **Handle randomness better** - there are obviosly determenistic games and probability based ones (for example with dice(s)), handling the second type of games better and matching the possible distribution the smarter way could be beneficial here.
- **Classic data generation** - takes a lot of compute, but must be valuble here.

## 8. References
[OpenFE](https://github.com/IIIS-Li-Group/OpenFE/tree/master) - An efficient automated feature generation tool.
