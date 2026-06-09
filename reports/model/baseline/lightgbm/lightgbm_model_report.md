# LightGBM Baseline Report

## 1. Model Setting

- Model: LightGBMClassifier
- Preprocessing for numeric variables: median imputation
- Preprocessing for categorical variables: most frequent imputation and One-Hot encoding
- n_estimators: 1000
- learning_rate: 0.03
- num_leaves: 31
- max_depth: -1
- min_child_samples: 30
- subsample: 0.9
- colsample_bytree: 0.9
- reg_lambda: 1.0
- Train size: 475355
- Validation size: 118839
- Positive class: Yes

## 2. Validation Metrics

| model    | positive_class   |   train_size |   valid_size |   accuracy |   balanced_accuracy |   precision |   recall |       f1 |   roc_auc |   average_precision |
|:---------|:-----------------|-------------:|-------------:|-----------:|--------------------:|------------:|---------:|---------:|----------:|--------------------:|
| LightGBM | Yes              |       475355 |       118839 |   0.860458 |            0.783457 |    0.709845 | 0.643351 | 0.674964 |  0.916411 |            0.754206 |

## 3. Baseline Model Comparison

| model              | strategy      | positive_class   |   train_size |   valid_size |   accuracy |   balanced_accuracy |   precision |   recall |       f1 |   roc_auc |   average_precision |
|:-------------------|:--------------|:-----------------|-------------:|-------------:|-----------:|--------------------:|------------:|---------:|---------:|----------:|--------------------:|
| DummyClassifier    | most_frequent | Yes              |       475355 |       118839 |   0.774796 |            0.5      |    0        | 0        | 0        |  0.5      |            0.225204 |
| LogisticRegression | nan           | Yes              |       475355 |       118839 |   0.854416 |            0.784753 |    0.683671 | 0.657998 | 0.670589 |  0.908428 |            0.727093 |
| RandomForest       | nan           | Yes              |       475355 |       118839 |   0.840423 |            0.752862 |    0.662676 | 0.593543 | 0.626207 |  0.892798 |            0.69061  |
| LightGBM           | nan           | Yes              |       475355 |       118839 |   0.860458 |            0.783457 |    0.709845 | 0.643351 | 0.674964 |  0.916411 |            0.754206 |

## 4. Top Original Feature Importances

| original_feature   |   importance |
|:-------------------|-------------:|
| TotalCharges       |         7987 |
| MonthlyCharges     |         6411 |
| tenure             |         5784 |
| PaymentMethod      |         1664 |
| Contract           |         1053 |
| MultipleLines      |          741 |
| PaperlessBilling   |          655 |
| SeniorCitizen      |          635 |
| OnlineSecurity     |          529 |
| Dependents         |          526 |
| StreamingTV        |          505 |
| gender             |          502 |
| DeviceProtection   |          498 |
| StreamingMovies    |          486 |
| Partner            |          483 |
| OnlineBackup       |          482 |
| TechSupport        |          421 |
| InternetService    |          395 |
| PhoneService       |          243 |

## 5. Interpretation

LightGBM是本项目Baseline阶段的梯度提升树模型。
与RandomForest相比，LightGBM采用Boosting思想逐步修正前一轮模型的误差，通常更适合结构化表格数据。

本节仍采用统一的训练集和验证集划分，并使用与前面模型一致的预处理流程，
从而保证DummyClassifier、LogisticRegression、RandomForest和LightGBM之间具有可比性。

评价时不应只关注Accuracy，还应重点比较流失类Recall、F1、Balanced Accuracy、ROC-AUC和Average Precision。
如果LightGBM在这些指标上超过LogisticRegression，说明非线性Boosting模型能够进一步提取客户流失模式。

特征重要性结果反映变量对模型分裂的贡献。
由于类别变量经过One-Hot编码后会被拆分为多个虚拟变量，因此报告中优先解释按原始变量聚合后的重要性。
需要注意的是，特征重要性只能说明预测贡献，不能直接解释为因果关系。