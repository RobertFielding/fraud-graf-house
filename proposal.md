# Fraud Model Performance Visualisation in Grafana

## Part 1 Explore

* Download fraud data e.g. https://www.kaggle.com/mlg-ulb/creditcardfraud
* Explore the data with e.g. https://seaborn.pydata.org/generated/seaborn.pairplot.html

## Part 2 Train

* Train a simple ML model to predict the `Class`. E.g. decision tree or random forest. The model does not need to be good.
* Maybe visualise the trees with: https://scikit-learn.org/stable/modules/generated/sklearn.tree.plot_tree.html
* Create a dataset of predictions including at least: `Time`, `Class`, `Prediction`, `Amount`. 

## Part 3 Offline Performance Metrics

Given the set of predictions/scores from Part 2 create plots that summarise the model performance, include:
* [ROC Curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html) and [auc](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html#sklearn.metrics.auc)
* Distribution of Scores
* Scorecard table of the form below, where each row records statistics for the rule `decline if Score >= Score Threshold`
 
 | Score Threshold | Genuine | Fraud | Fraud Volume Detection Rate | Genuine Volume Decline Rate | Fraud Value Detection Rate | Genuine Value Decline Rate |
 | --- | --- | --- | --- | --- | --- | --- |
 | 0.00 | 
 | 0.01 |
 | ... |
 | 0.99 | 
  
## Part 3 Grafana Performance Metrics

* Install [ClickHouse](https://clickhouse.tech/), [grafana](https://grafana.com/) and the [grafana ClickHouse datasource plugin](https://grafana.com/grafana/plugins/vertamedia-clickhouse-datasource).
* Import the dataset of predictions to ClickHouse.
* Try to create a grafana dashboard with an ROC curve and scorecard table from a ClickHouse query. 
