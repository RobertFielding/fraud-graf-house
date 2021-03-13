import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
import numpy as np

data_filename = 'creditcard.csv'

df = pd.read_csv(data_filename)

train, test = train_test_split(df, test_size=0.2)
train_features, train_labels = train.iloc[:, :-1], train.iloc[:, -1]
test_features, test_labels = test.iloc[:, :-1], test.iloc[:, -1]

# Configure Random Forest Classifier, then train and test
rf_classifer = RandomForestRegressor(max_depth=4, random_state=0)
rf_classifer = rf_classifer.fit(train_features, train_labels)
rf_predictions = rf_classifer.predict(test_features)
# Make rf_df dataframe
test["Prediction Percentile"] = rf_predictions

## Performance metrics
# ROC and auc
rf_fpr, rf_tpr, _ = roc_curve(test_labels, test["Prediction Percentile"])
rf_auc = roc_auc_score(test_labels, test["Prediction Percentile"], multi_class='ovr')

np.savetxt("ROC_rf_fpr", rf_fpr)
np.savetxt("ROC_rf_tpr", rf_tpr)
with open("ROC_rf_auc.txt", mode='w') as f:
    f.write(str(rf_auc))

# Plot ROC Curve
plt.plot(rf_fpr, rf_tpr, label="Random Forest ROC Curve")
# plt.show()

# Distribution of prediction scores
test["Label"] = "Genuine"
test.loc[test["Class"] == 1, "Label"] = "Fraud"

for data_type in ["Genuine", "Fraud"]:
    sns.histplot(data=test[test["Label"] == data_type],
                 common_norm=False,
                 x="Prediction Percentile",
                 stat="density",
                 )
    plt.title(data_type + "_Predictions")
    # plt.show()
    plt.savefig(f"Figures/Random_Forest_{data_type}.png")
    plt.clf()

test.loc[test["Label"] == "Fraud", "Amount"].sum()

## Creation of scorecard
score_df = test[["Label", "Amount", "Prediction Percentile"]]

score_df["Genuine Volume"] = (score_df["Label"] == "Genuine").astype(int)
score_df["Fraud Volume"] = (score_df["Label"] == "Fraud").astype(int)
score_df["Genuine Amount"] = np.where(score_df["Label"] == "Genuine", score_df["Amount"], 0)
score_df["Fraud Amount"] = np.where(score_df["Label"] == "Fraud", score_df["Amount"], 0)

scorecard = (score_df[["Genuine Volume", "Fraud Volume", "Genuine Amount", "Fraud Amount"]]
    .groupby((100 * score_df["Prediction Percentile"]).astype(int))
    .sum())

scorecard = scorecard.reindex(range(0, 100, 1), fill_value=0)
scorecard = scorecard[::-1].cumsum()[::-1]

pd.DataFrame.to_csv(scorecard, "Scorecard.csv")
