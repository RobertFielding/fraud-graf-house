import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


data_filename = 'creditcard.csv'

df = pd.read_csv(data_filename)
df["Label"] = "Genuine"
df.loc[df["Class"] == 1, "Label"] = "Fraud"

for x in df.columns:
    sns.histplot(data=df,
                 common_norm=False,
                 x=x,
                 hue="Label",
                 stat='density',
                 palette={"Genuine": 'green', "Fraud": 'red'})
    plt.title(x)
    plt.savefig(f"Figures/{x}.png")
    plt.clf()
