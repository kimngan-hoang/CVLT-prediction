import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def visualize(importance_df, feature_names, color, save_path):
    importance_df = importance_df.set_index(feature_names)
    importance_df_cols = importance_df.columns.values
    importance_df["Mean Importance"] = importance_df[importance_df_cols].mean(axis=1)
    importance_df["SD Importance"] = importance_df[importance_df_cols].std(axis=1)
    
    #Sort by mean importance
    importance_df = importance_df.sort_values(by=["Mean Importance"], ascending = False)
    importance_df[0:10]
    
    s = (pd.Series(importance_df["Mean Importance"], index = feature_names)).nlargest(10)
    err = importance_df["SD Importance"][0:10]

    plt.figure()
    s.plot(kind='barh', xerr=err, capsize=3, colormap = color)
    plt.title('Feature Importance based on outer training sets')
    plt.show()
    plt.savefig(save_path)