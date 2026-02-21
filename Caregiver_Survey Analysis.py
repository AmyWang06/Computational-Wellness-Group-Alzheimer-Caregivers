import pyreadstat
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import pearsonr

df1, meta = pyreadstat.read_sav('ACT V1 V2 Questionnaires_Cytokines_ACT EMA Participants (1).sav')

visits=["V1", "V2"]

#Cutoff
for VX in visits:
    cutoffs = {
        f'PSQI_Total_{VX}': {'Poor Sleep Quality': (5, float('inf'))},
        f'CSI_Total_{VX}': {'Relationship Distress': (-float('inf'), 104.5)},
        f'IPAQ_TOTAL_{VX}': {'Low': (-float('inf'), 600), 'Moderate': (600, 3000), 'High': (3000, float('inf'))},
        f'CTQ_EA_{VX}': {'None/Minimal': (5, 8), 'Low': (9, 12), 'Moderate': (13, 15), 'Severe': (16, float('inf'))},
        f'CTQ_PA_{VX}': {'None/Minimal': (5, 7), 'Low': (8, 9), 'Moderate': (10, 12), 'Severe': (13, float('inf'))},
        f'CTQ_SA_{VX}': {'None/Minimal': (5, 5), 'Low': (6, 7), 'Moderate': (8, 12), 'Severe': (13, float('inf'))},
        f'CTQ_EN_{VX}': {'None/Minimal': (5, 9), 'Low': (10, 14), 'Moderate': (15, 17), 'Severe': (18, float('inf'))},
        f'CTQ_PN_{VX}': {'None/Minimal': (5, 7), 'Low': (8, 9), 'Moderate': (10, 12), 'Severe': (13, float('inf'))},
        f'UCLALS_Total_{VX}': {'Severe Loneliness': (31, float('inf'))},
        f'ZBI_Total12_{VX}': {'Low Burden': (0, 12), 'Moderate Burden': (13, 20), 'High Burden': (21, 48)},
        f'CESD_Total_{VX}': {'Depression Risk': (16, float('inf'))},
        f'PSWQ_Total_WI_{VX}': {'Excessive Worry': (45, float('inf'))}
    }

    results = []
    for key, cutoff_values in cutoffs.items():
        if key in df1.columns:
            for label, (lower, upper) in cutoff_values.items():
                count = ((df1[key] >= lower) & (df1[key] <= upper)).sum()
                results.append({'Questionnaire': key, 'Cutoff': label, 'Count': count})
    print(f"Cutoff for {VX}:")
    for idx in range(0,len(results)):
        print("Questionnaire: ", results[idx]['Questionnaire'],
              "Cutoff: ", results[idx]['Cutoff'],
              "Count: ",  results[idx]['Count'])


#Mean and Std of Total score
for VX in visits:
    total_columns = [col for col in df1.columns if ("TOTAL" in col.upper()) and (VX in col)]
    df_total = df1[total_columns]
    mean_scores = df_total.mean()
    std_scores = df_total.std()

    print(f"Mean scores for {VX}: ", mean_scores)
    print(f"Standard deviations for {VX}: ", std_scores)

# Survey Histogram
for VX in visits:
    total_columns = [col for col in df1.columns if ("TOTAL" in col.upper())
                     and (VX in col)]
    df_total = df1[total_columns]
    mean_scores = df_total.mean()
    mean_df = pd.DataFrame({'Questionnaire': mean_scores.index, 'Mean Score': mean_scores.values})

    plt.figure(figsize=(10, 7))
    ax = sns.barplot(x='Questionnaire', y='Mean Score', data=mean_df)
    plt.title(f'Histogram of survey for {VX}')
    plt.xticks(rotation=45, fontsize=6)
    plt.yticks(fontsize=6)
    plt.xlabel('Questionnaire', fontsize=8)
    plt.ylabel('Mean Score', fontsize=10)
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.2f'),
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 9),
                    textcoords='offset points', fontsize=6)
    plt.show()


#Correlation Analysis for Total Scores
for VX in visits:
    total_columns = [col for col in df1.columns if ("TOTAL" in col.upper()) and (VX in col)]
    df_total = df1[total_columns]
    df_total = df_total.apply(lambda x: x.fillna(x.mean()), axis=0)
    correlation_matrix = df_total.corr()

    p_value_matrix = pd.DataFrame(np.ones((len(total_columns), len(total_columns))), columns=total_columns,
                                  index=total_columns)

    for i in range(len(total_columns)):
        for j in range(i + 1, len(total_columns)):
            col1, col2 = total_columns[i], total_columns[j]
            if i != j:
                r, p = pearsonr(df_total[col1], df_total[col2])
                p_value_matrix.loc[col1, col2] = p
                p_value_matrix.loc[col2, col1] = p

    significance_matrix = p_value_matrix.applymap(
        lambda x: '***' if x < 0.001 else '**' if x < 0.01 else '*' if x < 0.05 else '')

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f', annot_kws={"size": 7})

    for i in range(len(total_columns)):
        for j in range(len(total_columns)):
            if significance_matrix.iloc[i, j] != '':
                plt.text(j + 0.5, i + 0.3, significance_matrix.iloc[i, j], ha='center', va='center', color='black',
                         fontsize=8)

    plt.title(f'Correlation Matrix for {VX}')
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.show()

    print(f"correlation matrix for {VX}")
    print(correlation_matrix)
    print("p-value matrix")
    print(p_value_matrix)
    print("significant correlations (marked with p < 0.05)")
    print(significance_matrix)


#Correlation Analysis for Visits
v1_columns = [col for col in df1.columns if "TOTAL" in col.upper() and "V1" in col]
v2_columns = [col for col in df1.columns if "TOTAL" in col.upper() and "V2" in col]

v1_columns.sort()
v2_columns.sort()

correlation_matrix = pd.DataFrame(np.zeros((len(v1_columns), len(v2_columns))),
                                  index=v1_columns, columns=v2_columns)

p_value_matrix = pd.DataFrame(np.ones((len(v1_columns), len(v2_columns))),
                              index=v1_columns, columns=v2_columns)

for col1 in v1_columns:
    for col2 in v2_columns:
        if col1.replace("V1", "") == col2.replace("V2", ""):
            r, p = pearsonr(df1[col1].fillna(df1[col1].mean()), df1[col2].fillna(df1[col2].mean()))
            correlation_matrix.loc[col1, col2] = r
            p_value_matrix.loc[col1, col2] = p

significance_matrix = p_value_matrix.applymap(
    lambda x: '***' if x < 0.001 else '**' if x < 0.01 else '*' if x < 0.05 else '')

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f", annot_kws={"size": 8})

for i in range(len(v1_columns)):
    for j in range(len(v2_columns)):
        if significance_matrix.iloc[i, j] != '':
            plt.text(j + 0.5, i + 0.3, significance_matrix.iloc[i, j], ha='center', va='center', color='black', fontsize=8)

plt.title("Correlation Matrix: Visit 1 vs Visit 2")
plt.xticks(ticks=np.arange(len(v2_columns)) + 0.5, labels=v2_columns, fontsize=7, rotation=90)
plt.yticks(ticks=np.arange(len(v1_columns)) + 0.5, labels=v1_columns, fontsize=7)
plt.show()

print("Correlation matrix between Visit 1 and Visit 2:")
print(correlation_matrix)
print("\nP-value matrix:")
print(p_value_matrix)
print("\nSignificance markers (p < 0.05):")
print(significance_matrix)


#T-test
total_columns_V1 = [col for col in df1.columns if ("TOTAL" in col.upper())
                     and ("V1" in col)]
total_columns_V2 = [col for col in df1.columns if ("TOTAL" in col.upper())
                    and ("V2" in col)]
df_total_V1= df1[total_columns_V1]
df_total_V2= df1[total_columns_V2]
df_total_V1 = df_total_V1.fillna(df_total_V1.mean())
df_total_V2 = df_total_V2.fillna(df_total_V2.mean())

matching_columns = {col_V1: col_V1.replace("V1", "V2") for col_V1 in total_columns_V1 if col_V1.replace("V1", "V2") in total_columns_V2}

for col_V1, col_V2 in matching_columns.items():
    data_V1 = df_total_V1[col_V1]
    data_V2 = df_total_V2[col_V2]

    t_stat, p_value = stats.ttest_rel(data_V1, data_V2)

    print(f"T-statistic for {col_V1} and {col_V2}: {t_stat}")
    print(f"P-value for {col_V1} and {col_V2}: {p_value}")

    if p_value < 0.05:
        print(f"There is a significant change in the total score from baseline to visit 2 for {col_V1}.")
    else:
        print(f"There is no significant change in the total score from baseline to visit 2 for {col_V1}.")


#AGS Individual Score
for VX in visits:
    AGS_total_column = [col for col in df1.columns if f"AGS_Total_{VX}" in col]
    df_total = df1[AGS_total_column]

    print(f"AGS Individual Total Scores for {VX}: ", df_total.to_string(index=False))

#COPE Score
df2 = pd.read_csv("ACT - Visit 2 EMA Participant Questionnaires_April 17, 2025_12.26.csv",
                 na_values=["", " "])

score_map = {
    "I haven't been doing this at all.": 1,
    "I've been doing this a little bit.": 2,
    "I've been doing this a medium amount.": 3,
    "I've been doing this a lot.": 4
}

cope_columns = [f"COPE{i}_V2" for i in range(1, 29)]

df2[cope_columns] = df2[cope_columns].replace(score_map)
df2[cope_columns] = df2[cope_columns].apply(pd.to_numeric, errors="coerce")

problem_focused_items = [2, 7, 10, 12, 14, 17, 23, 25]
emotion_focused_items = [5, 9, 13, 15, 18, 20, 21, 22, 24, 26, 27, 28]
avoidant_coping_items = [1, 3, 4, 6, 8, 11, 16, 19]

df2["Problem_Focused_Coping"] = df2[[f"COPE{i}_V2" for i in problem_focused_items]].mean(axis=1)
df2["Emotion_Focused_Coping"] = df2[[f"COPE{i}_V2" for i in emotion_focused_items]].mean(axis=1)
df2["Avoidant_Coping"] = df2[[f"COPE{i}_V2" for i in avoidant_coping_items]].mean(axis=1)

facet_items = {
    "Active_Coping": [2, 7],
    "Informational_Support": [10, 23],
    "Positive_Reframing": [12, 17],
    "Planning": [14, 25],
    "Emotional_Support": [5, 15],
    "Venting": [9, 21],
    "Humor": [18, 28],
    "Acceptance": [20, 24],
    "Religion": [22, 27],
    "Self_Blame": [13, 26],
    "Self_Distraction": [1, 19],
    "Denial": [3, 8],
    "Substance_Use": [4, 11],
    "Behavioral_Disengagement": [6, 16]
}

for facet, items in facet_items.items():
    df2[facet] = df2[[f"COPE{i}_V2" for i in items]].sum(axis=1)

df2.to_csv("ACT - Visit 2 EMA Participant Questionnaires_with COPE Scores.csv", index=False)

print("Done.")

#Correlation Analysis for COPE Scores
score_df = pd.read_csv("ACT - Visit 2 EMA Participant Questionnaires_with COPE Scores.csv", skiprows=[1, 2])

columns_of_interest = [col for col in score_df.columns if "COPE" in col or col in [
    "DCA8_V2", "Problem_Focused_Coping", "Emotion_Focused_Coping", "Avoidant_Coping",
    "Active_Coping", "Informational_Support", "Positive_Reframing", "Planning",
    "Emotional_Support", "Venting", "Humor", "Acceptance", "Religion", "Self_Blame",
    "Self_Distraction", "Denial", "Substance_Use", "Behavioral_Disengagement"]]

df_subset = score_df[columns_of_interest].copy()
df_filled = df_subset.apply(lambda x: x.fillna(x.mean()), axis=0)

correlation_matrix = df_filled.corr()

p_value_matrix = pd.DataFrame(np.ones(correlation_matrix.shape), columns=correlation_matrix.columns, index=correlation_matrix.index)
for i in correlation_matrix.columns:
    for j in correlation_matrix.columns:
        corr, pval = pearsonr(df_filled[i], df_filled[j])
        p_value_matrix.loc[i, j] = pval

significance_matrix = p_value_matrix.applymap(
    lambda x: '***' if x < 0.001 else '**' if x < 0.01 else '*' if x < 0.05 else ''
)

plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f", annot_kws={"size": 8})

for i in range(len(correlation_matrix.columns)):
    for j in range(len(correlation_matrix.columns)):
        if significance_matrix.iloc[i, j] != '':
            plt.text(j + 0.5, i + 0.3, significance_matrix.iloc[i, j],
                     ha='center', va='center', color='black', fontsize=8)

plt.title("Correlation Matrix with Significance Annotations", fontsize=14)
plt.xticks(ticks=np.arange(len(correlation_matrix.columns)) + 0.5,
           labels=correlation_matrix.columns, fontsize=7, rotation=90)
plt.yticks(ticks=np.arange(len(correlation_matrix.columns)) + 0.5,
           labels=correlation_matrix.columns, fontsize=7)

plt.tight_layout()
plt.show()

#Correlation between Total Scores & COPE Scores
cope_columns = [col for col in df2.columns if "COPE" in col or col in [
    "DCA8_V2", "Problem_Focused_Coping", "Emotion_Focused_Coping", "Avoidant_Coping",
    "Active_Coping", "Informational_Support", "Positive_Reframing", "Planning",
    "Emotional_Support", "Venting", "Humor", "Acceptance", "Religion", "Self_Blame",
    "Self_Distraction", "Denial", "Substance_Use", "Behavioral_Disengagement"]]

for VX in visits:
    total_columns = [col for col in df1.columns if ("TOTAL" in col.upper())
                     and (VX in col)]
    df_all = pd.concat([df1, df2], axis=1)
    combined_columns = total_columns + cope_columns
    df_combined = df_all[combined_columns].copy()

    df_combined = df_combined.apply(pd.to_numeric, errors='coerce')
    df_combined = df_combined.apply(lambda x: x.fillna(x.mean()), axis=0)

    correlation_matrix = df_combined.corr()

    p_value_matrix = pd.DataFrame(np.ones(correlation_matrix.shape), columns=correlation_matrix.columns,
                                  index=correlation_matrix.index)
    for i in correlation_matrix.columns:
        for j in correlation_matrix.columns:
            r, pval = pearsonr(df_combined[i], df_combined[j])
            p_value_matrix.loc[i, j] = pval

    significance_matrix = p_value_matrix.applymap(
        lambda x: '***' if x < 0.001 else '**' if x < 0.01 else '*' if x < 0.05 else '')

    plt.figure(figsize=(14, 12))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f", annot_kws={"size": 8})

    for i in range(len(correlation_matrix.columns)):
        for j in range(len(correlation_matrix.columns)):
            if significance_matrix.iloc[i, j] != '':
                plt.text(j + 0.5, i + 0.3, significance_matrix.iloc[i, j],
                         ha='center', va='center', color='black', fontsize=8)

    plt.title(f'Correlation Matrix for {VX} (TOTAL + COPE)', fontsize=14)
    plt.xticks(ticks=np.arange(len(correlation_matrix.columns)) + 0.5,
               labels=correlation_matrix.columns, fontsize=7, rotation=90)
    plt.yticks(ticks=np.arange(len(correlation_matrix.columns)) + 0.5,
               labels=correlation_matrix.columns, fontsize=7)
    plt.tight_layout()
    plt.show()

    print(f"Correlation matrix for {VX} (TOTAL + COPE):")
    print(correlation_matrix)
    print("\nP-value matrix:")
    print(p_value_matrix)
    print("\nSignificant correlations (p < 0.05):")
    print(significance_matrix)