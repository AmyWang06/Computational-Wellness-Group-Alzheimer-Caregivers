import pandas as pd
import pyreadstat
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np


# 1. Load Data
df, meta = pyreadstat.read_sav("data/ACT V1 V2 Questionnaires_ACT EMA Participants.sav")
visits = ["V1", "V2"]


# 2. Calculation Logic for Research Metrics
def calculate_research_stats(df, visit):
    stats = {}
    sample_col = f"PSQI_Total_{visit}"
    actual_psqi_col = next((c for c in df.columns if c.upper() == sample_col.upper()), None)

    if actual_psqi_col is None or df[actual_psqi_col].isnull().all():
        print(f"Warning: No valid data found for {visit}. Skipping...")
        return None

    def get_col(base_name):
        full_name = f"{base_name}_{visit}"
        return next((c for c in df.columns if c.upper() == full_name.upper()), None)

    col = get_col("PSQI_Total")
    if col: stats["Poor Sleep Quality"] = (df[col] > 5).mean() * 100

    col = get_col("CSI_Total")
    if col: stats["Relationship Distress"] = (df[col] < 104.5).mean() * 100

    col = get_col("IPAQ_TOTAL")
    if col:
        stats["Low Physical Activity"] = (df[col] < 600).mean() * 100
        stats["Moderate Physical Activity"] = ((df[col] >= 600) & (df[col] <= 3000)).mean() * 100
        stats["High Physical Activity"] = (df[col] > 3000).mean() * 100

    col = get_col("UCLALS_Total")
    if col: stats["Severe Loneliness"] = (df[col] >= 31).mean() * 100

    col = get_col("CESD_Total")
    if col: stats["Depression Risk"] = (df[col] >= 16).mean() * 100

    pswq_col = "PSWQ_Total_WI"
    if pswq_col in df.columns:
        stats["Excessive Worry"] = (df[pswq_col] > 45).mean() * 100

    zbi_total_col = get_col("ZBI_Total12")
    zbi_cutoff_col = get_col("ZBI_CutOff")
    if zbi_total_col in df.columns and zbi_cutoff_col in df.columns:
        is_probable_depression = df[zbi_total_col] >= df[zbi_cutoff_col]
        stats["Probable Depression (via ZBI)"] = is_probable_depression.mean() * 100
        stats["High Caregiver Burden"] = (df[zbi_total_col] >= 21).mean() * 100

    return stats


# 3. Research Issues Visualization
for vx in visits:
    res = calculate_research_stats(df, vx)
    if res:
        sorted_res = dict(sorted(res.items(), key=lambda item: item[1], reverse=True))
        plt.figure(figsize=(12, 7))
        bars = plt.barh(list(sorted_res.keys()), list(sorted_res.values()), color='skyblue')
        plt.xlabel('Percentage of Participants (%)')
        plt.title(f'Prevalence of Health & Relationship Issues ({vx})')
        plt.gca().invert_yaxis()
        for bar in bars:
            plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2, f'{bar.get_width():.1f}%', va='center')
        plt.tight_layout()
        plt.show()
        plt.close()


# 4. CTQ Multi-level Analysis
def plot_ctq(df, visit="V1"):
    ctq_map = {
        "Emotional Abuse": {"col": f"CTQ_EA_{visit}", "cutoffs": [8, 12, 15]},
        "Physical Abuse": {"col": f"CTQ_PA_{visit}", "cutoffs": [7, 9, 12]},
        "Sexual Abuse": {"col": f"CTQ_SA_{visit}", "cutoffs": [5, 7, 12]},
        "Emotional Neglect": {"col": f"CTQ_EN_{visit}", "cutoffs": [9, 14, 17]},
        "Physical Neglect": {"col": f"CTQ_PN_{visit}", "cutoffs": [7, 9, 12]}
    }
    ctq_results = []
    for category, info in ctq_map.items():
        col = info["col"]
        if col in df.columns:
            vals = df[col]
            c = info["cutoffs"]
            none = (vals <= c[0]).mean() * 100
            low = ((vals > c[0]) & (vals <= c[1])).mean() * 100
            mod = ((vals > c[1]) & (vals <= c[2])).mean() * 100
            sev = (vals > c[2]).mean() * 100
            for level, p in zip(["None", "Low", "Moderate", "Severe"], [none, low, mod, sev]):
                ctq_results.append([category, level, p])

    if ctq_results:
        ctq_df = pd.DataFrame(ctq_results, columns=["Category", "Level", "Percentage"])
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(data=ctq_df, x="Category", y="Percentage", hue="Level", palette="pastel")
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}%',
                            (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='bottom',
                            xytext=(0, 5), textcoords='offset points')
        plt.title(f"Childhood Trauma Categories and Levels ({visit})")
        plt.ylabel("Percentage (%)")
        plt.tight_layout()
        plt.show()
        plt.close()


plot_ctq(df, "V1")
plot_ctq(df, "V2")

# 5. Demographics
demographic_df = df.copy()

# Age Calculation & Plotting
for VX in visits:
    date_column = f'DGF1_{VX}_1'
    if date_column in demographic_df.columns:
        demographic_df[date_column] = pd.to_datetime(demographic_df[date_column], errors='coerce')
        current_date = datetime.now()
        demographic_df['Age'] = demographic_df[date_column].apply(
            lambda x: current_date.year - x.year - ((current_date.month, current_date.day) < (x.month, x.day))
            if pd.notnull(x) else np.nan
        )

        if not demographic_df['Age'].dropna().empty:
            mean_age = demographic_df['Age'].mean()
            std_age = demographic_df['Age'].std()
            print(f"Mean for {VX}: {mean_age:.2f}, STD: {std_age:.2f}")
            title_str = f'Age Distribution for {VX}\n(Mean={mean_age:.2f}, SD={std_age:.2f})'

            bins = range(0, 101, 5)
            labels = [f'{i}-{i + 4}' for i in bins[:-1]]
            demographic_df['Age Group'] = pd.cut(demographic_df['Age'], bins=bins, labels=labels, right=False)
            age_group_counts = demographic_df['Age Group'].value_counts().sort_index()
            age_group_counts = age_group_counts[age_group_counts > 0]

            plt.figure(figsize=(10, 6))
            ax = sns.barplot(x=age_group_counts.index, y=age_group_counts.values, palette='pastel')
            for p in ax.patches:
                ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                            textcoords='offset points')
            plt.title(title_str, fontsize=18)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            plt.close()

# Race
race_categories = {0: "White", 1: "Black or African American", 2: "Asian",
                   3: "Native Hawaiian or Pacific Islander", 4: "American Indian, Alaska Native", 5: "Other"}
for visit in visits:
    visit_columns = [f"DGF5_{visit}_{i}" for i in range(6)]
    available_columns = [col for col in visit_columns if col in df.columns]
    if available_columns:
        visit_data = df[available_columns].notnull().sum()
        plt.figure(figsize=(8, 5))
        ax = sns.barplot(x=list(race_categories.values()), y=visit_data.values, palette='pastel')
        plt.title(f"Race Distribution for Visit {visit}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        plt.close()

# Mappings for Categorical Data
value_mappings = {
    'DGF2_VX': {0: 'Male', 1: 'Female', 2: 'Non-binary'},
    'DGF3_VX': {0: 'No', 1: 'Yes'},
    'DGF6_VX': {0: 'No', 1: 'Yes'},
    'DGF8_VX': {0: 'Graduate/professional training', 1: '3 or more years of college',
                2: 'Up to 3 years of college', 3: 'High school graduate',
                4: '7 to 12 years (non-graduate)', 5: 'Less than 7 years', 999: 'Not applicable'},
    'DGF9_VX': {0: 'Full-time for pay', 1: 'Part-time for pay', 2: 'Retired', 3: 'Disabled', 4: 'Unemployed',
                999: 'Not applicable'},
    'DGF13_VX': {0: 'No', 1: 'Yes'}, 'DGF14_VX': {0: 'No', 1: 'Yes'},
    'DGF15_VX': {0: 'No', 1: 'Yes'}, 'DGF16_VX': {0: 'No', 1: 'Yes'},
    'DGF22_VX': {0: 'Low dose aspirin', 1: 'Standard aspirin', 2: 'Extra strength', 3: 'Does not take aspirin'},
    'DGF24_VX': {0: 'Over the counter', 1: 'Prescription', 2: 'Do not take Ibuprofen'},
    'DGF27_VX': {0: 'No', 1: 'Yes'}, 'DGF28_VX': {0: 'No', 1: 'Yes'},
    'DGF30_VX': {0: 'Much more active', 1: 'Somewhat more active', 2: 'About the same', 3: 'Somewhat less active',
                 4: 'Much less active'},
    'DGF31_VX': {0: 'No', 1: 'Yes'}, 'DGF33_VX': {0: 'No', 1: 'Yes'},
    'DGF35_VX': {0: 'No', 1: 'Yes'}, 'DGF38_VX': {0: 'No', 1: 'Yes'},
    'DGF39_VX': {0: 'No', 1: 'Yes'}, 'DGF41_VX': {0: 'No', 1: 'Yes'},
    'DGF43_VX': {0: 'No', 1: 'Yes'}, 'DGF45_VX': {0: 'No', 1: 'Yes'},
    'DGF47_VX': {0: 'No', 1: 'Yes'}, 'DGF49_VX': {0: 'No', 1: 'Yes'},
    'DGF51_VX': {0: 'No', 1: 'Yes'}, 'DGF53_VX': {0: 'No', 1: 'Yes'},
    'DGF55_VX': {0: 'No', 1: 'Yes'}, 'DGF57_VX': {0: 'No', 1: 'Yes'},
    'DGF59_VX': {0: 'No', 1: 'Yes'}, 'DGF61_VX': {0: 'No', 1: 'Yes'},
    'DGF63_VX': {0: 'No', 1: 'Yes'}, 'DGF66_VX': {0: 'No', 1: 'Yes'},
    'DGF68_VX': {0: 'No', 1: 'Yes'}, 'DGF70_VX': {0: 'No', 1: 'Yes'},
    'DGF73_VX': {0: 'No', 1: 'Yes'}
}

description_mappings = {
    'DGF2_VX': 'Gender', 'DGF3_VX': 'Native English Speaker', 'DGF6_VX': 'Hispanic/Latino',
    'DGF8_VX': 'Education', 'DGF9_VX': 'Employment', 'DGF13_VX': 'Currently vape',
    'DGF14_VX': 'Ever vaped', 'DGF15_VX': 'Currently smoke/nicotine', 'DGF16_VX': 'Ever smoked',
    'DGF22_VX': 'Aspirin', 'DGF24_VX': 'Ibuprofen', 'DGF27_VX': 'Vegetarian', 'DGF28_VX': 'Vegan',
    'DGF30_VX': 'Activity Level', 'DGF31_VX': 'Night Shift', 'DGF33_VX': 'First Marriage',
    'DGF35_VX': 'Children', 'DGF38_VX': 'Post-menopausal', 'DGF39_VX': 'Immune Disorders',
    'DGF41_VX': 'Arthritis', 'DGF43_VX': 'Asthma/COPD', 'DGF45_VX': 'Strokes',
    'DGF47_VX': 'Heart Problems', 'DGF49_VX': 'Diabetes', 'DGF51_VX': 'Seizures',
    'DGF53_VX': 'Liver/Kidney', 'DGF55_VX': 'Thyroid', 'DGF57_VX': 'Hormone Problems',
    'DGF59_VX': 'Ulcerative Colitis', 'DGF61_VX': 'Digestive Problems', 'DGF63_VX': 'Cancer History',
    'DGF66_VX': 'High Blood Pressure', 'DGF68_VX': 'Cardiovascular', 'DGF70_VX': 'Other Health',
    'DGF73_VX': 'Hospitalized/Surgery'
}

# Final Plotting
for visit in visits:
    v_val_map = {col.replace("VX", visit): val for col, val in value_mappings.items()}
    v_desc_map = {col.replace("VX", visit): val for col, val in description_mappings.items()}

    for col in [c for c in v_desc_map.keys() if c in demographic_df.columns]:
        plot_series = demographic_df[col].map(v_val_map.get(col, {})).fillna("Unknown")
        plt.figure(figsize=(10, 6))
        ax = sns.countplot(x=plot_series, palette='pastel')
        for p in ax.patches:
            if p.get_height() > 0:
                ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', xytext=(0, 5), textcoords='offset points')
        plt.title(f'{v_desc_map[col]} ({visit})')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        plt.close()