import pandas as pd
import pyreadstat
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

df, meta = pyreadstat.read_sav("ACT V1 V2 Questionnaires_ACT EMA Participants.sav")

demographic_columns = [col for col in df.columns if col.startswith('DGF')]
demographic_df = df[demographic_columns]
demographic_df = demographic_df.copy()

visits = ["V1", "V2"]

#Cutoff visual histogram
#V1
issues = [
    "Poor Sleep Quality",
    "Relationship Distress",
    "Low Physical Activity",
    "Moderate Physical Activity",
    "High Physical Activity",
    "Severe Loneliness",
    "Low Burden",
    "Moderate Burden",
    "High Burden",
    "Depression Risk"
]

percentages = [
    70,
    99,
    16,
    40,
    43,
    83,
    21,
    24,
    54,
    30
]

combined_data = list(zip(issues, percentages))
sorted_data = sorted(combined_data, key=lambda x: x[1], reverse=True)

sorted_issues, sorted_percentages = zip(*sorted_data)

plt.figure(figsize=(12, 8))
bars = plt.barh(sorted_issues, sorted_percentages, color='skyblue')
plt.xlabel('Percentage of Participants (%)')
plt.title('Issues Ordered by Percentage of Participants for V1')
plt.gca().invert_yaxis()

for bar, percentage in zip(bars, sorted_percentages):
    plt.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2, f'{percentage}%', va='center', ha='left')

plt.show()

#five abuses
categories = [
    "Emotional Abuse",
    "Physical Abuse",
    "Sexual Abuse",
    "Emotional Neglect",
    "Physical Neglect"
]

levels = ["None/Minimal", "Low", "Moderate", "Severe"]

percentages = [
    [67, 15, 2, 15],  # Emotional Abuse
    [72, 10, 4, 13],  # Physical Abuse
    [78, 2, 6, 13],   # Sexual Abuse
    [56, 26, 6, 11],  # Emotional Neglect
    [80, 6, 4, 9]     # Physical Neglect
]

data = []
for i, category in enumerate(categories):
    for j, level in enumerate(levels):
        data.append([category, level, percentages[i][j]])

df = pd.DataFrame(data, columns=["Category", "Level", "Percentage"])

plt.figure(figsize=(14, 8))
sns.set_style("whitegrid")

bar_plot = sns.barplot(x="Category", y="Percentage", hue="Level", data=df, palette="pastel")

for p in bar_plot.patches:
    if p.get_height() > 0:
        bar_plot.annotate(f'{p.get_height()}%',
                          (p.get_x() + p.get_width() / 2., p.get_height()),
                          ha='center', va='center',
                          xytext=(0, 9),
                          textcoords='offset points')

plt.xlabel('Categories')
plt.ylabel('Percentage of Participants (%)')
plt.title('Abuse and Neglect Categories and Levels')
plt.legend(title='Abuse/Neglect Levels')

plt.tight_layout()
plt.show()


#V2
issues = [
    "Poor Sleep Quality",
    "Relationship Distress",
    "Low Physical Activity",
    "Moderate Physical Activity",
    "High Physical Activity",
    "Severe Loneliness",
    "Low Burden",
    "Moderate Burden",
    "High Burden",
    "Depression Risk"
]

percentages = [
    75,
    99,
    27,
    29,
    43,
    80,
    16,
    32,
    51,
    27
]

combined_data = list(zip(issues, percentages))
sorted_data = sorted(combined_data, key=lambda x: x[1], reverse=True)

sorted_issues, sorted_percentages = zip(*sorted_data)

plt.figure(figsize=(12, 8))
bars = plt.barh(sorted_issues, sorted_percentages, color='skyblue')
plt.xlabel('Percentage of Participants (%)')
plt.title('Issues Ordered by Percentage of Participants for V2')
plt.gca().invert_yaxis()

for bar, percentage in zip(bars, sorted_percentages):
    plt.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2, f'{percentage}%', va='center', ha='left')

plt.show()


#Age
for VX in visits:
    date_column = f'DGF1_{VX}_1'
    if date_column in demographic_df.columns:
        try:
            demographic_df[date_column] = pd.to_datetime(demographic_df[date_column], errors='coerce',
                                                         format='%m/%d/%Y')
        except ValueError:
            demographic_df[date_column] = pd.to_datetime(demographic_df[date_column], errors='coerce',
                                                         format='%Y-%m-%d')

        current_date = datetime.now()

        demographic_df['Age'] = demographic_df[date_column].apply(
            lambda x: current_date.year - x.year - (
                        (current_date.month, current_date.day) < (x.month, x.day)) if pd.notnull(x) else None
        )

        mean_age = demographic_df['Age'].mean()
        std_age = demographic_df['Age'].std()

        print(f"Mean for {VX}: {mean_age:.2f}")
        print(f"Standard deviation for {VX}: {std_age:.2f}")

        bins = range(0, 101, 5)
        labels = [f'{i}-{i + 4}' for i in bins[:-1]]
        demographic_df['Age Group'] = pd.cut(demographic_df['Age'], bins=bins, labels=labels, right=False)

        age_group_counts = demographic_df['Age Group'].value_counts().sort_index()
        age_group_counts = age_group_counts[age_group_counts > 0]

        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=age_group_counts.index, y=age_group_counts.values, palette='pastel')
        for p in ax.patches:
            ax.annotate(f'{p.get_height()}',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                        textcoords='offset points')
        plt.title(f'Age Distribution (5-year groups) for {VX}', fontsize=18)
        plt.xlabel('Age Group', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

#Race
race_categories = {
    "DGF5_VX_0": "White",
    "DGF5_VX_1": "Black or African American",
    "DGF5_VX_2": "Asian",
    "DGF5_VX_3": "Native Hawaiian or Pacific Islander",
    "DGF5_VX_4": "American Indian, Alaska Native",
    "DGF5_VX_5": "Other"
}

for visit in visits:
    visit_columns = [f"DGF5_{visit}_{i}" for i in range(6)]
    available_columns = [col for col in visit_columns if col in df.columns]
    if not available_columns:
        continue
    visit_data = df[available_columns].notnull().sum()

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x=list(race_categories.values()), y=visit_data.values, palette='pastel')
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=12, color='black', xytext=(0, 5),
                    textcoords='offset points')

    plt.xlabel("Race Categories")
    plt.ylabel("Count")
    plt.title(f"Race Distribution for Visit {visit}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

#Mrital Distribution
marriage_categories = {
    "DGF7_VX_0": "Married",
    "DGF7_VX_1": "Widow/Widower",
    "DGF7_VX_999": "Not Applicable"
}

for visit in visits:
    visit_columns = [f"DGF7_{visit}_{i}" for i in [0, 1, 999]]
    available_columns = [col for col in visit_columns if col in df.columns]
    if not available_columns:
        continue
    visit_data = df[available_columns].notnull().sum()

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x=list(marriage_categories.values()), y=visit_data.values, palette='pastel')

    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=12, color='black', xytext=(0, 5),
                    textcoords='offset points')

    plt.xlabel("Marital Status Before Current Marriage")
    plt.ylabel("Count")
    plt.title(f"Marital History Distribution for Visit {visit}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

#Job
df_occupation1 = df["DGF10_V1"]
df_occupation2 = df["DGF10_V2"]
df_occupation1 = df_occupation1.dropna()
df_occupation2 = df_occupation2.dropna()
for VX in ["V1","V2"]:
    plt.figure(figsize=(5, 6))
    ax = sns.countplot(data=df, y=f"DGF10_{VX}", palette='pastel')
    for p in ax.patches:
        ax.annotate(f'{p.get_width()}',
                    (p.get_width(), p.get_y() + p.get_height() / 2),
                    ha='center', va='center', fontsize=5, color='black', xytext=(5, 0),
                    textcoords='offset points')
    plt.title(f'Job Distribution for {VX}', fontsize=8)
    plt.xlabel('Count', fontsize=6)
    plt.ylabel('Job', fontsize=6)
    ax.tick_params(axis='y', labelsize=5)
    plt.show()

#Most other categories
value_mappings = {
    'DGF2_VX': {0: 'Male', 1: 'Female', 2: 'Non-binary'},  # Gender
    'DGF3_VX': {0: 'No', 1: 'Yes'},  # Native English Speaker
    'DGF6_VX': {0: 'No', 1: 'Yes'},  # Hispanic/Latino
    'DGF8_VX': {0: 'Graduate/professional training', 1: '3 or more years of college',
                2: 'Up to 3 years of college', 3: 'High school graduate',
                4: '7 to 12 years (non-graduate)', 5: 'Less than 7 years',
                999: 'Not applicable'},  # Education
    'DGF9_VX': {0: 'Full-time for pay', 1: 'Part-time for pay',
                2: 'Retired', 3: 'Disabled', 4: 'Unemployed', 999: 'Not applicable'},  # Employment
    'DGF13_VX': {0: 'No', 1: 'Yes'},  # Currently vape
    'DGF14_VX': {0: 'No', 1: 'Yes'},  # Ever vaped
    'DGF15_VX': {0: 'No', 1: 'Yes'},  # Currently smoke or use nicotine
    'DGF16_VX': {0: 'No', 1: 'Yes'},  # Ever smoked
    'DGF22_VX': {0: 'Low dose aspirin (baby aspirin)', 1: 'Standard aspirin',
             2: 'Extra strength', 3: 'Does not take aspirin'},  # Aspirin
    'DGF24_VX': {0: 'Over the counter', 1: 'Prescription', 2: 'Do not take Ibuprofen'},  # Ibuprofen
    'DGF27_VX': {0: 'No', 1: 'Yes'},  # Vegetarian
    'DGF28_VX': {0: 'No', 1: 'Yes'},  # Vegan
    'DGF30_VX': {0: 'Much more active', 1: 'Somewhat more active',
                 2: 'About the same', 3: 'Somewhat less active',
                 4: 'Much less active'},  # Physical activity compared to peers
    'DGF31_VX': {0: 'No', 1: 'Yes'},  # Work night shift
    'DGF33_VX': {0: 'No', 1: 'Yes'},  # First marriage
    'DGF35_VX': {0: 'No', 1: 'Yes'},  # Have children/stepchildren
    'DGF38_VX': {0: 'No', 1: 'Yes'},  # Post-menopausal
    'DGF39_VX': {0: 'No', 1: 'Yes'},  # Immunological disorders
    'DGF41_VX': {0: 'No', 1: 'Yes'},  # Arthritis, joint problems
    'DGF43_VX': {0: 'No', 1: 'Yes'},  # Asthma, COPD
    'DGF45_VX': {0: 'No', 1: 'Yes'},  # Strokes, vascular disease
    'DGF47_VX': {0: 'No', 1: 'Yes'},  # Implantable defibrillator or heart problems
    'DGF49_VX': {0: 'No', 1: 'Yes'},  # Diabetes, hypoglycemia
    'DGF51_VX': {0: 'No', 1: 'Yes'},  # Seizures, epilepsy
    'DGF53_VX': {0: 'No', 1: 'Yes'},  # Liver, kidney, urinary tract problems
    'DGF55_VX': {0: 'No', 1: 'Yes'},  # Thyroid disorders
    'DGF57_VX': {0: 'No', 1: 'Yes'},  # Hormone problems
    'DGF59_VX': {0: 'No', 1: 'Yes'},  # Ulcerative colitis
    'DGF61_VX': {0: 'No', 1: 'Yes'},  # Digestive problems
    'DGF63_VX': {0: 'No', 1: 'Yes'},  # History of cancer/tumor
    'DGF66_VX': {0: 'No', 1: 'Yes'},  # High blood pressure
    'DGF68_VX': {0: 'No', 1: 'Yes'},  # Cardiovascular problems
    'DGF70_VX': {0: 'No', 1: 'Yes'},  # Other major health problems
    'DGF73_VX': {0: 'No', 1: 'Yes'},  # Hospitalized/surgery past year
}

description_mappings = {
    'DGF2_VX': 'Gender',
    'DGF3_VX': 'Native English Speaker',
    'DGF6_VX': 'Hispanic/Latino',
    'DGF8_VX': 'Education',
    'DGF9_VX': 'Employment',
    'DGF13_VX': 'Currently vape',
    'DGF14_VX': 'Ever vaped',
    'DGF15_VX': 'Currently smoke or use nicotine',
    'DGF16_VX': 'Ever smoked',
    'DGF22_VX': 'Aspirin',
    'DGF24_VX': 'Ibuprofen',
    'DGF27_VX': 'Vegetarian',
    'DGF28_VX': 'Vegan',
    'DGF30_VX': 'Physical activity compared to peers',
    'DGF31_VX': 'Work night shift',
    'DGF33_VX': 'First marriage',
    'DGF35_VX': 'Have children/stepchildren',
    'DGF38_VX': 'Post-menopausal',
    'DGF39_VX': 'Immunological disorders',
    'DGF41_VX': 'Arthritis, joint problems',
    'DGF43_VX': 'Asthma, COPD',
    'DGF45_VX': 'Strokes, vascular disease',
    'DGF47_VX': 'Implantable defibrillator or heart problems',
    'DGF49_VX': 'Diabetes, hypoglycemia',
    'DGF51_VX': 'Seizures, epilepsy',
    'DGF53_VX': 'Liver, kidney, urinary tract problems',
    'DGF55_VX': 'Thyroid disorders',
    'DGF57_VX': 'Hormone problems',
    'DGF59_VX': 'Ulcerative colitis',
    'DGF61_VX': 'Digestive problems',
    'DGF63_VX': 'History of cancer/tumor',
    'DGF66_VX': 'High blood pressure',
    'DGF68_VX': 'Cardiovascular problems',
    'DGF70_VX': 'Other major health problems',
    'DGF73_VX': 'Hospitalized/surgery past year'
}

for visit in visits:
   visit_value_mappings = {col.replace("VX", visit): val for col, val in value_mappings.items()}
   visit_description_mappings = {col.replace("VX", visit): val for col, val in description_mappings.items()}

   for col, mapping in visit_value_mappings.items():
       if col in demographic_df.columns:
           demographic_df[col] = demographic_df[col].map(mapping).fillna("Unknown")

   for col in demographic_df.columns:
       if col in visit_description_mappings:
           description = visit_description_mappings.get(col, col)
           plt.figure(figsize=(10, 6))
           ax = sns.countplot(data=demographic_df, x=col, palette='pastel')
           for p in ax.patches:
               ax.annotate(f'{p.get_height()}',
                           (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='center', fontsize=12, color='black', xytext=(0, 5),
                           textcoords='offset points')
           plt.title(f'{description} Distribution for {visit}', fontsize=16)
           plt.xlabel(description, fontsize=14)
           plt.ylabel('Count', fontsize=14)
           plt.show()


