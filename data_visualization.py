import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("/mnt/c/kaggle_datasets/nih_xray/cleaned_labels.csv")




# Melt the dataframe to make it suitable for plotting
disease_columns = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 
                   'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 
                   'No Finding', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']

# Add a 'Follow-up #' column
df_melted = pd.melt(df, id_vars=['Follow-up #'], value_vars=disease_columns, var_name='Disease', value_name='Has Disease')

# Filter out rows where the disease is not present (i.e., 'Has Disease' == 0)
df_melted = df_melted[df_melted['Has Disease'] == 1]

# Calculate the average number of follow-up visits per disease
avg_visits = df_melted.groupby('Disease')['Follow-up #'].mean().sort_values(ascending=False)

# Create the plot
plt.figure(figsize=(12, 6))
sns.barplot(x=avg_visits.index, y=avg_visits.values, palette="viridis")
plt.title('Average Number of Follow-up Visits per Disease')
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.tight_layout()

# Save the plot as a PNG file
plt.savefig("disease_visits_plot.png")




columns_to_drop = ['Image Index', 'Follow-up #', 'Patient ID', 'Patient Age', 'Patient Gender', 'View Position', 
                  'OriginalImage[Width','Height]', 'OriginalImagePixelSpacing[x','y]', 'Unnamed: 11']
df = df.drop(columns=columns_to_drop)

class_names = df.columns[9:]

#count occurences of each disease
df[class_names] = df[class_names].apply(pd.to_numeric, errors='coerce')
df[class_names] = df[class_names].fillna(0).astype(int)

# count occurences of each disease
disease_counts = df[class_names].sum().sort_values(ascending=False)

#plot bar graph
plt.figure(figsize=(12,6))
sns.barplot(x=disease_counts.index, y=disease_counts.values, hue=disease_counts.index,palette="viridis", legend=True)
plt.xticks(rotation=45, ha="right")
plt.title("Number of Images per Disease (Including No Finding)")
plt.xlabel("Disease")
plt.ylabel("Image Count")
plt.tight_layout()
plt.savefig('disease_counts_plot.png')

# Add a new column 'num_labels' to count the diseases per image (including 'No Finding')
df['num_labels'] = df[class_names].sum(axis=1)

# Plot a histogram of the number of diseases per image (including 'No Finding')
plt.figure(figsize=(8, 6))
sns.histplot(df['num_labels'], bins=10, kde=True, color='royalblue')
plt.title("Distribution of Disease Counts per Image (Including No Finding)")
plt.xlabel("Number of Diseases per Image")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig('disease_counts_histogram.png')

# correlation matrix
correlation_matrix = df[class_names].corr()

#make a heatmap for correlations
plt.figure(figsize=(12,8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
plt.title("Correlation Matrix of Disease Labels")
plt.tight_layout()
plt.savefig("disease_correlation.png")

