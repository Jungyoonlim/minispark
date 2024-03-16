from main import predicted_label
import matplotlib.pyplot as plt
import seaborn as sns

predicted_labels = [int(row.prediction) for row in predicted_label]

# Count the occurrences of each label
label_counts = [predicted_labels.count(0), predicted_labels.count(1)]

# Create a bar plot
plt.figure(figsize=(8, 6))
sns.barplot(x=['Negative', 'Positive'], y=label_counts)
plt.title('Predicted Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()