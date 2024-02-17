import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from wordcloud import WordCloud
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



# Ensure necessary NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing function
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove numbers, special characters, and punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I | re.A)
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    # Reconstruct the text from tokens
    return ' '.join(lemmatized_tokens)

# Load dataset
test_data = pd.read_csv('dreaddit/dreaddit-test.csv')
train_data = pd.read_csv('dreaddit/dreaddit-train.csv')

# Display basic info about the dataset
print(train_data.info())

# Distribution of labels
plt.figure(figsize=(6, 4))
sns.countplot(data=train_data, x='label')
plt.title('Distribution of Stress Labels')
plt.tight_layout()
plt.savefig('distribution_of_stress_labels.png')  # Save the plot
plt.show()

# Post lengths
train_data['post_length'] = train_data['text'].apply(len)
plt.figure(figsize=(8, 6))
sns.histplot(data=train_data, x='post_length', bins=30, kde=True, hue='label')
plt.title('Distribution of Post Lengths')
plt.tight_layout()
plt.savefig('distribution_of_post_lengths.png')  # Save the plot
plt.show()

# Subreddit distribution
plt.figure(figsize=(10, 8))
sns.countplot(data=train_data, y='subreddit', order=train_data['subreddit'].value_counts().index)
plt.title('Distribution of Posts across Subreddits')
plt.tight_layout()
plt.savefig('distribution_of_posts_across_subreddits.png')  # Save the plot
plt.show()

# Convert 'social_timestamp' to datetime
train_data['date'] = pd.to_datetime(train_data['social_timestamp'], unit='s')

# Plotting posts over time
plt.figure(figsize=(10, 6))
train_data.resample('M', on='date').size().plot(label='Total posts per month', color='blue')
plt.title('Frequency of Posts Over Time')
plt.xlabel('Month')
plt.ylabel('Number of Posts')
plt.legend()
plt.tight_layout()
plt.savefig('frequency_of_posts_over_time.png')  # Save the plot
plt.show()


# Preprocess text data
train_data['preprocessed_text'] = train_data['text'].apply(preprocess_text)
test_data['preprocessed_text'] = test_data['text'].apply(preprocess_text)

# Vectorizing the preprocessed text
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_train = vectorizer.fit_transform(train_data['preprocessed_text'])
X_test = vectorizer.transform(test_data['preprocessed_text'])

# Labels
y_train = train_data['label']
y_test = test_data['label']

# Model training with hyperparameter tuning
parameters = {'alpha': [0.01, 0.1, 1, 10]}
nb_model = MultinomialNB()
clf = GridSearchCV(nb_model, parameters, cv=5, scoring='accuracy')
clf.fit(X_train, y_train)

# Output best parameters and score
print("Best Parameters for Naive Bayes:", clf.best_params_)
print("Best Score for Naive Bayes:", clf.best_score_)

# Training RandomForestClassifier

# Define a range of hyperparameters for tuning
param_grid = {
    'n_estimators': [100, 200, 300],  # Number of trees in the forest
    'max_depth': [None, 10, 20],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4]  # Minimum number of samples required to be at a leaf node
}

# Initialize the RandomForest model
rf = RandomForestClassifier(random_state=42)

# Initialize the GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=0, scoring='accuracy')

# Fit the grid search to the data
grid_search.fit(X_train, y_train)


# Best parameters and score
print("Best Parameters for Random Forest:", grid_search.best_params_)
print("Best Score for Random Forest:", grid_search.best_score_)

# Model evaluation
nb_predictions = clf.best_estimator_.predict(X_test)

rf_best_model = grid_search.best_estimator_
rf_predictions = rf_best_model.predict(X_test)

# Evaluating both models
print("Naive Bayes Classification Report:")
print(classification_report(y_test, nb_predictions))

print("Random Forest Classification Report:")
print(classification_report(y_test, rf_predictions))

cm = confusion_matrix(y_test, rf_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf_best_model.classes_)
disp.plot()
plt.show()


# Getting feature importances
importances = rf_best_model.feature_importances_

# Plotting the top N feature importances
top_n = 20
indices = np.argsort(importances)[-top_n:]
plt.figure(figsize=(10, 8))
plt.title("Feature Importances")
plt.barh(range(top_n), importances[indices], color='b', align='center')
plt.yticks(range(top_n), [vectorizer.get_feature_names_out()[i] for i in indices])
plt.xlabel("Relative Importance")
plt.tight_layout()  # Ensure everything fits without clipping
plt.savefig('top_n_feature_importances.png')  # Save the plot before showing it
plt.show()

# Generate word clouds for stressed vs. non-stressed posts
stressed_text = ' '.join(train_data.loc[train_data['label'] == 1, 'preprocessed_text'])
non_stressed_text = ' '.join(train_data.loc[train_data['label'] == 0, 'preprocessed_text'])

wordcloud_stressed = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = stopwords.words('english'),
                min_font_size = 10).generate(stressed_text)

wordcloud_non_stressed = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = stopwords.words('english'),
                min_font_size = 10).generate(non_stressed_text)

# Plot the word clouds
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud_stressed)
plt.axis("off")
plt.title("Word Cloud - Stressed Posts")
plt.tight_layout(pad = 0)
plt.show()

plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud_non_stressed)
plt.axis("off")
plt.title("Word Cloud - Non-Stressed Posts")
plt.tight_layout(pad = 0)
plt.show()

wordcloud_stressed.to_file('wordcloud_stressed.png')
wordcloud_non_stressed.to_file('wordcloud_non_stressed.png')

# Assuming 'post_length' is already calculated
plt.figure(figsize=(10, 6))
sns.histplot(data=train_data, x='post_length', hue='label', bins=50, kde=True, palette='coolwarm')
plt.title('Distribution of Post Lengths by Stress Label')
plt.xlabel('Post Length')
plt.ylabel('Frequency')
plt.xlim(0, train_data['post_length'].quantile(0.95))  # Limiting to 95th percentile for better visualization
plt.tight_layout()  # Ensure everything fits without clipping
plt.savefig('distribution_of_post_lengths_by_stress_label.png')  # Save the plot before showing it
plt.show()


# Selecting a subset of features for correlation analysis
features = ['syntax_ari', 'sentiment', 'lex_liwc_WC', 'lex_liwc_posemo', 'lex_liwc_negemo', 'label']
correlation_matrix = train_data[features].corr()

# Plotting the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Selected Features')
plt.tight_layout()
plt.savefig('correlation_matrix_of_selected_features.png')  # Save the plot
plt.show()
