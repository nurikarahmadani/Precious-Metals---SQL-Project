import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load data
df = pd.read_csv("C:\Users\NURIKA RAHMADANI\OneDrive - Balikpapan Cerdas\Data Analyst\Project\Precious Metals Data & News (2000-Present)\gold_data_repaired.csv")  # Ganti dengan nama file kamu
df.dropna(subset=['close', 'headlines'], inplace=True)

# 2. Hitung perubahan harga
df['price_change'] = df['close'].diff()
df['target'] = df['price_change'].apply(lambda x: 'up' if x > 0 else 'down')

# 3. Preprocessing teks
df['news_clean'] = df['news_column'].str.lower()

# 4. TF-IDF vektorisasi
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(df['news_clean'])
y = df['target']

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Model klasifikasi
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 7. Evaluasi
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 8. Kata-kata paling berpengaruh
feature_names = vectorizer.get_feature_names_out()
coefs = model.coef_[0]
coef_df = pd.DataFrame({'word': feature_names, 'coef': coefs})
top_positive = coef_df.sort_values(by='coef', ascending=False).head(10)
top_negative = coef_df.sort_values(by='coef').head(10)

# Visualisasi
plt.figure(figsize=(10, 5))
sns.barplot(x='coef', y='word', data=top_positive)
plt.title('Top words that predict gold price going UP')
plt.show()

plt.figure(figsize=(10, 5))
sns.barplot(x='coef', y='word', data=top_negative)
plt.title('Top words that predict gold price going DOWN')
plt.show()
