import pandas as pd

# Load the two source datasets
x = pd.read_csv('./datasets/XSS_dataset.csv')
s = pd.read_csv('./datasets/Modified_SQL_Dataset.csv')

# Keep only the text + label columns, rename to a common schema
x2 = x[['Sentence', 'Label']].rename(columns={'Sentence': 'text'})
s2 = s[['Query', 'Label']].rename(columns={'Query': 'text'})

x2['type'] = 'xss'
s2['type'] = 'sqli'

# Stack both datasets into one combined table
df = pd.concat([x2, s2], ignore_index=True)
df = df.dropna(subset=['text'])
df['text'] = df['text'].astype(str)

print('Combined shape:', df.shape)
print(df['Label'].value_counts())

df.to_csv('combined.csv', index=False)
print("Saved combined.csv")