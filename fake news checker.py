from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from bs4 import BeautifulSoup
import pandas
import contractions
import string
import requests
import nltk
nltk.download('stopwords')
def remove_lines(text):
    return text.strip("\n")
def remove_contractions(text):
    return ' '.join([contractions.fix(word) for word in text.split()])
def remove_punctuation(text):
    return ''.join([words for words in text if words not in string.punctuation])
def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word not in stopword])
def remove_quotemarks(text):
    return text.translate({ord(c): None for c in '“”’'})
stopword = nltk.corpus.stopwords.words('english')
num = int(input("Enter sample size: "))
fn = pandas.read_csv("Fake.csv", nrows=num)
fn.drop_duplicates(inplace=True)
tn = pandas.read_csv("True.csv", nrows=num)
tn.drop_duplicates(inplace=True)
fn['truth'] = 0
tn['truth'] = 1
fn.rename(columns={0: "title", 1: "text", 2: "subject", 3: "date", 4: "truth" }, inplace=True)
news = pandas.concat([tn, fn], axis=0, ignore_index=True)
news['title']=news['title'].apply(lambda x: remove_quotemarks(remove_stopwords(remove_punctuation(remove_contractions(x)))))
news['text']=news['text'].apply(lambda x: remove_quotemarks(remove_stopwords(remove_punctuation(remove_contractions(x)))))
y = news['truth'].astype('int')
X = news['text']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
text_clf = Pipeline([('vect', TfidfVectorizer(ngram_range=(2,3), binary=True)), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
text_clf = text_clf.fit(X_train, y_train)
ton = int(input("What type of news would you like to test the model on?:\n1: True\n2: Mostly True\n3: Half True\n4: Barely True\n5: False\n6: Extremely False\n"))
pages = ["true", "mostly-true", "half-true", "barely-true", "false", "pants-fire"]
types = ["true", "mostly true", "half true", "barely true", "false", "extremely false"]
soup = BeautifulSoup(requests.get('https://www.politifact.com/factchecks/list/?ruling=' + pages[ton - 1]).text, "html.parser")
lst = [s.replace('\n', "") for s in [data.get_text() for data in soup.find_all("div",attrs={"class":"m-statement__quote"})]]
res = []
[res.append(x) for x in lst if x not in res]
cat = [types[ton - 1]] * len(res)
d = {'text':res,'truth':cat}
df = pandas.DataFrame(data = d)
df1 = df.copy()
df['text'] = df['text'].apply(lambda x: remove_quotemarks(remove_stopwords(remove_punctuation(remove_contractions(x)))))
sample_predict = text_clf.predict(pandas.Series.tolist(df["text"]))
df1['predicted'] = sample_predict.tolist()
df1['predicted'].mask(df1['predicted'] == 0, 'false', inplace=True)
df1['predicted'].mask(df1['predicted'] == 1, 'true', inplace=True)
print(df1.head(df1.size))