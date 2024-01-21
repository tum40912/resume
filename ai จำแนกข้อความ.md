import pandas as pd
df=pd.read_csv('pizza-UTF8-traindataset-3.csv')
df

import re
import string
from pythainlp.corpus.common import thai_stopwords
from pythainlp.tokenize import word_tokenize

thai_stopwords = list(thai_stopwords())
thai_stopwords

def perform_removal(word):
    #กำจัดช่องว่างก่อน/หลังคำ
    word = word.strip()

    #เปลี่ยนภาษาอังกฤษเป็นตัวอักษรตัวเล็ก
    word = word.lower()

    #กำจัดเครื่องหมายวรรคตอน
    word = word.translate(str.maketrans('','', string.punctuation))

    #กำจัด stop words และตัวเลขโดดๆ
    if(word.isdigit() ):
        return ""
    else:
        return word

def clean_text(text):
  text = "".join(u for u in text if u not in ("?", ".", ";", ":", "!", '"', "ๆ", "ฯ","'"))
  tokens=word_tokenize(text, engine="oskut", keep_whitespace=False)

  tokens = [word for word in tokens if word.lower not in thai_stopwords]

  tokens = [word for word in tokens if len(word)>1]

  tokens = [perform_removal(word) for word in tokens]


  text = ' '.join(tokens)
  return text

  df['text'] = df['text'].apply(lambda x:clean_text(x))
df

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

df_pos = df[df['class'] == 0]
pos_word_all = " ".join(text for text in df_pos['text'])
reg = r"[ก-๙a-zA-Z']+"
fp = 'sudsakorn.ttf'
wordcloud = WordCloud(stopwords=thai_stopwords, background_color = 'white', max_words=2000, height = 2000, width=4000, font_path=fp, regexp=reg).generate(pos_word_all)
plt.figure(figsize = (16,8))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

df_neg = df[df['class'] == 1]
neg_word_all = " ".join(text for text in df_neg['text'])
wordcloud = WordCloud(stopwords=thai_stopwords, background_color = 'white', max_words=2000, height = 2000, width=4000, font_path=fp, regexp=reg).generate(neg_word_all)
plt.figure(figsize = (16,8))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df, test_size=0.20, stratify=df['class'])

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer


vec = CountVectorizer(
    ngram_range=(1,3)
)

#tfidf = TfidfVectorizer(
#    ngram_range=(1,3)
#)

vec.fit_transform(df_train['text'])
vec.vocabulary_

import numpy as np

#สุ่มช่วงของ 5 เอกสารที่ติดกันมาทดลองใช้งาน
count_vector= vec.fit_transform(df_train['text'][:10])
count_array = np.array(count_vector.todense())

#แปลงเป็น DataFrame เพื่อง่ายแก่การอ่าน
df_x = pd.DataFrame(count_array,columns=vec.get_feature_names_out())
df_x

X_train = vec.fit_transform(df_train.text)
X_test = vec.transform(df_test.text)

y_train = df_train['class']
y_test = df_test['class']

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, classification_report

dt=DecisionTreeClassifier()
dt.fit(X_train, y_train)

preds = dt.predict(X_test)
print(classification_report(y_test, preds))

from sklearn.tree import export_graphviz
import graphviz

# Export decision tree to graphviz format
dot_data = export_graphviz(dt, out_file=None,
                           #feature_names=df.columns,
                           class_names=['Neg','Pos'],
                           filled=True, rounded=True,
                           special_characters=True)

# Visualize decision tree using graphviz
graph = graphviz.Source(dot_data)
graph.format = 'png'
graph.render("sizzler_decision_tree")
graph # Display decision tree

