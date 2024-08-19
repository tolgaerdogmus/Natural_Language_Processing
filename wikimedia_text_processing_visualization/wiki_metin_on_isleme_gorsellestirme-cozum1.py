#################################################
# WIKI 1 - Metin önişleme ve Görselleştirme (NLP - Text Preprocessing & Text Visualization)
#################################################

###################f##############################
# Problemin Tanımı
#################################################
# Wikipedia örnek datasından metin ön işleme, temizleme işlemleri gerçekleştirip, görselleştirme çalışmaları yapmak.

#################################################
# Veri Seti Hikayesi
#################################################
# Wikipedia datasından alınmış metinleri içermektedir.

#################################################
# Gerekli Kütüphaneler ve ayarlar


import re
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
from textblob import Word, TextBlob
from warnings import filterwarnings


filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 200)

# Datayı okumak
df = pd.read_csv("wiki_data.csv", index_col=0)
df.head()
df = df[:2000]

df.head()
df.shape

#################################################
# Görevler:
#################################################

# Görev 1: Metindeki ön işleme işlemlerini gerçekleştirecek bir fonksiyon yazınız.
# •	Büyük küçük harf dönüşümünü yapınız.
# •	Noktalama işaretlerini çıkarınız.
# •	Numerik ifadeleri çıkarınız.


# def clean_text(text):
#     # Normalizing Case Folding
#     text = text.str.lower()
#     # Punctuations
#     text = text.str.replace(r'[^\w\s]', '')
#     text = text.str.replace("\n" , '')
#     # Numbers
#     text = text.str.replace('\d', '')
#     return text

# def clean_text(text):
#     # Normalizing Case Folding
#     text = text.str.lower()
#     # Punctuations
#     text = text.str.replace(r'[^\w\s]', '', regex=True)
#     # Remove newlines
#     text = text.str.replace("\n", ' ', regex=False)
#     # Numbers
#     text = text.str.replace(r'\d+', '', regex=True)
#     return text

# df["text"] = clean_text(df["text"])

# df.head()

nltk.download('punkt')
nltk.download('stopwords')

def clean_text_nltk(text):
    # Ensure text is a string
    text = str(text)
    
    # Lowercase
    text = text.lower()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove punctuation and numbers
    tokens = [word for word in tokens if word.isalpha()]
    
    # Remove stopwords (optional)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Join tokens back into string
    cleaned_text = ' '.join(tokens)
    
    return cleaned_text

# Görev 2: Metin içinde öznitelik çıkarımı yaparken önemli olmayan kelimeleriçıkaracak fonksiyon yazınız.

# def remove_stopwords(text):
#     stop_words = stopwords.words('English')
#     text = text.apply(lambda x: " ".join(x for x in str(x).split() if x not in stop_words))
#     return text

df["text"] = df["text"].apply(clean_text_nltk)




# Görev 3: Metinde az tekrarlayan kelimeleri bulunuz.

pd.Series(' '.join(df['text']).split()).value_counts()[-1000:]



# Görev 4: Metinde az tekrarlayan kelimeleri metin içerisinden çıkartınız. (İpucu: lambda fonksiyonunu kullanınız.)

sil = pd.Series(' '.join(df['text']).split()).value_counts()[-1000:]
df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in sil))




# Görev 5: Metinleri tokenize edip sonuçları gözlemleyiniz.

df["text"].apply(lambda x: TextBlob(x).words)


# Görev 6: Lemmatization işlemini yapınız.
# ran, runs, running -> run (normalleştirme)

df['text'] = df['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

df.head()

# Görev 7: Metindeki terimlerin frekanslarını hesaplayınız. (İpucu: Barplot grafiği için gerekli)

tf = df["text"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index() # kodu güncellemek gerekecek

tf.head()

# Görev 8: Barplot grafiğini oluşturunuz.

# Sütunların isimlendirilmesi
tf.columns = ["words", "tf"]
# 2000'den fazla geçen kelimelerin görselleştirilmesi
tf[tf["tf"] > 2000].plot.bar(x="words", y="tf")
plt.show()

# Kelimeleri WordCloud ile görselleştiriniz.

# kelimeleri birleştirdik
text = " ".join(i for i in df["text"])

# wordcloud görselleştirmenin özelliklerini belirliyoruz
wordcloud = WordCloud(max_font_size=50,
max_words=100,
background_color="black").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# Görev 9: Tüm aşamaları tek bir fonksiyon olarak yazınız.
# •	Metin ön işleme işlemlerini gerçekleştiriniz.
# •	Görselleştirme işlemlerini fonksiyona argüman olarak ekleyiniz.
# •	Fonksiyonu açıklayan 'docstring' yazınız.

df = pd.read_csv("Modül_8_Dogal_Dil_İşleme/datasets/wiki_data.csv", index_col=0)


def wiki_preprocess(text, Barplot=False, Wordcloud=False):
    """
    Textler üzerinde ön işleme işlemleri yapar.

    :param text: DataFrame'deki textlerin olduğu değişken
    :param Barplot: Barplot görselleştirme
    :param Wordcloud: Wordcloud görselleştirme
    :return: text


    Example:
            wiki_preprocess(dataframe[col_name])

    """
    # Normalizing Case Folding
    text = text.str.lower()
    # Punctuations
    text = text.str.replace('[^\w\s]', '')
    text = text.str.replace("\n", '')
    # Numbers
    text = text.str.replace('\d', '')
    # Stopwords
    sw = stopwords.words('English')
    text = text.apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))
    # Rarewords / Custom Words
    sil = pd.Series(' '.join(text).split()).value_counts()[-1000:]
    text = text.apply(lambda x: " ".join(x for x in x.split() if x not in sil))


    if Barplot:
        # Terim Frekanslarının Hesaplanması
        tf = text.apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
        # Sütunların isimlendirilmesi
        tf.columns = ["words", "tf"]
        # 5000'den fazla geçen kelimelerin görselleştirilmesi
        tf[tf["tf"] > 2000].plot.bar(x="words", y="tf")
        plt.show()

    if Wordcloud:
        # Kelimeleri birleştirdik
        text = " ".join(i for i in text)
        # wordcloud görselleştirmenin özelliklerini belirliyoruz
        wordcloud = WordCloud(max_font_size=50,
                              max_words=100,
                              background_color="white").generate(text)
        plt.figure()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()

    return text

wiki_preprocess(df["text"])

wiki_preprocess(df["text"], True, True)

###############################################################

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
from collections import Counter

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
stop_words = set(stopwords.words('english'))

def wiki_preprocess(df, text_column, Barplot=False, Wordcloud=False):
    """
    Preprocesses text data in a DataFrame and optionally creates visualizations.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the text data.
    text_column (str): The name of the column containing the text to be processed.
    Barplot (bool): If True, creates a bar plot of word frequencies.
    Wordcloud (bool): If True, creates a word cloud visualization.

    Returns:
    pandas.DataFrame: The DataFrame with an additional 'cleaned_text' column.
    """

    def clean_and_tokenize(text):
        text = str(text).lower()
        return [word for word in word_tokenize(text) if word.isalpha() and word not in stop_words]

    def process_text(tokens):
        doc = nlp(' '.join(tokens))
        return [token.lemma_ for token in doc]

    # Apply cleaning and tokenization
    df['tokens'] = df[text_column].apply(clean_and_tokenize)

    # Apply stemming
    df['cleaned_tokens'] = df['tokens'].apply(process_text)

    # Count word frequencies
    word_freq = Counter([word for tokens in df['cleaned_tokens'] for word in tokens])

    # Remove rare words (appearing 5 times or less)
    rare_words = {word for word, count in word_freq.items() if count <= 5000}
    df['cleaned_text'] = df['cleaned_tokens'].apply(lambda tokens: ' '.join(word for word in tokens if word not in rare_words))

    # Clean up intermediate columns
    df = df.drop(['tokens', 'cleaned_tokens'], axis=1)

    if Barplot or Wordcloud:
        # Calculate word frequencies for visualization
        vis_word_freq = Counter([word for text in df['cleaned_text'] for word in text.split()])

    if Barplot:
        # Create and display bar plot
        top_words = dict(sorted(vis_word_freq.items(), key=lambda x: x[1], reverse=True)[:20])
        plt.figure(figsize=(12, 6))
        plt.bar(top_words.keys(), top_words.values())
        plt.title("Top 20 Word Frequencies")
        plt.xlabel("Words")
        plt.ylabel("Frequency")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    if Wordcloud:
        # Create and display word cloud
        wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate_from_frequencies(vis_word_freq)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title("Word Cloud")
        plt.tight_layout()
        plt.show()

    return df

# Usage
# df = pd.read_csv("/kaggle/input/wikimedia/wiki_data.csv", index_col=0)
# processed_df = wiki_preprocess(df, 'text', Barplot=True, Wordcloud=True)