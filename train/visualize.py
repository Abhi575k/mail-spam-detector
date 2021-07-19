import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

def show_frequent_words(words):
    cloud_stopwords = set(STOPWORDS)
    print("Total words {}".format(len(words)))
    wordcloud = WordCloud(
        width=1000,
        height=1000,
        background_color="white",
        stopwords=cloud_stopwords,
        min_font_size=10,
    ).generate(words)
    plt.figure(figsize=(15, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()