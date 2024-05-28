from sklearn.feature_extraction.text import TfidfVectorizer


def do_tfidf(token):
    if token !=[''] and len(token) >20:
        tfidf = TfidfVectorizer()
        words = tfidf.fit_transform(token)
        sentence = " ".join(tfidf.get_feature_names_out())
        return sentence
    else:
        return ['']
