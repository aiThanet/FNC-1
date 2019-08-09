from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from utils.dataset import DataSet
from sklearn import feature_extraction
from tqdm import tqdm
import re
import nltk


_wnl = nltk.WordNetLemmatizer()


def preprocessing(text,lemma=True,toLower=True,punctuationRemove=True,stopWordRemove=True,join=True):
    if punctuationRemove and lemma and stopWordRemove and toLower:
        text = re.findall(r'\w+', text, flags=re.UNICODE)
        W = []
        for w in text:
            w = _wnl.lemmatize(w.lower())
            if w not in feature_extraction.text.ENGLISH_STOP_WORDS:
                W.append(w)
        return " ".join(W) if join else W
    if punctuationRemove:
        text = " ".join(re.findall(r'\w+', text, flags=re.UNICODE))
    if toLower:
        text = text.lower()
    if lemma and stopWordRemove:
        W = []
        for w in nltk.word_tokenize(text):
            w = _wnl.lemmatize(w)
            if w not in feature_extraction.text.ENGLISH_STOP_WORDS:
                W.append(w)
        return " ".join(W)
    if lemma:
        text = " ".join([_wnl.lemmatize(t) for t in nltk.word_tokenize(text)])
    if stopWordRemove:
        text = " ".join([w for w in nltk.word_tokenize(text) if w not in feature_extraction.text.ENGLISH_STOP_WORDS])

    return text

def doc2vecModelGenerator(lemma=True,toLower=True,punctuationRemove=True,stopWordRemove=True):
    d = DataSet()
    headlines, bodies= [], []
    print("Preprocessing data...")
    for i,stance in tqdm(enumerate(d.stances)):
        _h = stance['Headline']
        _h = preprocessing(_h,lemma=lemma,toLower=toLower,punctuationRemove=punctuationRemove,stopWordRemove=stopWordRemove)
        headlines.append(_h)
        _b = d.articles[stance['Body ID']]
        _b = preprocessing(_b,lemma=lemma,toLower=toLower,punctuationRemove=punctuationRemove,stopWordRemove=stopWordRemove)
        bodies.append(_b)

    print("Tagging data...")
    h_tagged_data = [TaggedDocument(words=nltk.word_tokenize(_d), tags=[str(i)]) for i, _d in tqdm(enumerate(headlines))]
    b_tagged_data = [TaggedDocument(words=nltk.word_tokenize(_d), tags=[str(i)]) for i, _d in tqdm(enumerate(bodies))]
    h_model = Doc2Vec(vector_size=5,alpha=0.025,min_alpha=0.00025,min_count=1,dm =1,dm_concat=1,epochs=100)
    b_model = Doc2Vec(vector_size=20,alpha=0.025,min_alpha=0.00025,min_count=1,dm =1,dm_concat=1,epochs=100)
    print("doc2vec Model Vocab Building...")
    h_model.build_vocab(h_tagged_data,progress_per=5000)
    b_model.build_vocab(b_tagged_data,progress_per=5000)

    print("doc2vec Model Vocab Training...")
    h_model.train(h_tagged_data,total_examples=h_model.corpus_count,epochs=h_model.epochs)
    b_model.train(b_tagged_data,total_examples=b_model.corpus_count,epochs=b_model.epochs)
    h_model.save("models/h_d2v.model")
    b_model.save("models/b_d2v.model")


if __name__ == "__main__":
    doc2vecModelGenerator()

