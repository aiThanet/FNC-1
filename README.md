# FNC-1 COMP9417 project

Fake New Challenge : [FakeNewsChallenge.org](http://fakenewschallenge.org).

This project start from provided baseline on [github](https://github.com/FakeNewsChallenge/fnc-1-baseline).

## Requirements

        python >= 3.7.0 (tested with 3.7.2)

## Installation

1.  Install required python packages.

        pip install -r requirements.txt --upgrade

2.  Parts of the Natural Language Toolkit (NLTK) might need to be installed manually.

        python3 -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"

3.  All features have been generated. If you want to reproduce it, delete all files in features directory. Keep the features, you can skip to 6.

4.  To generate name entity feature, you need to run CoreNLP server version 3.9.2: Download [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/history.html), extract anywhere and execute following command in corenlp directory (It takes about 5 hours on dev enviroment to generate):

        java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9020

5.  To generate doc2vec feature, you need 2 paragraph vector models in models directory name h_d2v.model and b_d2v.model. You can generate from doc2vecModelGenerator.py or download `models.zip` from [drive](https://drive.google.com/drive/folders/1dUrKZuVctHLy1PBvCRV3pZPbE5iFZL7i?usp=sharing).

6.  To run and generate the model (if features does not exist, automatically generate and save in features directory).

        python3 test_2_models.py

7.  `test_model.py` is the old version of project which classify 4 classes by only one model.

## References

- [FakeNewsChallenge.org](http://fakenewschallenge.org).
- [FNC-1 Baseline](https://github.com/FakeNewsChallenge/fnc-1-baseline).
- [Sentiment Analysis](http://www.nltk.org/howto/sentiment.html).
- [CoreNLP Server](https://stanfordnlp.github.io/CoreNLP/index.html).
- [stanford-corenlp](https://github.com/Lynten/stanford-corenlp).
- [DOC2VEC gensim tutorial](https://medium.com/@mishra.thedeepak/doc2vec-simple-implementation-example-df2afbbfbad5).
