# FNC-1 COMP9417 project

Fake New Challenge : [FakeNewsChallenge.org](http://fakenewschallenge.org).

this project start from baseline that provided by the orgainzatio can be found on [github](https://github.com/FakeNewsChallenge/fnc-1-baseline).

## Requirements

        python >= 3.7.0 (tested with 3.7.2)

## Installation

1.  Install required python packages.

        pip install -r requirements.txt --upgrade

2.  All features have been generated. If you want to reproduce it, delete all files in features directory.

3.  Parts of the Natural Language Toolkit (NLTK) might need to be installed manually.

        python3 -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"

4.  To generate name entity feature, you need to run CoreNLP server version 3.9.2: Download [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/history.html), extract anywhere and execute following command in corenlp directory:

        java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9020

5.  To run and generate the model (if features does not exist, automatically generate and save in features directory).

        python3 test_model.py

## References

- [FakeNewsChallenge.org](http://fakenewschallenge.org).
- [FNC-1 Baseline](https://github.com/FakeNewsChallenge/fnc-1-baseline).
- [Sentiment Analysis](http://www.nltk.org/howto/sentiment.html).
- [CoreNLP Server](https://stanfordnlp.github.io/CoreNLP/index.html).
- [stanford-corenlp](https://github.com/Lynten/stanford-corenlp).
