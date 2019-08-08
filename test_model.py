import sys
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from feature_engineering import refuting_features, polarity_features, hand_features, sentiment_analyzer, gen_or_load_feats
from feature_engineering import word_overlap_features, name_entity_similarity
from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds
from utils.score import report_score, LABELS, score_submission
from utils.system import parse_params, check_version

import xgboost as xgb
import matplotlib.pyplot as plt

def generate_features(stances,dataset,name):
    h, b, y = [],[],[]

    for stance in stances:
        y.append(LABELS.index(stance['Stance']))
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])

    X_overlap = gen_or_load_feats(word_overlap_features, h, b, "features/overlap."+name+".npy")
    X_refuting = gen_or_load_feats(refuting_features, h, b, "features/refuting."+name+".npy")
    X_polarity = gen_or_load_feats(polarity_features, h, b, "features/polarity."+name+".npy")
    X_sentiment = gen_or_load_feats(sentiment_analyzer, h, b, "features/sentiment."+name+".npy")
    X_hand = gen_or_load_feats(hand_features, h, b, "features/hand."+name+".npy")
    X_ner = gen_or_load_feats(name_entity_similarity, h, b, "features/ner."+name+".npy")
    X = np.c_[X_hand, X_sentiment, X_polarity, X_refuting, X_overlap, X_ner]
    return X,y


def plot_feature_importance(importances,feature_name, show=False):
    
    indices = np.argsort(importances)[::-1]
    sorted_feature_names = [feature_name[i] for i in indices]

    plt.figure()
    plt.title("Feature Importance")
    plt.bar(range(len(feature_name)), importances[indices])
    plt.xticks(range(len(name_features)), sorted_feature_names, rotation=90)
    plt.show()

    if(show):
        idx = 0
        for _name in sorted_feature_names:
            print(str(idx) + "\t" + _name + "\t", importances[indices[idx]])
            idx = idx+1

if __name__ == "__main__":
    check_version()
    parse_params()

    #Load the training dataset and generate folds
    d = DataSet()
    folds,hold_out = kfold_split(d,n_folds=10)
    fold_stances, hold_out_stances = get_stances_for_folds(d,folds,hold_out)

    # Load the competition dataset
    competition_dataset = DataSet("competition_test")
    X_competition, y_competition = generate_features(competition_dataset.stances, competition_dataset, "competition")

    Xs = dict()
    ys = dict()

    # Load/Precompute all features now
    X_holdout,y_holdout = generate_features(hold_out_stances,d,"holdout")
    for fold in fold_stances:
        Xs[fold],ys[fold] = generate_features(fold_stances[fold],d,str(fold))


    best_score = 0
    best_fold = None


    # Classifier for each fold
    for fold in fold_stances:
        ids = list(range(len(folds)))
        del ids[fold]

        X_train = np.vstack(tuple([Xs[i] for i in ids]))
        y_train = np.hstack(tuple([ys[i] for i in ids]))

        X_test = Xs[fold]
        y_test = ys[fold]

        # clf = GradientBoostingClassifier(n_estimators=200, random_state=14128, verbose=True)
        # clf = xgb.XGBClassifier(objective= "multi:softmax",num_class=4,seed=12345)
        clf = xgb.XGBClassifier(seed=12345)
        clf.fit(X_train, y_train)

        predicted = [LABELS[int(a)] for a in clf.predict(X_test)]
        actual = [LABELS[int(a)] for a in y_test]

        fold_score, _ = score_submission(actual, predicted)
        max_fold_score, _ = score_submission(actual, actual)

        score = fold_score/max_fold_score

        print("Score for fold "+ str(fold) + " was - " + str(score))
        if score > best_score:
            best_score = score
            best_fold = clf



    #Run on Holdout set and report the final score on the holdout set
    predicted = [LABELS[int(a)] for a in best_fold.predict(X_holdout)]
    actual = [LABELS[int(a)] for a in y_holdout]

    

    print("Scores on the dev set")
    report_score(actual,predicted)
    print("")
    print("")

    #Run on competition dataset
    predicted = [LABELS[int(a)] for a in best_fold.predict(X_competition)]
    actual = [LABELS[int(a)] for a in y_competition]

    print("Scores on the test set")
    report_score(actual,predicted)

    _refuting_words = ['fake','fraud','hoax','false','deny', 'denies','not','despite','nope','doubt', 'doubts','bogus','debunk','pranks','retract']

    bin_feature_name = ['bin_C','bin_C_E','bin_C_S','bin_C_E_S']
    chargarm_size = [2,4,8,16]
    chargarm_feature_name = []
    for i in chargarm_size:
        chargarm_feature_name.append(str(i) + "_cgram_H")
        chargarm_feature_name.append(str(i) + "_cgram_EH")
        chargarm_feature_name.append(str(i) + "_cgram_FH")
    
    ngarm_size = [2,3,4,5,6]
    ngarm_feature_name = []
    for i in ngarm_size:
        ngarm_feature_name.append(str(i) + "_gram_H")
        ngarm_feature_name.append(str(i) + "_gram_E_H")

    hand_feature_name = bin_feature_name + chargarm_feature_name + ngarm_feature_name
    refuting_feature_name = [ "Refuse_"+refute_word for refute_word in _refuting_words]
    polarity_feature_name = ['Polar_HL','Polar_BD']

    
    sentiment_list = ['com','neg','neu','pos']
    sentiment_headline_feature_name = ['hl_' + i for i in sentiment_list]
    sentiment_body_feature_name = ['bd_' + i for i in sentiment_list]
    sentiment_feature_name = sentiment_headline_feature_name + sentiment_body_feature_name

    ner_feature_name = ['sim_person','diff_person','sim_location','diff_location','sim_organization','diff_organization']
    # ner_feature_name = ['sim_person','sim_location','sim_organization']
    
    name_features = hand_feature_name + sentiment_feature_name + polarity_feature_name + refuting_feature_name + ['overlap'] + ner_feature_name
    
    plot_feature_importance(best_fold.feature_importances_,name_features,show=True)