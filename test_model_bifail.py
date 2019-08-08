import sys
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from feature_engineering import refuting_features, polarity_features, hand_features, sentiment_analyzer, gen_or_load_feats
from feature_engineering import word_overlap_features
from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds
from utils.score import report_score, report_score_biClass, LABELS, LABELS_RELATED, score_submission
from utils.system import parse_params, check_version

import xgboost as xgb
import matplotlib.pyplot as plt

def generate_features(stances,dataset,name,only_related=False):
    h, b, y, y_bi = [],[],[],[]

    related_dir = "re_" if only_related else ""
    for stance in stances:
        y_bi.append(stance['Stance_biClass'])
        y.append(LABELS.index(stance['Stance']))
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])

    X_overlap = gen_or_load_feats(word_overlap_features, h, b, "features/" + related_dir + "overlap."+name+".npy")
    X_refuting = gen_or_load_feats(refuting_features, h, b, "features/" + related_dir + "refuting."+name+".npy")
    X_polarity = gen_or_load_feats(polarity_features, h, b, "features/" + related_dir + "polarity."+name+".npy")
    X_sentiment = gen_or_load_feats(sentiment_analyzer, h, b, "features/" + related_dir + "sentiment."+name+".npy")
    X_hand = gen_or_load_feats(hand_features, h, b, "features/" + related_dir + "hand."+name+".npy")

    X = np.c_[X_hand, X_sentiment, X_polarity, X_refuting, X_overlap]
    return X,y,y_bi


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
            print(str(idx) + "\t" + _name + "\t",importances[indices[idx]])
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
    X_competition, y_competition, y_competition_bi = generate_features(competition_dataset.stances, competition_dataset, "competition")

    Xs = dict()
    ys = dict()
    ys_bi = dict()

    # Load/Precompute all features now
    X_holdout,y_holdout,y_holdout_bi = generate_features(hold_out_stances,d,"holdout")
    for fold in fold_stances:
        Xs[fold],ys[fold],ys_bi[fold] = generate_features(fold_stances[fold],d,str(fold))


    best_score = 0
    best_fold = None


    # Classifier for each fold
    for fold in fold_stances:
        ids = list(range(len(folds)))
        del ids[fold]

        X_train = np.vstack(tuple([Xs[i] for i in ids]))
        y_train = np.hstack(tuple([ys_bi[i] for i in ids]))

        X_test = Xs[fold]
        y_test = ys_bi[fold]

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

    # step2
    related_folds, related_hold_out = kfold_split(d,n_folds=10,biClass=True)
    related_fold_stances, related_hold_out_stances = get_stances_for_folds(d,related_folds,related_hold_out)
    related_Xs = dict()
    related_ys = dict()
    related_ys_bi = dict()

    # Load/Precompute all features now
    related_X_holdout,related_y_holdout,related_y_holdout_bi = generate_features(hold_out_stances,d,"holdout",only_related=True)
    for fold in related_fold_stances:
        related_Xs[fold],related_ys[fold],related_ys_bi[fold] = generate_features(related_fold_stances[fold],d,str(fold),only_related=True)


    related_best_score = 0
    related_best_fold = None


    # Classifier for each fold
    for fold in related_fold_stances:
        ids = list(range(len(folds)))
        del ids[fold]

        X_train = np.vstack(tuple([related_Xs[i] for i in ids]))
        y_train = np.hstack(tuple([related_ys[i] for i in ids]))

        X_test = related_Xs[fold]
        y_test = related_ys[fold]

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
        if score > related_best_score:
            related_best_score = score
            related_best_fold = clf


    
    #Run on Holdout set and report the final score on the holdout set

    

    predicted = [LABELS_RELATED[int(a)] for a in best_fold.predict(X_holdout)]
    actual = [LABELS_RELATED[int(a)] for a in y_holdout_bi]
    related_X_holdout = []
    related_actual = []
    unrelated_actual = []
    r_pre = []
    u_pre = []
    un_count = 0
    for i in range(len(predicted)):
        if predicted[i] == "related":
            related_X_holdout.append(X_holdout[i])
            related_actual.append(LABELS[int(y_holdout[i])])
            # related_actual.append(actual[i])
        else :
            unrelated_actual.append(LABELS[int(y_holdout[i])])
            # unrelated_actual.append(actual[i])
            un_count+= 1
    
    predicted = [LABELS[int(a)] for a in related_best_fold.predict(related_X_holdout)] + ["unrelated"] * un_count
    actual = related_actual + unrelated_actual

    print("Scores on the dev set")
    report_score(actual,predicted)
    print("")
    print("")





    #Run on competition dataset
    predicted = [LABELS_RELATED[int(a)] for a in best_fold.predict(X_competition)]
    actual = [LABELS_RELATED[int(a)] for a in y_competition_bi]

    print("Scores on the test set")
    report_score_biClass(actual,predicted)

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

    name_features = hand_feature_name + sentiment_feature_name + polarity_feature_name + refuting_feature_name + ['overlap']
    
    plot_feature_importance(best_fold.feature_importances_,name_features,show=True)