import sys
import numpy as np

# from sklearn.ensemble import GradientBoostingClassifier
from feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats
from feature_engineering import word_overlap_features
from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds
from utils.score import report_score, LABELS, score_submission

from utils.system import parse_params, check_version

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier


def generate_features(stances,dataset,name):
    h, b, y = [],[],[]

    for stance in stances:
        y.append(LABELS.index(stance['Stance']))
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])

    X_overlap = gen_or_load_feats(word_overlap_features, h, b, "features/overlap."+name+".npy")
    X_refuting = gen_or_load_feats(refuting_features, h, b, "features/refuting."+name+".npy")
    X_polarity = gen_or_load_feats(polarity_features, h, b, "features/polarity."+name+".npy")
    X_hand = gen_or_load_feats(hand_features, h, b, "features/hand."+name+".npy")

    X = np.c_[X_hand, X_polarity, X_refuting, X_overlap]
    return X,y

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


    best_score = [0,0]
    best_fold = [None, None]


    
    
    print("====> Start Classifier for each fold ..")
    # Classifier for each fold
    for fold in fold_stances:
        ids = list(range(len(folds)))
        del ids[fold]

        X_train = np.vstack(tuple([Xs[i] for i in ids]))
        y_train = np.hstack(tuple([ys[i] for i in ids]))

        X_test = Xs[fold]
        y_test = ys[fold]

        # clf = GradientBoostingClassifier(n_estimators=200, random_state=14128, verbose=True)
        # clf.fit(X_train, y_train)

        rfo_clf = RandomForestClassifier()
        xgb_clf = xgb.XGBClassifier()
        clfs = [rfo_clf,xgb_clf]


        actual = [LABELS[int(a)] for a in y_test]
        max_fold_score, _ = score_submission(actual, actual)
        for i in range(len(clfs)):
            clfs[i].fit(X_train, y_train)

            predicted = [LABELS[int(a)] for a in clfs[i].predict(X_test)]
            fold_score, _ = score_submission(actual, predicted)
            score = fold_score/max_fold_score

            print("Score for fold " + str(i) + " : "+ str(fold) + " was - " + str(score))
            if score > best_score[i]:
                best_score[i] = score
                best_fold[i] = clfs[i]
        # break

    
    #Run on Holdout set and report the final score on the holdout set
    # predict_clfs = [[int(a) for a in best_clf.predict(X_holdout)] for best_clf in best_fold ]
    # predict_clfs = np.asarray(predict_clfs).T
    # predict_clfs = [np.bincount(predict_clf) for predict_clf in predict_clfs]
    # predicted = [LABELS[int(np.argmax(vote))] for vote in predict_clfs]
    # actual = [LABELS[int(a)] for a in y_holdout]

    predict_clfs = [[a for a in best_clf.predict_proba(X_holdout)] for best_clf in best_fold ]
    predict_clfs = np.mean(predict_clfs, axis = 0)
    predicted = [LABELS[int(np.argmax(vote))] for vote in predict_clfs]
    actual = [LABELS[int(a)] for a in y_holdout]

    print("Scores on the dev set")
    report_score(actual,predicted)
    print("")
    print("")

    #Run on competition dataset
    # predict_clfs = [[int(a) for a in best_clf.predict(X_competition)] for best_clf in best_fold ]
    # predict_clfs = np.asarray(predict_clfs).T
    # predict_clfs = [np.bincount(predict_clf) for predict_clf in predict_clfs]
    # predicted = [LABELS[int(np.argmax(vote))] for vote in predict_clfs]
    # actual = [LABELS[int(a)] for a in y_competition]
    predict_clfs = [[a for a in best_clf.predict_proba(X_competition)] for best_clf in best_fold ]
    predict_clfs = np.mean(predict_clfs, axis = 0)
    predicted = [LABELS[int(np.argmax(vote))] for vote in predict_clfs]
    actual = [LABELS[int(a)] for a in y_competition]

    print("Scores on the test set")
    report_score(actual,predicted)
