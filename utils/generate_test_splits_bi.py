import random
import os
from collections import defaultdict


def generate_hold_out_split (dataset, training = 0.8, base_dir="splits",biClass=False):
    r = random.Random()
    r.seed(1489215)

    biClass_dir = "bi_" if biClass else ""

    if biClass:
        related_body = set()
        for s in dataset.stances:
            if s['Stance_biClass'] == 1:
                related_body.add(s['Body ID'])
        article_ids = list(related_body)
    else :
        article_ids = list(dataset.articles.keys())  # get a list of article ids
    r.shuffle(article_ids)  # and shuffle that list


    training_ids = article_ids[:int(training * len(article_ids))]
    hold_out_ids = article_ids[int(training * len(article_ids)):]

    # write the split body ids out to files for future use
    with open(base_dir+ "/"+ biClass_dir + "training_ids.txt", "w+") as f:
        f.write("\n".join([str(id) for id in training_ids]))

    with open(base_dir+ "/"+ biClass_dir + "hold_out_ids.txt", "w+") as f:
        f.write("\n".join([str(id) for id in hold_out_ids]))



def read_ids(file,base):
    ids = []
    with open(base+"/"+file,"r") as f:
        for line in f:
           ids.append(int(line))
        return ids


def kfold_split(dataset, training = 0.8, n_folds = 10, base_dir="splits",biClass=False):

    biClass_dir = "bi_" if biClass else ""
    if not (os.path.exists(base_dir+ "/" + biClass_dir + "training_ids.txt")
            and os.path.exists(base_dir+ "/" + biClass_dir + "hold_out_ids.txt")):
        generate_hold_out_split(dataset,training,base_dir,biClass)

    training_ids = read_ids(biClass_dir + "training_ids.txt", base_dir)
    hold_out_ids = read_ids(biClass_dir + "hold_out_ids.txt", base_dir)

    folds = []
    for k in range(n_folds):
        folds.append(training_ids[int(k*len(training_ids)/n_folds):int((k+1)*len(training_ids)/n_folds)])

    return folds,hold_out_ids


def get_stances_for_folds(dataset,folds,hold_out,only_related = False):
    stances_folds = defaultdict(list)
    stances_hold_out = []
    for stance in dataset.stances:
        if stance['Body ID'] in hold_out and ((only_related and stance['Stance_biClass'] == 1) or (not only_related)):
            stances_hold_out.append(stance)
        else:
            fold_id = 0
            for fold in folds:
                if stance['Body ID'] in fold and ((only_related and stance['Stance_biClass'] == 1) or (not only_related)):
                    stances_folds[fold_id].append(stance)
                fold_id += 1

    return stances_folds,stances_hold_out
