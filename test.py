from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds
# dataset = DataSet()

import numpy as np

# # print("Total stances: " + str(len(dataset.stances)))
# # print("Total article bodies: " + str(len(dataset.articles)))
t = np.load("./features/hand.0.npy")

# for i in range(10):
#     print((t[i][6]))

# from sklearn import feature_extraction

# # for i in feature_extraction.text.ENGLISH_STOP_WORDS:
# #     print(i)

# # print(dataset.stances[0])

# print(len(feature_extraction.text.ENGLISH_STOP_WORDS))

# folds,hold_out = kfold_split(dataset,n_folds=10)
# fold_stances, hold_out_stances = get_stances_for_folds(dataset,folds,hold_out)

# print()
# print()

# print("len hold_out_stances",len(hold_out_stances))
# print("len hold_out",len(hold_out))

# print()
# print()

# for i in range(len(folds)):
#     print(i)
#     print(len(folds[i]))
#     print(len(fold_stances[i]))

# import xgboost as xgb
# clf = xgb.XGBClassifier()

# clf2 = xgb.XGBClassifier(learning_rate =0.1, n_estimators=1000,max_depth=5,min_child_weight=1,gamma=0,subsample=0.8,colsample_bytree=0.8,objective= "multi:softmax",num_class=4,scale_pos_weight=1,seed=12345)
# print(clf)

# print(clf2)

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re

sid = SentimentIntensityAnalyzer()


sentences = '''"A bereaved Afghan mother took revenge on the Taliban after watching them kill her son in an ambush. Reza Gul killed 25 Taliban fighters and injured five others in a seven-hour gunbattle in Farah province.

Gul, who was joined by her daughter and daughter in-law, engaged the Taliban using AK-47s and grenades, despitenever before having used a weapon.

The embattled mother told Tolo news, a 24-hour Afghan news broadcaster, she was awakened by shots early Tuesday. After seeing that her son had been killed, Gul and the other two women fought back.

“I couldn't stop myself and picked up a weapon,” Gul told Tolo News. “I went to the check post and began shooting back.”

Seema, her daughter-in-law, added: “The fighting was intensified when we reached the battlefield along with light and heavy weapons. We were committed to fight until the last bullet.”

Gul said that the battlefield was covered in Talib fighters after the deadly exchange ended.

While the Taliban have not publicly commented on the incident, the Afghan government labeled it a symbol of a public uprising against the Taliban.

Taliban and other groups have regained large swathes of the country as U.S. and NATO forces slowly pull out troops after 14 years of war. The Taliban have targeted government and foreign infrastructure as the group attempts to claw back power it lost in 2001.

While the Taliban have made key gains in rural regions, members continue to employ suicide bomber tactics in well protected towns and cities. Earlier this week, 50 people were killed after a suicide bomber detonated a vest during a volleyball competition in Yahyakahil, Paktika province.

That particular attack prompted President Ashraf Ghani to order a complete overview of the country’s defense forces and to rethink the ban on nighttime raids, which were outlawed by his predecessor, Hamid Karzai."
'''

def clean(s):
    # Cleans a string: Lowercasing, trimming, removing non-alphanumeric

    return " ".join(re.findall(r'\w+', s, flags=re.UNICODE)).lower()

clean_sen = clean(sentences)

# print(clean_sen)

ss = sid.polarity_scores(clean_sen)

lis = ['compound','neg','neu','pos']
sss = [ss[l] for l in lis]
print(ss)
print(sss)