from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds
dataset = DataSet()

# print("Total stances: " + str(len(dataset.stances)))
# print("Total article bodies: " + str(len(dataset.articles)))



print(dataset.stances[0])



folds,hold_out = kfold_split(dataset,n_folds=10)
fold_stances, hold_out_stances = get_stances_for_folds(dataset,folds,hold_out)

print()
print()

print("len hold_out_stances",len(hold_out_stances))
print("len hold_out",len(hold_out))

print()
print()

for i in range(len(folds)):
    print(i)
    print(len(folds[i]))
    print(len(fold_stances[i]))