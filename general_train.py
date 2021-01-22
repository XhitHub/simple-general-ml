import numpy as np
import sklearn
from sklearn import tree

# def eval_decision_tree_children(children, res):
#   i = 0
#   while i < len(children):
#     if (children[i] == -1):
#       # is leaf

#     i += 1
#   for item in children:
#     if (item ==)

def sort_tree_eval(resItem):
  return resItem['impurity']
def eval_decision_tree(tree):
  res = []
  cl = tree.children_left
  cr = tree.children_right
  # for children in [cl, cr]:
  i = 0
  # can only look at cl. if node i is leaf, 
  while i < len(cl):
    if (cl[i] == -1):
      # is leaf
      resItem = {
        'feature': tree.feature[i],
        'n_node_samples': tree.n_node_samples[i],
        'impurity': tree.impurity[i]
      }
      res.append(resItem)
    i += 1
  # sort res
  res.sort(key=sort_tree_eval)
  return res

# grid search / RandomizedSearchCV
# cross validate