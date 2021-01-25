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

SAMPLES_THRESHOLD = 20

def sort_tree_eval(resItem):
  return resItem['n_node_samples']
  # return resItem['impurity']
def filter_tree_eval(resItem):
  return resItem['n_node_samples'] >= SAMPLES_THRESHOLD
  # return resItem['impurity']


# ref: https://stackoverflow.com/questions/32506951/how-to-explore-a-decision-tree-built-using-scikit-learn
def eval_decision_tree(tree):
  res = []
  cl = tree.children_left
  cr = tree.children_right
  # for children in [cl, cr]:
  i = 0
  # can only look at cl. if node i is leaf, 
  while i < len(cl):
    # if (True or cl[i] == -1):
    if (cl[i] == -1):
      # is leaf
      resItem = {
        'i': i,
        'feature': tree.feature[i],
        'n_node_samples': tree.n_node_samples[i],
        'impurity': tree.impurity[i]
      }
      if (resItem['n_node_samples'] >= SAMPLES_THRESHOLD):
        res.append(resItem)
    # if (cr[i] == -1):
    #   # is leaf
    #   resItem = {
    #     'i': i,
    #     'feature': tree.feature[i],
    #     'n_node_samples': tree.n_node_samples[i],
    #     'impurity': tree.impurity[i]
    #   }
    #   if (resItem['n_node_samples'] >= SAMPLES_THRESHOLD):
    #     res.append(resItem)
    i += 1
  # sort res
  res.sort(key=sort_tree_eval)
  # resFiltered = filter(filter_tree_eval, res)

  return res

# grid search / RandomizedSearchCV
# cross validate