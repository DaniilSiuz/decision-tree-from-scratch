# A decision tree, built from scratch

No sklearn.tree anywhere in this repository. The entropy calculation, the information gain, the best-split search and the recursive tree construction are all written directly in NumPy, then tested on wine quality prediction.

**Stack**: Python · NumPy · pandas

## Why build one by hand

Calling DecisionTreeClassifier() tells you nothing about why a tree splits where it does, or what "overfitting" looks like mechanically. Writing it out forces every decision into the open: what impurity measure to use, how to search thresholds, when to stop recursing, and what a leaf predicts when its samples disagree.

## What's implemented
Function	What it does
calc_entropy	Binary Shannon entropy of a label array
calc_information_gain	Parent entropy minus the size-weighted entropy of the two children
best_split	Exhaustive search over every feature and every observed threshold
train_decision_tree	Recursive construction, returning the tree as nested dictionaries
predict_sample / predict_tree	Traversal for one sample and for a matrix

Recursion stops on any of three conditions: the node is pure, the maximum depth is reached, or no candidate split yields positive information gain. Impure leaves predict the majority class.

## The data

UCI red wine quality — 1,599 samples, 11 physicochemical features. The 0–10 quality score is binarised into good (>= 6) and bad (< 6), turning an ordinal problem into binary classification.

## Results

Split 60 / 20 / 20 into training, validation and test.

Stopping depth was chosen on the validation set, sweeping 1 to 7. The test set was used exactly once, after the depth was fixed and the tree retrained on training plus validation.

	
| Selected depth | **4** |
| Held-out test accuracy | **71.6%** (229 / 320) |
| Macro F1 | 0.714 |
| Majority-class baseline | 50.3% |

Accuracy rises with depth, plateaus, then declines as the tree begins to fit noise rather than signal — the standard bias-variance picture, visible here in a tree whose every split you can inspect.

## Limitations
Exhaustive threshold search is O(features x unique values) per node, so this is fine on 1,599 rows and hopeless on a large dataset. A real implementation sorts once per feature and scans.
Binary classification only; no multi-class support, no regression trees.
No pruning. Depth is the only complexity control, where scikit-learn offers minimum samples per leaf, cost-complexity pruning and more.
Binarising quality at 6 discards ordinal information and creates an arbitrary threshold.

## Running it

analysis.ipynb, top to bottom. Requires NumPy, pandas and scikit-learn (used only for train_test_split, not for the tree).
