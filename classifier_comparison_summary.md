# Classifier Comparison Summary

## Overview
This report compares the performance of two classifiers:
1. **Decision Tree Classifier** - The original trained model
2. **Rules-Based Classifier** - Extracted logical rules from the decision tree

## Dataset
- **Total samples**: 622
- **Number of classes**: 17
- **Data source**: textos_procesados_anonimo.csv

## Performance Metrics

### Decision Tree Classifier
- **Accuracy**: 61.74%
- **Weighted Precision**: 56.97%
- **Weighted Recall**: 61.74%
- **Weighted F1-score**: 55.50%

### Rules-Based Classifier
- **Accuracy**: 30.55%
- **Weighted Precision**: 56.77%
- **Weighted Recall**: 30.55%
- **Weighted F1-score**: 39.08%

## Key Findings

### 1. Accuracy Comparison
- The Decision Tree classifier significantly outperforms the Rules-Based classifier in accuracy (61.74% vs 30.55%)
- This suggests that the rule extraction process may have lost some of the decision tree's predictive power

### 2. Precision Analysis
- Both classifiers have similar weighted precision scores (~57%)
- This indicates that when the Rules-Based classifier makes a prediction, it's about as accurate as the Decision Tree
- However, the Rules-Based classifier makes predictions much less frequently

### 3. Recall Analysis
- The Decision Tree has much higher recall (61.74% vs 30.55%)
- This shows the Rules-Based classifier fails to classify many instances that the Decision Tree can handle

### 4. Class-Specific Performance

**Classes with good performance in both:**
- `COMPL ALQUILER PNC`: Both classifiers perform well (82% recall for Decision Tree, 82% for Rules)
- `NOMINAS`: Decision Tree 41% recall, Rules 40% recall
- `PNC CONTROL ANUAL`: Decision Tree 49% recall, Rules 33% recall

**Classes with poor performance:**
- Many classes (--ND--, CONCURRENCIAS, FALLECIDOS, etc.) have 0% recall in both classifiers
- This suggests these classes may be underrepresented or difficult to classify

## Limitations of Rules-Based Classifier

1. **Incomplete Rule Coverage**: The extracted rules don't cover all decision paths from the original tree
2. **Sequential Application**: Rules are applied in order, and once a rule matches, no further rules are checked
3. **Missing Conditions**: Some complex conditions from the decision tree may not have been captured in the rules
4. **Confidence Handling**: The rules-based approach uses binary matching rather than probabilistic confidence

## Recommendations

1. **Improve Rule Extraction**: Review the rule extraction process to ensure all decision paths are captured
2. **Rule Optimization**: Consider optimizing the rule order and combining overlapping rules
3. **Hybrid Approach**: Use the rules-based classifier for interpretability and fall back to the decision tree for uncertain cases
4. **Class Balancing**: Address the imbalance in class distribution to improve performance on minority classes

## Files Generated

1. `classifier_comparison_results.csv` - Detailed prediction results for each sample
2. `classifier_confusion_data.csv` - Confusion matrix data for both classifiers

## Conclusion

While the Rules-Based classifier provides interpretable logical rules, it currently underperforms compared to the original Decision Tree classifier. The main advantage of the rules-based approach is interpretability, but this comes at the cost of reduced accuracy and coverage. Further refinement of the rule extraction process could help bridge this performance gap.
