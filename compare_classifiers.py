import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from apply_rules import classify_text

def load_decision_tree_classifier():
    """Load the pre-trained decision tree classifier and vectorizer"""
    try:
        clf = joblib.load('clasificador_cvo_exp.pkl')
        vectorizer = joblib.load('vectorizer_cvo_exp.pkl')
        categories = joblib.load('categorias_cvo_exp.pkl')
        return clf, vectorizer, categories
    except FileNotFoundError as e:
        print(f"Error loading classifier files: {e}")
        return None, None, None

def predict_with_decision_tree(text, clf, vectorizer):
    """Predict class using decision tree classifier"""
    if clf is None or vectorizer is None:
        return "UNKNOWN", 0.0
    
    # Vectorize the text
    text_vec = vectorizer.transform([text])
    
    # Predict
    prediction = clf.predict(text_vec)
    probabilities = clf.predict_proba(text_vec)
    
    # Get confidence (max probability)
    confidence = np.max(probabilities) if probabilities.size > 0 else 0.0
    
    return prediction[0], confidence

def predict_with_rules(text):
    """Predict class using rules-based classifier"""
    result = classify_text(text)
    return result['class'], result['confidence']

def compare_classifiers():
    """Compare both classifiers on the test data"""
    # Load the data
    df = pd.read_csv('textos_procesados.csv')
    df["asunto preprocesado"] = df["asunto preprocesado"].fillna('').astype(str)
    
    # Load decision tree classifier
    clf, vectorizer, categories = load_decision_tree_classifier()
    if clf is None:
        print("Decision tree classifier not found. Please run dectree_clasificacion.py first.")
        return
    
    print(f"Loaded decision tree classifier with {len(categories)} categories")
    print(f"Testing on {len(df)} samples")
    
    # Initialize results
    results = []
    
    # Test each sample
    for idx, row in df.iterrows():
        text = row['asunto preprocesado']
        true_class = row['procedimiento']
        
        # Predict with decision tree
        dt_class, dt_confidence = predict_with_decision_tree(text, clf, vectorizer)
        
        # Predict with rules
        rules_class, rules_confidence = predict_with_rules(text)
        
        results.append({
            'text': text,
            'true_class': true_class,
            'dt_predicted': dt_class,
            'dt_confidence': dt_confidence,
            'rules_predicted': rules_class,
            'rules_confidence': rules_confidence,
            'dt_correct': dt_class == true_class,
            'rules_correct': rules_class == true_class
        })
        
        # Print progress
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(df)} samples")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate metrics
    y_true = results_df['true_class']
    y_dt = results_df['dt_predicted']
    y_rules = results_df['rules_predicted']
    
    print("\n" + "="*60)
    print("DECISION TREE CLASSIFIER RESULTS")
    print("="*60)
    print(f"Accuracy: {accuracy_score(y_true, y_dt):.4f}")
    print(f"Precision (weighted): {precision_score(y_true, y_dt, average='weighted', zero_division=0):.4f}")
    print(f"Recall (weighted): {recall_score(y_true, y_dt, average='weighted', zero_division=0):.4f}")
    print(f"F1-score (weighted): {f1_score(y_true, y_dt, average='weighted', zero_division=0):.4f}")
    
    print("\n" + "="*60)
    print("RULES-BASED CLASSIFIER RESULTS")
    print("="*60)
    print(f"Accuracy: {accuracy_score(y_true, y_rules):.4f}")
    print(f"Precision (weighted): {precision_score(y_true, y_rules, average='weighted', zero_division=0):.4f}")
    print(f"Recall (weighted): {recall_score(y_true, y_rules, average='weighted', zero_division=0):.4f}")
    print(f"F1-score (weighted): {f1_score(y_true, y_rules, average='weighted', zero_division=0):.4f}")
    
    print("\n" + "="*60)
    print("DETAILED CLASSIFICATION REPORT - DECISION TREE")
    print("="*60)
    print(classification_report(y_true, y_dt, zero_division=0))
    
    print("\n" + "="*60)
    print("DETAILED CLASSIFICATION REPORT - RULES-BASED")
    print("="*60)
    print(classification_report(y_true, y_rules, zero_division=0))
    
    # Save detailed results
    results_df.to_csv('classifier_comparison_results.csv', index=False, encoding='utf-8')
    print(f"\nDetailed results saved to 'classifier_comparison_results.csv'")
    
    # Calculate confusion matrix (simplified)
    confusion_data = []
    for true_class in categories:
        for pred_class in categories:
            count = len(results_df[(results_df['true_class'] == true_class) & (results_df['dt_predicted'] == pred_class)])
            if count > 0:
                confusion_data.append({
                    'true_class': true_class,
                    'predicted_class': pred_class,
                    'count': count,
                    'classifier': 'decision_tree'
                })
    
    for true_class in categories:
        for pred_class in categories:
            count = len(results_df[(results_df['true_class'] == true_class) & (results_df['rules_predicted'] == pred_class)])
            if count > 0:
                confusion_data.append({
                    'true_class': true_class,
                    'predicted_class': pred_class,
                    'count': count,
                    'classifier': 'rules_based'
                })
    
    confusion_df = pd.DataFrame(confusion_data)
    confusion_df.to_csv('classifier_confusion_data.csv', index=False, encoding='utf-8')
    print(f"Confusion data saved to 'classifier_confusion_data.csv'")

if __name__ == "__main__":
    compare_classifiers()
