import joblib
import numpy as np

def extract_decision_tree_rules():
    # Load the trained model and vectorizer
    clf = joblib.load('clasificador_cvo_exp.pkl')
    vectorizer = joblib.load('vectorizer_cvo_exp.pkl')
    categorias = joblib.load('categorias_cvo_exp.pkl')
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # Extract tree structure
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    value = clf.tree_.value
    
    # Function to extract rules from a node
    def extract_rules_from_node(node_id, current_rule=[]):
        if children_left[node_id] == children_right[node_id]:  # Leaf node
            class_probabilities = value[node_id][0]
            predicted_class = np.argmax(class_probabilities)
            confidence = class_probabilities[predicted_class] / np.sum(class_probabilities)
            
            if confidence >= 0.5:  # Only include rules with confidence > 50%
                rule_conditions = " AND ".join(current_rule)
                return [(rule_conditions, categorias[predicted_class], confidence)]
            else:
                return []
        
        # Internal node - extract feature and threshold
        feature_name = feature_names[feature[node_id]]
        threshold_value = threshold[node_id]
        
        # Left child rule (feature <= threshold)
        left_rule = current_rule + [f"'{feature_name}' <= {threshold_value:.4f}"]
        left_rules = extract_rules_from_node(children_left[node_id], left_rule)
        
        # Right child rule (feature > threshold)
        right_rule = current_rule + [f"'{feature_name}' > {threshold_value:.4f}"]
        right_rules = extract_rules_from_node(children_right[node_id], right_rule)
        
        return left_rules + right_rules
    
    # Extract all rules starting from root node
    all_rules = extract_rules_from_node(0)
    
    # Sort rules by confidence (descending)
    all_rules.sort(key=lambda x: x[2], reverse=True)
    
    return all_rules

def simplify_rules_for_sequential_application(rules):
    """
    Convert decision tree rules to sequential logical rules
    focusing on presence/absence of n-grams
    """
    simplified_rules = []
    
    for rule_conditions, predicted_class, confidence in rules:
        # Parse the rule conditions to extract n-grams and their presence/absence
        conditions = rule_conditions.split(" AND ")
        simplified_conditions = []
        
        for condition in conditions:
            if "'" in condition and ("<=" in condition or ">" in condition):
                try:
                    # Extract feature name and threshold info - handle different formats
                    if "<=" in condition:
                        parts = condition.split("<=")
                        feature_name = parts[0].strip().strip("'")
                        threshold_str = parts[1].strip()
                        operator = "<="
                    elif ">" in condition:
                        parts = condition.split(">")
                        feature_name = parts[0].strip().strip("'")
                        threshold_str = parts[1].strip()
                        operator = ">"
                    else:
                        continue
                    
                    threshold = float(threshold_str)
                    
                    # For TF-IDF features, threshold > 0 typically means presence
                    if operator == "<=" and threshold < 0.001:
                        # Feature is absent (TF-IDF ≈ 0)
                        simplified_conditions.append(f"NOT '{feature_name}'")
                    elif operator == ">" and threshold < 0.001:
                        # Feature is present (TF-IDF > 0)
                        simplified_conditions.append(f"'{feature_name}'")
                    elif operator == "<=":
                        # Feature has low value (could be considered absent for binary classification)
                        simplified_conditions.append(f"NOT '{feature_name}'")
                    elif operator == ">":
                        # Feature has significant value (present)
                        simplified_conditions.append(f"'{feature_name}'")
                        
                except (ValueError, IndexError):
                    # Skip conditions that can't be parsed
                    continue
        
        if simplified_conditions:
            simplified_rule = " AND ".join(simplified_conditions)
            simplified_rules.append((simplified_rule, predicted_class, confidence))
    
    return simplified_rules

def save_rules_to_file(rules, filename="decision_tree_rules_cvo.txt"):
    """Save the extracted rules to a text file"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("DECISION TREE LOGICAL RULES\n")
        f.write("=" * 50 + "\n\n")
        f.write("Rules are applied sequentially. When a rule matches, the class is returned and no further rules are checked.\n\n")
        
        for i, (rule, predicted_class, confidence) in enumerate(rules, 1):
            f.write(f"RULE {i}:\n")
            f.write(f"IF {rule}\n")
            f.write(f"THEN CLASS: {predicted_class}\n")
            f.write(f"CONFIDENCE: {confidence:.3f}\n")
            f.write("-" * 50 + "\n\n")

if __name__ == "__main__":
    print("Extracting decision tree rules...")
    
    # Extract raw decision tree rules
    raw_rules = extract_decision_tree_rules()
    print(f"Extracted {len(raw_rules)} raw rules")
    
    # Simplify rules for sequential application
    simplified_rules = simplify_rules_for_sequential_application(raw_rules)
    print(f"Simplified to {len(simplified_rules)} rules")
    
    # Save to file
    save_rules_to_file(simplified_rules)
    print("Rules saved to 'decision_tree_rules_cvo.txt'")
    
    # Print first 10 rules as example
    print("\nFirst 10 rules:")
    for i, (rule, predicted_class, confidence) in enumerate(simplified_rules[:10], 1):
        print(f"{i}. IF {rule}")
        print(f"   THEN: {predicted_class} (confidence: {confidence:.3f})")
        print()
