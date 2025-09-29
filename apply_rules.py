import re

def parse_rules_from_file(filename="decision_tree_rules.txt"):
    """Parse the rules from the generated text file"""
    rules = []
    
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split into individual rules
    rule_blocks = content.split('RULE ')[1:]  # Skip the header
    
    for block in rule_blocks:
        # Extract rule number
        rule_number = int(block.split(':')[0].strip())
        
        # Extract IF condition
        if_match = re.search(r'IF (.*?)\nTHEN CLASS:', block, re.DOTALL)
        if not if_match:
            continue
            
        condition_str = if_match.group(1).strip()
        
        # Extract THEN class
        class_match = re.search(r'THEN CLASS: (.*?)\n', block)
        if not class_match:
            continue
            
        predicted_class = class_match.group(1).strip()
        
        # Extract confidence
        confidence_match = re.search(r'CONFIDENCE: (.*?)\n', block)
        confidence = float(confidence_match.group(1).strip()) if confidence_match else 0.0
        
        # Parse individual conditions
        conditions = []
        for part in condition_str.split(' AND '):
            part = part.strip()
            if part.startswith("NOT '"):
                conditions.append(('absent', part[5:-1]))  # Remove "NOT '" and "'"
            elif part.startswith("'"):
                conditions.append(('present', part[1:-1]))  # Remove "'" and "'"
        
        rules.append({
            'number': rule_number,
            'conditions': conditions,
            'predicted_class': predicted_class,
            'confidence': confidence
        })
    
    return rules

def check_condition(condition_type, ngram, text):
    """Check if a condition is satisfied for the given text"""
    if condition_type == 'present':
        return ngram in text
    elif condition_type == 'absent':
        return ngram not in text
    return False

def apply_rules_to_text(text, rules):
    """Apply rules sequentially to text and return the first matching class"""
    text_lower = text.lower()
    
    for rule in rules:
        rule_matches = True
        
        for condition_type, ngram in rule['conditions']:
            if not check_condition(condition_type, ngram.lower(), text_lower):
                rule_matches = False
                break
        
        if rule_matches:
            return {
                'class': rule['predicted_class'],
                'rule_number': rule['number'],
                'confidence': rule['confidence'],
                'matched_rule': rule
            }
    
    return {
        'class': 'UNKNOWN',
        'rule_number': None,
        'confidence': 0.0,
        'matched_rule': None
    }

def classify_text(text, rules_file="decision_tree_rules_cvo.txt"):
    """Classify text using the extracted rules"""
    rules = parse_rules_from_file(rules_file)
    return apply_rules_to_text(text, rules)

if __name__ == "__main__":
    # Example usage
    test_texts = [
        "declaracion anual pnc 2024",
        "solicitud pension no contributiva",
        "complemento alquiler pnc",
        "reenvio de documentacion",
        "certificado de empadronamiento"
    ]
    
    print("Loading rules...")
    rules = parse_rules_from_file()
    print(f"Loaded {len(rules)} rules")
    
    print("\nTesting classification:")
    print("=" * 50)
    
    for text in test_texts:
        result = apply_rules_to_text(text, rules)
        print(f"Text: '{text}'")
        print(f"Predicted class: {result['class']}")
        print(f"Rule number: {result['rule_number']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print("-" * 30)
