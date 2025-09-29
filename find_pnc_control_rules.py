import re

def find_pnc_control_rules():
    with open('decision_tree_rules.txt', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all rules that predict PNC CONTROL ANUAL
    pnc_control_rules = re.findall(r'RULE \d+:(.*?THEN CLASS: PNC CONTROL ANUAL.*?)--------------------------------------------------', content, re.DOTALL)
    
    print('Found {} PNC CONTROL ANUAL rules:'.format(len(pnc_control_rules)))
    print("=" * 60)
    
    for i, rule in enumerate(pnc_control_rules, 1):
        print('Rule {}:'.format(i))
        print(rule.strip())
        print("-" * 40)
        print()

if __name__ == "__main__":
    find_pnc_control_rules()
