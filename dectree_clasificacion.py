import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

def entrenar():
    df=pd.read_csv("textos_procesados.csv")
    df["asunto preprocesado"]=df["asunto preprocesado"].apply(lambda x: str(x))


    #divide data into train and test
    X_train, X_test, y_train, y_test = train_test_split(df["asunto preprocesado"], df["procedimiento"], test_size=0.2, random_state=42)

    #read spanish_stopwords.txt into a list
    with open("spanish_stopwords.txt") as f:
        stopwords = f.readlines()
    stopwords = [x.strip() for x in stopwords]


    vectorizer = TfidfVectorizer(strip_accents="unicode",lowercase=True,ngram_range=(1,5),max_features=1000)
    # replace NaN with empty string and ensure values are strings so the vectorizer doesn't receive np.nan
    X_train = X_train.fillna('').astype(str)
    X_test = X_test.fillna('').astype(str)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = DecisionTreeClassifier(random_state=42, min_samples_leaf=2)
    clf.fit(X_train_vec, y_train)
    #get classes from classifier
    clases = clf.classes_

    plt.figure(figsize=(12,8))
    plot_tree(clf, feature_names=vectorizer.get_feature_names_out(), class_names=clases, filled=True)
    plt.savefig('decision_tree.pdf', format='pdf', bbox_inches='tight')

    plt.show()

    # 2. Get the cost-complexity pruning path
    path = clf.cost_complexity_pruning_path(X_train_vec, y_train)
    ccp_alphas = path.ccp_alphas

    # 3. Cross-validate for each alpha value
    scores = []
    for ccp_alpha in ccp_alphas:
        clf_temp = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha, min_samples_leaf=2)
        score = cross_val_score(clf_temp, X_train_vec, y_train, cv=5).mean()
        scores.append(score)
    # 4. Find the alpha that gives the highest cross-validation score
    optimal_alpha = ccp_alphas[np.argmax(scores)]

    # 5. Train your final model with the optimal alpha
    final_model = DecisionTreeClassifier(random_state=42, ccp_alpha=optimal_alpha, min_samples_leaf=5)
    final_model.fit(X_train_vec, y_train)

    #display model

   

    plt.figure(figsize=(12,8))
    plot_tree(final_model, feature_names=vectorizer.get_feature_names_out(), class_names=clases, filled=True)
    plt.savefig('decision_tree_simplified.pdf', format='pdf', bbox_inches='tight')

    plt.show()

    y_pred = final_model.predict(X_test_vec)

    evaluation=[]
    for i in range(len(y_test)):

        asunto=X_test.iloc[i]
        proc=y_test.iloc[i]
        #decode one hot encoding
        proc_ai=y_pred[i]
        evaluation.append([asunto,proc,proc_ai])

    df_eval=pd.DataFrame(evaluation,columns=["asunto preprocesado","procedimiento","procedimiento_ai"])
    df_eval.to_csv("evaluacion_dectree_bandeja_cvo_procedimientos.csv",index=False)

    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))
    print(final_model.score(X_test_vec, y_test))


    #save model for later use
    joblib.dump(final_model, 'clasificador_cvo_exp.pkl')
    #save categorias
    joblib.dump(clases, 'categorias_cvo_exp.pkl')
    #save vectorizer
    joblib.dump(vectorizer, 'vectorizer_cvo_exp.pkl')

if __name__ == "__main__":
    entrenar()  







