
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from io import BytesIO
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor

st.set_page_config(page_title='BalanceBite Analytics', layout='wide')

# -------------------- Load Data --------------------
@st.cache_data
def load_data(path):
    return pd.read_csv(path)
data = load_data('data/balancebite_survey_synthetic_1000.csv')

st.sidebar.title('BalanceBite Dashboard')
st.sidebar.write('Filter & Navigation')
page = st.sidebar.selectbox('Select a Tab', 
    ['Data Visualisation', 'Classification', 'Clustering', 'Association Rules', 'Regression'])

# Shared filter example
city_filter = st.sidebar.multiselect('Filter by City', sorted(data['City'].unique()), default=list(data['City'].unique()))
df = data[data['City'].isin(city_filter)]

# -------------------- Helper functions --------------------
def descriptive_insights(df):
    st.subheader('Quick KPIs')
    col1, col2, col3 = st.columns(3)
    col1.metric('Total Respondents', len(df))
    col2.metric('Avg Spend (₹)', round(df['Avg_Spend_Order'].mean(), 1))
    yes_rate = (df['Likely_Try_30_Days']=='Yes').mean()*100
    col3.metric('Adoption Intent (%)', f'{yes_rate:.1f}')

    # 10 complex plots
    st.markdown('---')
    st.markdown('### Distributions & Relationships')
    import seaborn as sns
    import matplotlib.pyplot as plt
    plots = [
        ('Age vs Spend', 'Age_Bracket', 'Avg_Spend_Order'),
        ('Income vs Spend', 'Monthly_Income', 'Avg_Spend_Order'),
        ('Workout vs Orders', 'Weekly_Workouts', 'Orders_Per_Week'),
        ('Price Perception', 'Fair_Price_Bundle', 'Avg_Spend_Order'),
    ]
    for title, x, y in plots:
        fig, ax = plt.subplots()
        if df[x].dtype == object:
            sns.boxplot(x=df[x], y=df[y], ax=ax)
        else:
            sns.scatterplot(x=df[x], y=df[y], ax=ax)
        ax.set_title(title)
        st.pyplot(fig)

    # Heatmap correlation
    corr_cols = ['Monthly_Income', 'Avg_Spend_Order', 'Weekly_Workouts', 'Orders_Per_Week']
    fig, ax = plt.subplots()
    sns.heatmap(df[corr_cols].corr(), annot=True, ax=ax)
    ax.set_title('Numeric Feature Correlations')
    st.pyplot(fig)

def run_classification(df):
    st.header('Classification – Adoption Prediction')
    target_col = 'Likely_Try_30_Days'
    df_clf = df.copy()
    df_clf[target_col] = df_clf[target_col].map({'Yes':1, 'Maybe':0, 'No':0})
    # Basic preprocessing
    X = pd.get_dummies(df_clf.drop(columns=[target_col]), drop_first=True)
    y = df_clf[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    models = {
        'KNN': KNeighborsClassifier(),
        'DT': DecisionTreeClassifier(max_depth=6),
        'RF': RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42),
        'GBRT': GradientBoostingClassifier(random_state=42)
    }
    metrics = []
    probs = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        metrics.append((name, acc, prec, rec, f1))
        y_proba = model.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        probs[name] = (fpr, tpr, auc(fpr, tpr))

    st.subheader('Performance Table')
    perf_df = pd.DataFrame(metrics, columns=['Model','Accuracy','Precision','Recall','F1'])
    st.dataframe(perf_df.style.format({c:'{:.2f}' for c in perf_df.columns[1:]}))

    # Confusion matrix dropdown
    sel_model = st.selectbox('Select model for Confusion Matrix', list(models.keys()))
    cm = confusion_matrix(y_test, models[sel_model].predict(X_test))
    st.write('Confusion Matrix')
    st.write(pd.DataFrame(cm, index=['Actual 0','Actual 1'], columns=['Pred 0','Pred 1']))

    # ROC curve
    fig, ax = plt.subplots()
    for name, (fpr, tpr, roc_auc) in probs.items():
        ax.plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.2f})')
    ax.plot([0,1],[0,1],'--')
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title('ROC Curves')
    ax.legend()
    st.pyplot(fig)

    # Upload new data
    st.markdown('---')
    st.subheader('Predict on New Data')
    uploaded = st.file_uploader('Upload CSV of new respondents (without target)', type=['csv'])
    if uploaded:
        new_df = pd.read_csv(uploaded)
        new_X = pd.get_dummies(new_df, drop_first=True)
        # Align with training columns
        new_X = new_X.reindex(columns=X.columns, fill_value=0)
        new_scaled = scaler.transform(new_X)
        preds = models[sel_model].predict(new_scaled)
        result = new_df.copy()
        result['Predicted_Adoption'] = preds
        st.write(result.head())
        # Download
        csv_buff = result.to_csv(index=False).encode('utf-8')
        st.download_button('Download Predictions', csv_buff, 'predictions.csv', 'text/csv')

def run_clustering(df):
    st.header('Customer Segmentation – K‑Means')
    num_slider = st.sidebar.slider('Select #Clusters (k)', 2, 10, 4)
    feature_cols = ['Monthly_Income','Avg_Spend_Order','Weekly_Workouts','Orders_Per_Week']
    X = df[feature_cols].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # elbow
    ks = range(2, 11)
    sse = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X_scaled)
        sse.append(km.inertia_)
    fig, ax = plt.subplots()
    ax.plot(ks, sse, marker='o')
    ax.set_xlabel('k')
    ax.set_ylabel('SSE')
    ax.set_title('Elbow Curve')
    st.pyplot(fig)

    # Run clustering at selected k
    km_final = KMeans(n_clusters=num_slider, random_state=42)
    labels = km_final.fit_predict(X_scaled)
    df_seg = df.copy()
    df_seg['Cluster'] = labels
    # Persona table
    persona = df_seg.groupby('Cluster')[feature_cols].mean().round(1)
    st.subheader('Cluster Personas (mean values)')
    st.dataframe(persona)
    # Download cluster-labelled data
    csv_buf = df_seg.to_csv(index=False).encode('utf-8')
    st.download_button('Download Cluster‑Labelled Data', csv_buf, 'clustered_data.csv', 'text/csv')

def run_association(df):
    st.header('Association Rule Mining – Apriori')
    col1 = 'Favourite_Flavours'
    col2 = 'Spirits_Enjoyed'
    # One‑hot encode multi‑select columns
    def expand_multiselect(series):
        unique_vals = set(itertools.chain.from_iterable([str(x).split(',') for x in series]))
        unique_vals = [v.strip() for v in unique_vals if v and v.strip()!='None']
        return unique_vals
    import itertools
    # Build basket
    basket = []
    for _, row in df[[col1, col2]].iterrows():
        items = []
        for val in str(row[col1]).split(','):
            val = val.strip()
            if val:
                items.append(f'Flavour_{val}')
        for val in str(row[col2]).split(','):
            val = val.strip()
            if val and val!='None':
                items.append(f'Spirit_{val}')
        basket.append(items)
    # Transaction -> one‑hot DataFrame
    all_items = sorted(set(itertools.chain.from_iterable(basket)))
    onehot = pd.DataFrame(0, index=range(len(basket)), columns=all_items)
    for idx, items in enumerate(basket):
        onehot.loc[idx, items] = 1
    freq = apriori(onehot, min_support=0.05, use_colnames=True)
    rules = association_rules(freq, metric='confidence', min_threshold=0.3)
    rules = rules.sort_values('confidence', ascending=False).head(10)
    st.write('Top‑10 Rules by Confidence')
    st.dataframe(rules[['antecedents','consequents','support','confidence','lift']])

def run_regression(df):
    st.header('Regression Insights – Spend Prediction')
    target = 'Avg_Spend_Order'
    feat_cols = ['Monthly_Income','Orders_Per_Week','Weekly_Workouts']
    X = df[feat_cols]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    models = {
        'Linear': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'DT Regressor': DecisionTreeRegressor(max_depth=5, random_state=42)
    }
    st.subheader('Model Performance (R²)')
    rows = []
    for name,m in models.items():
        m.fit(X_train, y_train)
        r2 = m.score(X_test, y_test)
        rows.append((name, round(r2,3)))
    st.table(pd.DataFrame(rows, columns=['Model','R2']))

    # Quick scatter insights
    st.markdown('---')
    import seaborn as sns
    for col in feat_cols:
        fig, ax = plt.subplots()
        sns.scatterplot(x=df[col], y=y, ax=ax)
        ax.set_title(f'{col} vs Spend')
        st.pyplot(fig)

# -------------------- NAV ROUTER --------------------
if page == 'Data Visualisation':
    descriptive_insights(df)
elif page == 'Classification':
    run_classification(df)
elif page == 'Clustering':
    run_clustering(df)
elif page == 'Association Rules':
    run_association(df)
elif page == 'Regression':
    run_regression(df)
