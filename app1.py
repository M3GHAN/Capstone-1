import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, OneClassSVM
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, silhouette_score

# -------------------------------
# Streamlit Config
# -------------------------------
st.set_page_config(page_title="Water Flow Anomaly Detector", layout="wide")
st.title("💧 Water Flow Anomaly Detection")
st.write("Choose between **Supervised** and **Unsupervised** anomaly detection approaches. Upload your dataset and explore results.")

# -------------------------------
# Email Utility
# -------------------------------
def send_email(smtp_server, smtp_port, username, password, sender, recipient, subject, html_table, plain):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipient

    msg.attach(MIMEText(plain, "plain"))
    msg.attach(MIMEText(html_table, "html"))

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(username, password)
        server.sendmail(sender, recipient, msg.as_string())
    return True, "Email sent successfully"

# -------------------------------
# Load Data
# -------------------------------
def load_data(uploaded_file):
    if uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls'):
        return pd.read_excel(uploaded_file)
    else:
        return pd.read_csv(uploaded_file)

# -------------------------------
# Supervised Preprocessing
# -------------------------------
def preprocess_supervised(df):
    df = df.copy()
    unnamed_cols = [c for c in df.columns if c.startswith('Unnamed')]
    df.drop(columns=unnamed_cols, inplace=True, errors='ignore')

    if 'DateTime' not in df.columns:
        if 'Date' in df.columns and 'Time' in df.columns:
            df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str),
                                            dayfirst=True, errors='coerce')
            df.drop(columns=['Date','Time'], inplace=True, errors='ignore')

    if 'Consumption' not in df.columns:
        alt = [c for c in df.columns if 'consump' in c.lower() or 'flow' in c.lower()]
        if alt:
            df.rename(columns={alt[0]:'Consumption'}, inplace=True)
        else:
            raise ValueError("No 'Consumption' column found.")

    if 'DateTime' not in df.columns:
        # fallback: try to parse an index as datetime
        try:
            df['DateTime'] = pd.to_datetime(df.index)
        except:
            df['DateTime'] = pd.RangeIndex(start=0, stop=len(df))

    df.sort_values('DateTime', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Threshold to mark a basic point anomaly
    df['Target'] = (df['Consumption'] > 0.60).astype(int)

    # lags
    for lag in range(1,4):
        df[f'Consumption_Lag_{lag}'] = df['Consumption'].shift(lag)
    df.fillna(0, inplace=True)

    # continuous anomaly: 3 consecutive above-threshold
    df['continuous_anomaly'] = (df['Target'].rolling(window=3, min_periods=1).sum() >= 3).astype(int)
    df['rolling_mean'] = df['Consumption'].rolling(window=12, min_periods=1).mean()
    df['rolling_var'] = df['Consumption'].rolling(window=12, min_periods=1).var().fillna(0)

    return df

def get_features_and_target(df):
    X = df[['Consumption','Consumption_Lag_1','Consumption_Lag_2','rolling_mean','rolling_var']].fillna(0)
    y = df['continuous_anomaly']
    return X, y

# -------------------------------
# Unsupervised Preprocessing
# -------------------------------
def preprocess_unsupervised(df):
    df = df.copy()
    unnamed_cols = [c for c in df.columns if c.startswith('Unnamed')]
    df.drop(columns=unnamed_cols, inplace=True, errors='ignore')

    # DateTime construction with flexible parsing
    if 'DateTime' not in df.columns:
        if 'Date' in df.columns and 'Time' in df.columns:
            df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str),
                                            dayfirst=True, errors='coerce')
            df.drop(columns=['Date','Time'], inplace=True, errors='ignore')
        else:
            # if index is datetime-like, use it
            try:
                df = df.set_index(pd.to_datetime(df.index))
                df.index.name = "DateTime"
                df = df.reset_index()
            except:
                pass

    if 'DateTime' in df.columns:
        df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
        df = df.set_index("DateTime")

    # create lags and diffs
    for lag in range(1, 4):
        if 'Consumption' in df.columns:
            df[f'Consumption_Lag_{lag}'] = df['Consumption'].shift(lag)

    if 'Consumption' in df.columns:
        df['consumption_diff'] = df['Consumption'].diff().fillna(0)
        df['rmean_3']= df['Consumption'].rolling(window=3, min_periods=1).mean()
        df['rstd_3'] = df['Consumption'].rolling(window=3, min_periods=1).std().fillna(0)
        df['rmean_6']= df['Consumption'].rolling(window=6, min_periods=1).mean()
        df['rstd_6'] = df['Consumption'].rolling(window=6, min_periods=1).std().fillna(0)
    else:
        # if consumption missing, create zeros so functions downstream don't break
        df['Consumption'] = 0
        df['consumption_diff'] = 0
        df['rmean_3'] = 0
        df['rstd_3'] = 0
        df['rmean_6'] = 0
        df['rstd_6'] = 0

    if 'Totalizer' in df.columns:
        df['totalizer_diff'] = df['Totalizer'].diff().fillna(0)
        df['totlizer_rol_diff']= df['Totalizer'].diff().rolling(window=3, min_periods=1).mean().fillna(0)
    else:
        df['Totalizer'] = 0
        df['totalizer_diff'] = 0
        df['totlizer_rol_diff'] = 0

    # time-of-day cyclical features if we have a datetime index
    if isinstance(df.index, pd.DatetimeIndex):
        seconds = df.index.hour*3600 + df.index.minute*60 + df.index.second
        day_seconds = 24*3600
        df['time_sin'] = np.sin(2*np.pi*seconds/day_seconds)
        df['time_cos'] = np.cos(2*np.pi*seconds/day_seconds)
    else:
        df['time_sin'] = 0
        df['time_cos'] = 0

    df.fillna(0, inplace=True)
    return df

def get_unsupervised_features(df):
    # Ensure columns exist
    expected_cols = ['Consumption','Totalizer','consumption_diff','Consumption_Lag_1','Consumption_Lag_2',
               'totalizer_diff','rmean_3','rstd_3','rmean_6','rstd_6','time_sin','time_cos']
    for c in expected_cols:
        if c not in df.columns:
            df[c] = 0
    return df[expected_cols]

# -------------------------------
# Anomaly Injection
# -------------------------------
def inject_anomalies(df):
    df = df.copy().reset_index(drop=True)
    n = len(df)
    labels = np.zeros(n, dtype=int)

    if n == 0:
        return df

    # Spikes
    spike_count = min(30, max(1, int(0.01 * n)))  # scale with size
    spike_idx = np.random.choice(n, spike_count, replace=False)
    df.loc[spike_idx, "Consumption"] = df["Consumption"].max() + np.random.uniform(1,2,spike_count)
    labels[spike_idx] = 1

    # Leak (gradual trend)
    if n > 200:
        leak_start = np.random.randint(0, n-200)
        leak_length = min(100, n - leak_start - 1)
        leak_trend = np.linspace(0, 2.0, leak_length)
        df.loc[leak_start:leak_start+leak_length-1, "Consumption"] += leak_trend
        labels[leak_start:leak_start+leak_length-1] = 1

    # Event surge
    if n > 300:
        event_start = np.random.randint(0, n-300)
        event_length = min(60, n - event_start - 1)
        surge_factor = 3.0
        df.loc[event_start:event_start+event_length-1, "Consumption"] *= surge_factor
        labels[event_start:event_start+event_length-1] = 1

    df["label"] = labels.astype(int)
    return df

# -------------------------------
# Models
# -------------------------------
def build_supervised_model(name):
    if name == 'Logistic Regression':
        return LogisticRegression(max_iter=1000, random_state=42)
    if name == 'KNN (k=3)':
        return KNeighborsClassifier(n_neighbors=3)
    if name == 'SVM (RBF)':
        return SVC(kernel='rbf', probability=True, random_state=42)
    if name == 'Decision Tree':
        return DecisionTreeClassifier(random_state=42)
    if name == 'Random Forest':
        return RandomForestClassifier(n_estimators=100, random_state=42)

# -------------------------------
# Plots
# -------------------------------
def plot_confusion(cm):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    return fig

def plot_roc(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    ax.plot([0,1],[0,1], linestyle='--')
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.legend()
    return fig, roc_auc

# -------------------------------
# Main Layout
# -------------------------------
if "anomalies" not in st.session_state:
    st.session_state["anomalies"] = pd.DataFrame()
if "results" not in st.session_state:
    st.session_state["results"] = {}

uploaded_file = st.file_uploader("📂 Upload Excel/CSV file", type=['xlsx','xls','csv'])

if uploaded_file is not None:
    if "raw_df" not in st.session_state:
        st.session_state["raw_df"] = load_data(uploaded_file)
    raw_df = st.session_state["raw_df"]
    st.subheader("Raw Preview")
    st.dataframe(raw_df.head())

    mode = st.sidebar.radio("Choose Detection Mode", ["-- Select Mode --","Supervised", "Unsupervised"], index=0)

    if mode == "-- Select Mode --":
        st.warning("⚠️ Please choose a detection mode to continue.")

    # ----------------- SUPERVISED -----------------
    elif mode == "Supervised":
        try:
            df = preprocess_supervised(raw_df)
        except Exception as e:
            st.error(f"Preprocessing error: {e}")
            st.stop()

        st.subheader("Processed Data (Supervised)")
        st.dataframe(df.head())

        model_name = st.sidebar.selectbox("Model", 
                                          ["Logistic Regression","KNN (k=3)","SVM (RBF)","Decision Tree","Random Forest"])
        test_size = st.sidebar.slider("Test size", 0.05, 0.5, 0.2)
        use_smote = st.sidebar.checkbox("Use SMOTE", True)
        random_state = st.sidebar.number_input("Seed", value=42, step=1)

        if st.button("Run Supervised Model", key="btn_supervised"):
            X, y = get_features_and_target(df)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=False)

            if use_smote:
                try:
                    X_train, y_train = SMOTE(random_state=random_state).fit_resample(X_train, y_train)
                except Exception as e:
                    st.warning(f"SMOTE failed or not applicable: {e}")

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = build_supervised_model(model_name)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_score = model.predict_proba(X_test_scaled)[:,1] if hasattr(model,"predict_proba") else y_pred

            # Save results
            st.session_state["results"] = {
                "model_name": model_name,
                "accuracy": accuracy_score(y_test, y_pred),
                "report": classification_report(y_test, y_pred, output_dict=False),
                "confusion": confusion_matrix(y_test, y_pred),
                "roc": roc_curve(y_test, y_score),
                "y_test": y_test,
                "y_pred": y_pred
            }

            anomalies = df.iloc[y_test.index].copy()
            anomalies["Prediction"] = y_pred
            st.session_state['anomalies'] = anomalies[anomalies["Prediction"] == 1][["DateTime","Consumption","Prediction"]]

        # Render results if available
        if st.session_state["results"]:
            res = st.session_state["results"]
            st.markdown(f"### Supervised Model: **{res.get('model_name','(unknown)')}**")
            st.write("**Accuracy:**", res["accuracy"])
            st.text(res["report"])
            st.pyplot(plot_confusion(res["confusion"]))
            fpr, tpr, _ = res["roc"]
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f"AUC={auc(fpr,tpr):.3f}")
            ax.plot([0,1],[0,1],'--')
            ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
            ax.legend()
            st.pyplot(fig)

    # ----------------- UNSUPERVISED -----------------
    elif mode == "Unsupervised":
        inject = st.sidebar.radio("Inject anomalies?", ["No", "Yes"])
        df = raw_df.copy()
        if inject == "Yes":
            df = inject_anomalies(df)
            st.success(f"Injected anomalies: {int(df.get('label', pd.Series([])).sum())}")

        try:
            data = preprocess_unsupervised(df)
        except Exception as e:
            st.error(f"Preprocessing error: {e}")
            st.stop()

        st.subheader("Processed Data (Unsupervised)")
        st.dataframe(data.head())

        X = get_unsupervised_features(data)

        algo = st.sidebar.selectbox("Algorithm", ["Isolation Forest","One-Class SVM"])
        conta = st.sidebar.slider("Contamination / ν (nu)", 0.01, 0.2, 0.02, step=0.01)

        if st.button("Run Unsupervised Model", key="btn_unsupervised"):
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)

            # Fit model and obtain predictions and anomaly scores
            if algo == "Isolation Forest":
                model = IsolationForest(contamination=conta, random_state=42)
                model.fit(X_scaled)
                raw_pred = model.predict(X_scaled)               # +1 normal, -1 anomaly
                # decision_function: higher means less anomalous; invert to make "anomaly score" where higher = more anomalous
                try:
                    raw_scores = -model.decision_function(X_scaled)
                except Exception:
                    # fallback: use negative of estimated_samples or zeros
                    raw_scores = np.zeros(len(X_scaled))
            else:
                model = OneClassSVM(kernel="rbf", nu=conta, gamma="scale")
                model.fit(X_scaled)
                raw_pred = model.predict(X_scaled)               # +1 normal, -1 anomaly
                try:
                    raw_scores = -model.decision_function(X_scaled)
                except Exception:
                    raw_scores = np.zeros(len(X_scaled))

            # Convert predictions to 0=normal,1=anomaly
            pred = (raw_pred == -1).astype(int)
            data["Anomaly"] = pred
            data["Anomaly_Label"] = data["Anomaly"].map({0:"Normal",1:"Anomaly"})
            data["Anomaly_Score"] = raw_scores

            # Save anomalies in session
            anomalies_df = data[data["Anomaly"] == 1].copy()
            # ensure DateTime exists
            try:
                anomalies_df = anomalies_df.reset_index()
                if 'DateTime' in anomalies_df.columns:
                    anomalies_df = anomalies_df[['DateTime','Consumption']]
                else:
                    anomalies_df['DateTime'] = anomalies_df.index
            except Exception:
                anomalies_df['DateTime'] = anomalies_df.index
            anomalies_df["Prediction"] = 1
            st.session_state['anomalies'] = anomalies_df[['DateTime','Consumption','Prediction']].copy()

            st.markdown(f"### Unsupervised Algorithm: **{algo}**")

            # If ground truth label exists in original df, show accuracy + confusion
            if "label" in df.columns:
                y_true = df['label'].astype(int).values
                st.write("**Accuracy (using provided 'label')**:", accuracy_score(y_true, pred))
                st.text(classification_report(y_true, pred))
                st.pyplot(plot_confusion(confusion_matrix(y_true, pred)))
            else:
                # No label available — show anomaly score analysis, distributions, threshold ratio, silhouette
                st.info("No ground-truth 'label' column found — showing unsupervised diagnostic analyses.")

                # 1) Anomaly Score Distribution
                fig, ax = plt.subplots()
                sns.histplot(data=data, x="Anomaly_Score", hue="Anomaly_Label", kde=True, stat="density", common_norm=False, ax=ax)
                ax.set_title("Anomaly Score Distribution (higher = more anomalous)")
                st.pyplot(fig)

                # Summary statistics of scores for predicted classes
                score_stats = data.groupby("Anomaly_Label")["Anomaly_Score"].agg(['count','mean','std','min','max']).reset_index()
                st.write("**Anomaly Score Summary by Predicted Class**")
                st.dataframe(score_stats)

                # 2) Threshold-based Outlier Ratio
                detected_pct = 100.0 * data["Anomaly"].mean() if len(data) > 0 else 0.0
                expected_pct = 100.0 * conta
                st.write(f"**Detected anomaly percentage:** {detected_pct:.3f}%")
                st.write(f"**Expected (contamination) percentage:** {expected_pct:.3f}%")
                if detected_pct > expected_pct:
                    st.warning("Detected anomaly percentage is higher than expected contamination — consider lowering contamination or inspecting data.")
                elif detected_pct < expected_pct:
                    st.info("Detected anomaly percentage is lower than the expected contamination parameter.")

                # 3) Silhouette Score (attempt)
                try:
                    # silhouette needs at least 2 labels and each label has >1 sample
                    labels_for_sil = data["Anomaly"].values
                    unique, counts = np.unique(labels_for_sil, return_counts=True)
                    if len(unique) >= 2 and np.all(counts > 1):
                        sil = silhouette_score(X_scaled, labels_for_sil)
                        st.write(f"**Silhouette Score (using features & predicted anomaly labels):** {sil:.4f}")
                    else:
                        st.write("**Silhouette Score:** Cannot compute (need at least 2 clusters with >1 sample each).")
                except Exception as e:
                    st.write(f"**Silhouette Score:** Error computing silhouette score: {e}")

            # Scatter plot: Totalizer vs Consumption colored by anomaly
            fig, ax = plt.subplots()
            hue_order = ["Normal","Anomaly"]
            if 'Totalizer' in data.columns:
                sns.scatterplot(data=data.reset_index(), x="Totalizer", y="Consumption",
                                hue="Anomaly_Label", hue_order=hue_order, ax=ax)
                ax.set_title(f"{algo}: Anomaly Detection (Totalizer vs Consumption)")
                st.pyplot(fig)
            else:
                # If no Totalizer column, plot Anomaly Score vs Consumption
                fig, ax = plt.subplots()
                sns.scatterplot(data=data.reset_index(), x="Anomaly_Score", y="Consumption",
                                hue="Anomaly_Label", hue_order=hue_order, ax=ax)
                ax.set_title(f"{algo}: Anomaly Score vs Consumption")
                st.pyplot(fig)

    # ----------------- EMAIL ALERT -----------------
    if not st.session_state['anomalies'].empty:
        st.subheader('📧 Alert via Email')
        send_email_opt = st.checkbox('Send detected anomalies via email?')
        if send_email_opt:
            smtp_server = 'smtp.gmail.com'
            smtp_port = 587
            username  = 'meghan111117@gmail.com'
            password = 'uawi guir waik sdnj'  # NOTE: It's better to store this securely (secrets, env vars)
            sender = 'meghan111117@gmail.com'
            recipient = st.text_input("Enter recipient email for alerts")
            subject = 'Detected Anomalies - Water Flow'

            if st.button('Send Alert Email', key="btn_email"):
                anomalies = st.session_state.get('anomalies', pd.DataFrame())
                html_table = anomalies.to_html(index=False)
                plain = anomalies.to_string(index=False)
                with st.spinner('Sending email...'):
                    try:
                        ok, msg = send_email(smtp_server, smtp_port, username, password, sender, recipient, subject, html_table, plain)
                        st.success('Alert email sent successfully')
                    except Exception as e:
                        st.error(f'Failed to send email: {e}')
else:
    st.info("Upload data to begin.")
