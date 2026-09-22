import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Nata Supermarkets", layout="centered")

COST, REVENUE = 3, 11          # case economics: $ per contact, $ per response
FEATURES = ["Income", "Age", "Children", "Recency", "Tenure_Days", "TotalSpend",
            "MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts",
            "MntSweetProducts", "MntGoldProds", "NumWebPurchases",
            "NumCatalogPurchases", "NumStorePurchases", "NumDealsPurchases",
            "NumWebVisitsMonth", "AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3",
            "AcceptedCmp4", "AcceptedCmp5"]


@st.cache_data
def load():
    return pd.read_excel("nata_scored.xlsx")


@st.cache_resource
def get_model(df):
    X, y = df[FEATURES], df["Response"]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                random_state=42).fit(Xtr, ytr)
    test = pd.DataFrame({"y": yte.values,
                         "p": rf.predict_proba(Xte)[:, 1]})
    return rf, test


df = load()
rf, test = get_model(df)

st.title("Nata Supermarkets — Customer Analytics")
st.caption(f"{len(df):,} customers · base response rate "
           f"{df['Response'].mean()*100:.1f}% · contact cost ${COST} · "
           f"response value ${REVENUE}")

tab1, tab2, tab3 = st.tabs(["Segments", "Targeting ROI", "Score a customer"])

# ---- 1. Segment table -------------------------------------------------------
with tab1:
    st.subheader("Customer segments")
    prof = df.groupby("Segment").agg(
        Customers=("ID", "size"),
        Avg_income=("Income", "mean"),
        Avg_spend=("TotalSpend", "mean"),
        Response_rate=("Response", "mean"),
    )
    prof["Revenue_%"] = (df.groupby("Segment")["TotalSpend"].sum()
                         / df["TotalSpend"].sum() * 100)
    prof = prof.sort_values("Avg_spend", ascending=False).round(
        {"Avg_income": 0, "Avg_spend": 0, "Response_rate": 3, "Revenue_%": 1})
    prof["Response_rate"] = (prof["Response_rate"] * 100).round(0)
    st.dataframe(prof, use_container_width=True)
    st.caption("Two segments (~43% of customers) drive ~83% of revenue.")

# ---- 2. ROI slider ----------------------------------------------------------
with tab2:
    st.subheader("Targeted vs mass marketing")
    N = len(test)
    R = int(test["y"].sum())
    base = R * REVENUE - N * COST

    pct = st.slider("Contact the top X% by predicted response", 5, 100, 20, 5)
    k = int(N * pct / 100)
    sub = test.sort_values("p", ascending=False).head(k)
    got = int(sub["y"].sum())
    profit = got * REVENUE - k * COST

    c1, c2, c3 = st.columns(3)
    c1.metric("Contacted", f"{k}", f"{pct}%")
    c2.metric("Responders caught", f"{got}/{R}", f"{got/R*100:.0f}%")
    c3.metric("Profit", f"${profit}", f"${profit-base} vs all")

    st.write(f"**Mass-mailing everyone:** ${base}  (a loss)")
    st.write(f"**Targeting top {pct}%:** ${profit}")
    if profit > base:
        st.success(f"Targeting improves profit by ${profit-base}.")

# ---- 3. Single-customer scorer ---------------------------------------------
with tab3:
    st.subheader("Predict one customer's response")
    med = df[FEATURES].median()
    rec = med.copy()
    c1, c2 = st.columns(2)
    rec["Income"] = c1.number_input("Income", 0, 200000, 60000, 1000)
    rec["Age"] = c2.number_input("Age", 18, 100, 45)
    rec["TotalSpend"] = c1.number_input("Total spend (2 yrs)", 0, 3000, 800, 50)
    rec["Children"] = c2.number_input("Children at home", 0, 5, 0)
    rec["Recency"] = c1.number_input("Days since last purchase", 0, 100, 30)
    prob = rf.predict_proba(pd.DataFrame([rec])[FEATURES])[0, 1]
    st.metric("Predicted response probability", f"{prob*100:.1f}%")
    st.caption("Other attributes held at dataset medians.")
