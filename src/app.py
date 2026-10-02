import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Healthcare Analytics", page_icon="🏥", layout="wide")

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "doctor_visits.csv")
df = pd.read_csv(DATA)

st.title("🏥 Healthcare Analytics for Doctor Visits")
st.caption("Interactive exploration of the supplied student dataset")

# Sidebar filters
st.sidebar.header("Filters")
gender = st.sidebar.multiselect("Gender", sorted(df["gender"].unique()), default=sorted(df["gender"].unique()))
nchronic = st.sidebar.multiselect("Chronic condition", sorted(df["nchronic"].unique()), default=sorted(df["nchronic"].unique()))
lchronic = st.sidebar.multiselect("Long-term chronic condition", sorted(df["lchronic"].unique()), default=sorted(df["lchronic"].unique()))

filtered = df[
    df["gender"].isin(gender) &
    df["nchronic"].isin(nchronic) &
    df["lchronic"].isin(lchronic)
]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Records", f"{len(filtered):,}")
c2.metric("Average Visits", f"{filtered['visits'].mean():.2f}")
c3.metric("Average Illnesses", f"{filtered['illness'].mean():.2f}")
c4.metric("Avg Reduced Days", f"{filtered['reduced'].mean():.2f}")

tab1, tab2, tab3 = st.tabs(["Overview", "Health Patterns", "Data"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(7,4))
        vc = filtered["visits"].value_counts().sort_index()
        ax.bar(vc.index.astype(str), vc.values)
        ax.set_title("Doctor Visit Distribution")
        ax.set_xlabel("Doctor Visits")
        ax.set_ylabel("Records")
        st.pyplot(fig)
        plt.close(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(7,4))
        groups = [filtered.loc[filtered.gender == g, "visits"].values for g in sorted(filtered.gender.unique())]
        ax.boxplot(groups, labels=sorted(filtered.gender.unique()))
        ax.set_title("Visits by Gender")
        ax.set_xlabel("Gender")
        ax.set_ylabel("Doctor Visits")
        st.pyplot(fig)
        plt.close(fig)

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(7,4))
        groups = [filtered.loc[filtered.illness == i, "visits"].values for i in sorted(filtered.illness.unique())]
        ax.boxplot(groups, labels=sorted(filtered.illness.unique()))
        ax.set_title("Visits by Number of Illnesses")
        ax.set_xlabel("Illness Count")
        ax.set_ylabel("Doctor Visits")
        st.pyplot(fig)
        plt.close(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(7,4))
        ax.scatter(filtered["reduced"], filtered["visits"], alpha=0.4)
        ax.set_title("Reduced Activity Days vs Visits")
        ax.set_xlabel("Reduced Activity Days")
        ax.set_ylabel("Doctor Visits")
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Average Visits by Healthcare Support")
    support_cols = ["private", "freepoor", "freerepat"]
    support_summary = []
    for col in support_cols:
        temp = filtered.groupby(col)["visits"].mean().reset_index()
        temp["Support Type"] = col
        temp = temp.rename(columns={col: "Status"})
        support_summary.append(temp)
    st.dataframe(pd.concat(support_summary, ignore_index=True), use_container_width=True)

with tab3:
    st.dataframe(filtered, use_container_width=True)
    st.download_button(
        "Download filtered CSV",
        filtered.to_csv(index=False).encode("utf-8"),
        "filtered_doctor_visits.csv",
        "text/csv"
    )
