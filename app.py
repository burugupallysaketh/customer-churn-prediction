
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ==========================================
# PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📊",
    layout="wide"
)

# ==========================================
# CUSTOM DESIGN
# ==========================================

st.markdown("""
<style>
.stApp {
    background-color: #F4F7FC;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #172554, #1E3A8A);
}
[data-testid="stSidebar"] * {
    color: white !important;
}
.hero {
    background: linear-gradient(120deg, #1D4ED8, #7C3AED);
    padding: 30px;
    border-radius: 18px;
    color: white;
    margin-bottom: 25px;
}
.hero h1 {
    color: white;
    font-size: 34px;
}
.hero p {
    color: #E0E7FF;
    font-size: 16px;
}
.metric-card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    border-left: 5px solid #6366F1;
    box-shadow: 0 4px 14px rgba(30, 41, 59, 0.08);
    min-height: 115px;
}
.metric-label {
    color: #64748B;
    font-size: 14px;
}
.metric-value {
    color: #172554;
    font-size: 27px;
    font-weight: 750;
    margin-top: 8px;
}
.section-title {
    color: #172554;
    font-size: 23px;
    font-weight: 700;
    margin-top: 18px;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD DATA AND PIPELINE
# ==========================================

@st.cache_data
def load_data():
    data = pd.read_csv("Telco-Customer-Churn.csv")

    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"], errors="coerce"
    )

    data["TotalCharges"] = data["TotalCharges"].fillna(
        data["TotalCharges"].median()
    )

    return data


@st.cache_resource
def load_model():
    return joblib.load("churn_model.pkl")


df = load_data()

try:
    model = load_model()
    model_error = None
except Exception as error:
    model = None
    model_error = str(error)

# ==========================================
# DASHBOARD HEADER
# ==========================================

st.markdown("""
<div class="hero">
    <h1>📊 Customer Churn Analytics</h1>
    <p>Explore customer behavior, analyze retention trends,
    and predict possible customer churn.</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR FILTERS
# ==========================================

st.sidebar.title("🎛️ Dashboard Controls")
st.sidebar.write("Filter customer records.")

contracts = ["All"] + sorted(df["Contract"].unique().tolist())

contract_choice = st.sidebar.selectbox(
    "📄 Contract Type", contracts
)

churn_choice = st.sidebar.selectbox(
    "👥 Customer Status", ["All", "Stayed", "Left"]
)

internet_options = ["All"] + sorted(
    df["InternetService"].unique().tolist()
)

internet_choice = st.sidebar.selectbox(
    "🌐 Internet Service", internet_options
)

filtered = df.copy()

if contract_choice != "All":
    filtered = filtered[
        filtered["Contract"] == contract_choice
    ]

if churn_choice == "Stayed":
    filtered = filtered[filtered["Churn"] == "No"]
elif churn_choice == "Left":
    filtered = filtered[filtered["Churn"] == "Yes"]

if internet_choice != "All":
    filtered = filtered[
        filtered["InternetService"] == internet_choice
    ]

st.sidebar.markdown("---")
st.sidebar.caption("Customer Churn Data Science Project")

# ==========================================
# KPI CARDS
# ==========================================

total = len(filtered)
left_count = (filtered["Churn"] == "Yes").sum()
rate = left_count / total * 100 if total else 0
avg_charge = filtered["MonthlyCharges"].mean() if total else 0


def metric_card(label, value, color):
    st.markdown(f"""
    <div class="metric-card" style="border-left-color:{color}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


c1, c2, c3, c4 = st.columns(4)

with c1:
    metric_card("👥 Total Customers", f"{total:,}", "#2563EB")

with c2:
    metric_card("🚪 Customers Left", f"{left_count:,}", "#EF4444")

with c3:
    metric_card("📉 Churn Rate", f"{rate:.2f}%", "#F59E0B")

with c4:
    metric_card(
        "💳 Avg. Monthly Charges",
        f"${avg_charge:.2f}",
        "#10B981"
    )

# ==========================================
# CUSTOMER CHARTS
# ==========================================

st.markdown(
    '<div class="section-title">📈 Customer Insights</div>',
    unsafe_allow_html=True
)

chart1, chart2 = st.columns(2)

with chart1:
    with st.container(border=True):
        st.subheader("Customer Status")

        counts = filtered["Churn"].value_counts().reindex(
            ["No", "Yes"], fill_value=0
        )

        fig, ax = plt.subplots(figsize=(6, 4))

        bars = ax.bar(
            ["Stayed", "Left"],
            counts.values,
            color=["#2563EB", "#F97316"]
        )

        ax.set_ylabel("Number of Customers")
        ax.set_title("Stayed vs Left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.bar_label(bars, padding=4)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

with chart2:
    with st.container(border=True):
        st.subheader("Churn by Contract")

        contract_table = pd.crosstab(
            filtered["Contract"], filtered["Churn"]
        ).reindex(columns=["No", "Yes"], fill_value=0)

        ax = contract_table.rename(
            columns={"No": "Stayed", "Yes": "Left"}
        ).plot(
            kind="bar",
            figsize=(7, 4),
            color=["#2563EB", "#F97316"]
        )

        fig = ax.get_figure()
        ax.set_xlabel("Contract Type")
        ax.set_ylabel("Number of Customers")
        ax.legend(title="Status")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", rotation=10)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

chart3, chart4 = st.columns(2)

with chart3:
    with st.container(border=True):
        st.subheader("Monthly Charges by Status")

        if not filtered.empty:
            fig, ax = plt.subplots(figsize=(6, 4))

            sns.boxplot(
                data=filtered,
                x="Churn",
                y="MonthlyCharges",
                hue="Churn",
                palette={"No": "#2563EB", "Yes": "#F97316"},
                legend=False,
                ax=ax
            )

            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Stayed", "Left"])
            ax.set_xlabel("Customer Status")
            ax.set_ylabel("Monthly Charges ($)")
            ax.spines[["top", "right"]].set_visible(False)

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

with chart4:
    with st.container(border=True):
        st.subheader("Internet Service Breakdown")

        if not filtered.empty:
            service_counts = filtered["InternetService"].value_counts()

            fig, ax = plt.subplots(figsize=(6, 4))

            ax.pie(
                service_counts.values,
                labels=service_counts.index,
                autopct="%1.1f%%",
                startangle=90,
                colors=["#2563EB", "#10B981", "#F59E0B"],
                wedgeprops={
                    "width": 0.48,
                    "edgecolor": "white"
                }
            )

            ax.set_title("Customer Distribution")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

# ==========================================
# CUSTOMER TABLE
# ==========================================

st.markdown(
    '<div class="section-title">🗂️ Customer Records</div>',
    unsafe_allow_html=True
)

st.write(f"**{total:,} customers found**")

st.dataframe(
    filtered,
    width="stretch",
    hide_index=True,
    height=350
)

csv_data = filtered.to_csv(index=False).encode("utf-8")

st.download_button(
    "⬇️ Download Filtered Customer Data",
    data=csv_data,
    file_name="customer_churn_data.csv",
    mime="text/csv"
)

# ==========================================
# BUSINESS INSIGHT
# ==========================================

st.markdown("---")
st.markdown("### 💡 Quick Business Insight")

if total:
    st.info(
        f"Out of {total:,} customers in the current view, "
        f"{left_count:,} have left. The churn rate for this "
        f"filtered group is {rate:.2f}%. Investigate contract "
        "types and customer services to understand retention."
    )
else:
    st.warning("No customers match the selected filters.")

# ==========================================
# INDIVIDUAL CUSTOMER PREDICTION
# ==========================================

st.markdown("---")
st.markdown("## 🔮 Predict Customer Churn")

st.write(
    "Enter customer details below to estimate whether "
    "the customer may leave."
)

if model is None:
    st.error(
        "The trained model could not be loaded. "
        "Run python main.py in the project terminal, "
        "then refresh this page."
    )
    st.caption(model_error or "")

else:
    with st.form("customer_prediction_form"):
        st.subheader("👤 Customer Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Has Partner?", ["Yes", "No"])
            dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            multiple = st.selectbox(
                "Multiple Lines",
                ["No", "Yes", "No phone service"]
            )

        with col2:
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            internet = st.selectbox(
                "Internet Service", ["DSL", "Fiber optic", "No"]
            )
            security = st.selectbox(
                "Online Security", ["Yes", "No", "No internet service"]
            )
            backup = st.selectbox(
                "Online Backup", ["Yes", "No", "No internet service"]
            )
            protection = st.selectbox(
                "Device Protection", ["Yes", "No", "No internet service"]
            )
            support = st.selectbox(
                "Tech Support", ["Yes", "No", "No internet service"]
            )

        with col3:
            tv = st.selectbox(
                "Streaming TV", ["Yes", "No", "No internet service"]
            )
            movies = st.selectbox(
                "Streaming Movies", ["Yes", "No", "No internet service"]
            )
            contract = st.selectbox(
                "Contract", ["Month-to-month", "One year", "Two year"]
            )
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )

        col4, col5 = st.columns(2)

        with col4:
            monthly = st.number_input(
                "Monthly Charges ($)",
                min_value=0.0,
                max_value=1000.0,
                value=70.0
            )

        with col5:
            total_charges = st.number_input(
                "Total Charges ($)",
                min_value=0.0,
                max_value=100000.0,
                value=840.0
            )

        submitted = st.form_submit_button(
            "🔍 Predict Customer Churn",
            width="stretch"
        )

    if submitted:
        customer = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": protection,
            "TechSupport": support,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total_charges
        }])

        # The saved pipeline performs preprocessing and prediction.
        # Do not use pd.get_dummies() here.
        try:
            prediction = int(model.predict(customer)[0])
            probabilities = model.predict_proba(customer)[0]

            churn_class_index = list(model.classes_).index(1)
            churn_probability = float(
                probabilities[churn_class_index]
            )

            st.markdown("### 📊 Prediction Result")

            result_col1, result_col2 = st.columns(2)

            with result_col1:
                if prediction == 1:
                    st.error("⚠️ Prediction: Customer may leave")
                else:
                    st.success("✅ Prediction: Customer may stay")

            with result_col2:
                st.metric(
                    "Estimated churn probability",
                    f"{churn_probability * 100:.1f}%"
                )

            st.progress(churn_probability)

            st.caption(
                "This is a model estimate, not a guarantee. "
                "The model was trained on historical customer data."
            )

        except Exception as error:
            st.error("Prediction failed. Please check the error below.")
            st.code(str(error))

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.caption(
    "Customer Churn Analytics | Python | Pandas | "
    "Scikit-learn | Matplotlib | Seaborn | Streamlit"
)
