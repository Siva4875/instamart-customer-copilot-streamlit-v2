from pathlib import Path
import json

import pandas as pd
import plotly.express as px
import streamlit as st


# --------------------------------------------------
# Application configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "clean"
OUTPUT_DIR = BASE_DIR / "outputs"

st.set_page_config(
    page_title="Instamart Customer Intelligence",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# Data loading
# --------------------------------------------------

@st.cache_data
def load_data():
    customers = pd.read_csv(DATA_DIR / "customers.csv")
    orders = pd.read_csv(DATA_DIR / "orders.csv")
    products = pd.read_csv(DATA_DIR / "products.csv")

    segments = pd.read_csv(OUTPUT_DIR / "customer_segments.csv")
    tickets = pd.read_csv(OUTPUT_DIR / "triage_results.csv")
    retention = pd.read_csv(OUTPUT_DIR / "retention_list.csv")

    orders["order_ts"] = pd.to_datetime(
        orders["order_ts"], errors="coerce"
    )

    return customers, orders, products, segments, tickets, retention


customers, orders, products, segments, tickets, retention = load_data()


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("🛒 Instamart")
st.sidebar.caption("Customer Intelligence & Support Copilot")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Executive Dashboard",
        "Customer Intelligence",
        "Churn Prediction",
        "Support Ticket Management",
        "AI Support Copilot",
    ],
)

st.sidebar.divider()
st.sidebar.caption("Portfolio Edition | Version 2.0")


# --------------------------------------------------

# --------------------------------------------------

# --------------------------------------------------

# --------------------------------------------------
# --------------------------------------------------
# AI Support Copilot Page
# --------------------------------------------------

if page == "AI Support Copilot":

    from src.policy_rag import PolicyRAG

    st.title("AI Support Copilot")
    st.caption(
        "Search Instamart policy documents using TF-IDF-based retrieval. "
        "Review the original policy excerpts before making a support decision."
    )

    @st.cache_resource
    def load_policy_rag():
        return PolicyRAG()

    try:
        rag = load_policy_rag()

        st.info(
            "This portfolio demonstration uses offline policy retrieval "
            "with TF-IDF. It does not connect to a hosted generative AI service "
            "or automatically approve refunds."
        )

        st.subheader("Ask a policy question")

        with st.form("policy_search_form"):
            question = st.text_area(
                "Enter your customer support question",
                placeholder=(
                    "Example: What delivery time makes a completed order "
                    "eligible for a full refund?"
                ),
                height=100,
            )

            submitted = st.form_submit_button(
                "Search Policy Documents",
                type="primary",
                width="stretch",
            )

        st.caption(
            "Try questions about refunds, delivery fees, payments, privacy "
            "or escalation procedures."
        )

        if submitted:
            if not question.strip():
                st.warning("Please enter a question before searching.")
            else:
                with st.spinner("Searching policy documents..."):
                    results = rag.retrieve(question.strip(), k=3)

                    # Reject weak matches using the current evaluation threshold.
                    # The threshold separates all 12 supported and 2 unsupported
                    # questions in the current 14-question evaluation dataset.
                    relevance_threshold = 0.32

                    if (
                        results
                        and float(results[0].get("score", 0.0))
                        < relevance_threshold
                    ):
                        results = []

                if not results:
                    st.warning(
                        "No policy passages were retrieved. "
                        "Please contact human support for verification."
                    )
                else:
                    top_score = float(results[0]["score"])

                    if top_score < 0.05:
                        st.warning(
                            "No strong policy match was found. "
                            "Please verify this question with human support."
                        )
                    else:
                        st.success(
                            "Relevant policy passages were retrieved. "
                            "Review the evidence below before responding."
                        )

                    metric_col1, metric_col2 = st.columns(2)

                    metric_col1.metric(
                        "Retrieved passages",
                        len(results),
                    )

                    metric_col2.metric(
                        "Top similarity score",
                        f"{top_score:.3f}",
                    )

                    st.caption(
                        "Similarity scores measure text similarity, "
                        "not answer confidence or policy approval."
                    )

                    st.subheader("Relevant policy excerpts")

                    for index, hit in enumerate(results, start=1):
                        source = hit.get("source", "Unknown source")
                        score = float(hit.get("score", 0.0))
                        excerpt = hit.get("text", "")

                        with st.expander(
                            f"{index}. {source} | Similarity: {score:.3f}",
                            expanded=(index == 1),
                        ):
                            st.markdown("**Source document**")
                            st.code(source, language=None)

                            st.markdown("**Retrieved policy text**")
                            st.write(excerpt)

                    sources = list(
                        dict.fromkeys(
                            hit.get("source", "Unknown source")
                            for hit in results
                        )
                    )

                    st.subheader("Source documents")
                    for source in sources:
                        st.write(f"- {source}")

                    st.warning(
                        "Human verification is required for ambiguous cases, "
                        "refund approvals, and decisions involving customer money."
                    )

        st.divider()

        st.subheader("Copilot capabilities")
        capability_col1, capability_col2, capability_col3 = st.columns(3)

        capability_col1.markdown(
            "**Policy retrieval**\n\n"
            "Searches five policy documents."
        )

        capability_col2.markdown(
            "**Evidence display**\n\n"
            "Shows retrieved passages and similarity scores."
        )

        capability_col3.markdown(
            "**Safe handling**\n\n"
            "Does not execute refunds or modify customer records."
        )

        st.caption(
            "Implementation: PolicyRAG + TF-IDF + cosine similarity. "
            "This page displays retrieved evidence, not a verified "
            "generative AI answer."
        )

    except Exception as exc:
        st.error(f"Unable to load the policy retrieval system: {exc}")
        st.info(
            "Check that the five policy text files are available "
            "under data/policies."
        )

    st.stop()


# Support Ticket Management Page
# --------------------------------------------------

if page == "Support Ticket Management":

    st.title("Support Ticket Management")
    st.caption(
        "Monitor ticket classification, priority, routing and processing "
        "outcomes using the original Part C results."
    )

    support_data = tickets.copy()

    # Preserve missing categories as explicitly unclassified.
    support_data["category_display"] = (
        support_data["category"].fillna("Unclassified")
    )

    # Group original processing statuses.
    def processing_group(status):
        if status in ["classified_json", "classified_keyword"]:
            return "Classified"
        if str(status).startswith("flagged_"):
            return "Flagged"
        if str(status).startswith("skipped_"):
            return "Skipped"
        return "Other"

    support_data["processing_group"] = (
        support_data["status"].apply(processing_group)
    )

    # Sidebar filters
    st.sidebar.divider()
    st.sidebar.subheader("Ticket Filters")

    status_options = sorted(
        support_data["status"].dropna().unique().tolist()
    )
    category_options = sorted(
        support_data["category_display"].unique().tolist()
    )
    priority_options = sorted(
        support_data["priority"].dropna().unique().tolist()
    )
    route_options = sorted(
        support_data["route"].dropna().unique().tolist()
    )

    selected_statuses = st.sidebar.multiselect(
        "Processing Status",
        status_options,
        default=status_options,
        key="support_status_filter",
    )

    selected_categories = st.sidebar.multiselect(
        "Ticket Category",
        category_options,
        default=category_options,
        key="support_category_filter",
    )

    selected_priorities = st.sidebar.multiselect(
        "Priority",
        priority_options,
        default=priority_options,
        key="support_priority_filter",
    )

    selected_routes = st.sidebar.multiselect(
        "Assigned Route",
        route_options,
        default=route_options,
        key="support_route_filter",
    )

    # Apply filters
    filtered_tickets = support_data[
        support_data["status"].isin(selected_statuses)
        & support_data["category_display"].isin(selected_categories)
        & support_data["priority"].isin(selected_priorities)
        & support_data["route"].isin(selected_routes)
    ].copy()

    # KPI metrics
    st.subheader("Support Operations Overview")

    total_count = len(filtered_tickets)
    classified_count = int(
        filtered_tickets["processing_group"].eq("Classified").sum()
    )
    flagged_count = int(
        filtered_tickets["processing_group"].eq("Flagged").sum()
    )
    skipped_count = int(
        filtered_tickets["processing_group"].eq("Skipped").sum()
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Tickets in View", f"{total_count:,}")
    c2.metric("Classified", f"{classified_count:,}")
    c3.metric("Flagged", f"{flagged_count:,}")
    c4.metric("Skipped", f"{skipped_count:,}")

    if total_count:
        st.caption(
            f"Classification rate in current view: "
            f"{classified_count / total_count:.1%}"
        )

    st.divider()

    if filtered_tickets.empty:
        st.warning("No tickets match the selected filters.")
        st.stop()

    # Processing and category charts
    st.subheader("Ticket Processing Analysis")

    left, right = st.columns(2)

    with left:
        processing_summary = (
            filtered_tickets["processing_group"]
            .value_counts()
            .rename_axis("Processing Group")
            .reset_index(name="Tickets")
        )

        fig_processing = px.bar(
            processing_summary,
            x="Processing Group",
            y="Tickets",
            color="Processing Group",
            text="Tickets",
            title="Classified, flagged and skipped tickets",
        )

        fig_processing.update_layout(showlegend=False)
        st.plotly_chart(fig_processing, width="stretch")

    with right:
        category_summary = (
            filtered_tickets["category_display"]
            .value_counts()
            .rename_axis("Category")
            .reset_index(name="Tickets")
        )

        fig_category = px.pie(
            category_summary,
            names="Category",
            values="Tickets",
            hole=0.50,
            title="Ticket category distribution",
        )

        st.plotly_chart(fig_category, width="stretch")

    st.divider()

    # Priority and routing charts
    st.subheader("Priority & Routing Analysis")

    left, right = st.columns(2)

    with left:
        priority_summary = (
            filtered_tickets["priority"]
            .value_counts()
            .rename_axis("Priority")
            .reset_index(name="Tickets")
        )

        fig_priority = px.bar(
            priority_summary,
            x="Priority",
            y="Tickets",
            color="Priority",
            text="Tickets",
            title="Tickets by priority",
        )

        fig_priority.update_layout(showlegend=False)
        st.plotly_chart(fig_priority, width="stretch")

    with right:
        route_summary = (
            filtered_tickets["route"]
            .value_counts()
            .rename_axis("Route")
            .reset_index(name="Tickets")
        )

        fig_route = px.bar(
            route_summary,
            x="Route",
            y="Tickets",
            color="Route",
            text="Tickets",
            title="Tickets by assigned route",
        )

        fig_route.update_layout(showlegend=False)
        st.plotly_chart(fig_route, width="stretch")

    st.divider()

    # Ticket processing method
    st.subheader("Classification Method")

    method_summary = (
        filtered_tickets["status"]
        .value_counts()
        .rename_axis("Processing Status")
        .reset_index(name="Tickets")
    )

    fig_method = px.bar(
        method_summary,
        x="Processing Status",
        y="Tickets",
        color="Processing Status",
        text="Tickets",
        title="Detailed ticket processing outcomes",
    )

    fig_method.update_layout(showlegend=False)
    st.plotly_chart(fig_method, width="stretch")

    st.divider()

    # Ticket search
    st.subheader("Ticket-Level Details")

    search_text = st.text_input(
        "Search by Ticket ID or Customer ID",
        placeholder="Example: T0024 or C0486",
        key="support_ticket_search",
    )

    table_data = filtered_tickets.copy()

    if search_text.strip():
        query = search_text.strip().lower()

        mask = (
            table_data["ticket_id"].astype(str).str.lower().str.contains(
                query, na=False
            )
            | table_data["customer_id"].astype(str).str.lower().str.contains(
                query, na=False
            )
        )

        table_data = table_data[mask]

    display_columns = [
        "ticket_id",
        "customer_id",
        "status",
        "category_display",
        "segment_name",
        "priority",
        "route",
        "tokens_used",
    ]

    st.caption(f"Displaying {len(table_data):,} matching tickets.")

    st.dataframe(
        table_data[display_columns],
        width="stretch",
        hide_index=True,
    )

    csv_data = table_data[display_columns].to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download Filtered Support Ticket Report",
        data=csv_data,
        file_name="instamart_support_ticket_report.csv",
        mime="text/csv",
    )

    st.divider()

    st.caption(
        "Source: outputs/triage_results.csv. "
        "Flagged and skipped tickets are kept separate from successfully "
        "classified tickets. This page is read-only and does not rerun "
        "the original triage pipeline."
    )

    st.stop()


# Churn Prediction Page
# --------------------------------------------------

if page == "Churn Prediction":

    import joblib
    from src.customer_models import load, build_lapse_cohort, split_lapse_data

    st.title("Churn Prediction & Retention Intelligence")
    st.caption(
        "Historical lapse-risk analysis using the saved Part B model "
        "and its original feature preparation methodology."
    )

    @st.cache_resource
    def load_lapse_model():
        return joblib.load(OUTPUT_DIR / "lapse_model.joblib")

    @st.cache_data
    def prepare_heldout_predictions():
        # Reproduce the original Part B cohort and deterministic split.
        model_customers, model_orders = load()
        cohort = build_lapse_cohort(model_customers, model_orders)

        X_train, X_test, y_train, y_test = split_lapse_data(cohort)

        saved_model = load_lapse_model()
        expected_features = list(saved_model.feature_names_in_)

        if list(X_test.columns) != expected_features:
            raise ValueError(
                "Model feature order does not match the original saved model."
            )

        probabilities = saved_model.predict_proba(X_test)
        positive_index = list(saved_model.classes_).index(1)

        predictions = pd.DataFrame({
            "customer_id": cohort.loc[X_test.index, "customer_id"].to_numpy(),
            "actual_lapsed": y_test.to_numpy(),
            "lapse_probability": probabilities[:, positive_index],
        })

        predictions["predicted_lapsed"] = (
            saved_model.predict(X_test)
        )

        return predictions

    try:
        predictions = prepare_heldout_predictions()
    except Exception as exc:
        st.error(f"Unable to prepare churn predictions: {exc}")
        st.stop()

    # Dashboard presentation bands; these are not validated business cutoffs.
    predictions["risk_category"] = pd.cut(
        predictions["lapse_probability"],
        bins=[-0.001, 0.30, 0.60, 1.0],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    ).astype(str)

    # Historical cohort overview
    st.subheader("Held-out Model Evaluation Cohort")

    total_test = len(predictions)
    actual_lapsed = int(predictions["actual_lapsed"].sum())
    actual_not_lapsed = total_test - actual_lapsed
    high_risk_count = int(
        predictions["risk_category"].eq("High").sum()
    )
    average_probability = predictions["lapse_probability"].mean()

    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Held-out Customers", f"{total_test:,}")
    k2.metric("Actually Lapsed", f"{actual_lapsed:,}")
    k3.metric("High Risk Customers", f"{high_risk_count:,}")
    k4.metric("Average Lapse Probability", f"{average_probability:.1%}")

    st.caption(
        "Historical definition: predictors use orders before 1 September 2025; "
        "the lapse outcome is based on completed orders from 1-30 September 2025."
    )

    st.divider()

    # Risk category distribution
    st.subheader("Predicted Risk Distribution")

    risk_order = ["Low", "Medium", "High"]

    risk_summary = (
        predictions["risk_category"]
        .value_counts()
        .reindex(risk_order, fill_value=0)
        .rename_axis("Risk Category")
        .reset_index(name="Customers")
    )

    chart_left, chart_right = st.columns(2)

    with chart_left:
        fig_risk = px.bar(
            risk_summary,
            x="Risk Category",
            y="Customers",
            color="Risk Category",
            text="Customers",
            category_orders={"Risk Category": risk_order},
            title="Customers by predicted risk band",
        )
        fig_risk.update_layout(showlegend=False)
        st.plotly_chart(fig_risk, width="stretch")

    with chart_right:
        fig_probability = px.histogram(
            predictions,
            x="lapse_probability",
            nbins=20,
            title="Distribution of predicted lapse probabilities",
            labels={"lapse_probability": "Predicted lapse probability"},
        )
        st.plotly_chart(fig_probability, width="stretch")

    st.divider()

    # Actual outcomes by predicted risk band
    st.subheader("Actual Lapse Outcomes by Predicted Risk")

    outcome_summary = (
        predictions.groupby("risk_category", observed=False)
        .agg(
            Customers=("customer_id", "count"),
            Actually_Lapsed=("actual_lapsed", "sum"),
            Average_Predicted_Probability=("lapse_probability", "mean"),
        )
        .reindex(risk_order)
        .reset_index()
    )

    outcome_summary["Actual Lapse Rate"] = (
        outcome_summary["Actually_Lapsed"]
        / outcome_summary["Customers"].replace(0, pd.NA)
    )

    outcome_display = outcome_summary.rename(
        columns={"risk_category": "Risk Category"}
    )

    st.dataframe(
        outcome_display.round(3),
        width="stretch",
        hide_index=True,
    )

    st.caption(
        "Actual lapse rate is calculated from the held-out outcomes. "
        "Risk bands are descriptive presentation groups."
    )

    st.divider()

    # Customer-level risk ranking
    st.subheader("Customer-Level Churn Risk")

    selected_risks = st.multiselect(
        "Filter by risk category",
        options=risk_order,
        default=risk_order,
        key="churn_risk_filter",
    )

    risk_view = predictions[
        predictions["risk_category"].isin(selected_risks)
    ].copy()

    risk_view = risk_view.sort_values(
        "lapse_probability",
        ascending=False,
    )

    risk_view["lapse_probability"] = (
        risk_view["lapse_probability"].round(4)
    )

    st.dataframe(
        risk_view,
        width="stretch",
        hide_index=True,
    )

    csv_data = risk_view.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Churn Risk Report",
        data=csv_data,
        file_name="instamart_heldout_churn_predictions.csv",
        mime="text/csv",
    )

    st.divider()

    # Existing illustrative premium retention list
    st.subheader("Existing Part B Retention List")

    retention = pd.read_csv(OUTPUT_DIR / "retention_list.csv")

    st.dataframe(
        retention,
        width="stretch",
        hide_index=True,
    )

    st.warning(
        "The existing retention list contains up to 10 Premium test-set "
        "customers selected using a post-hoc whole-window Premium label. "
        "It is illustrative and is not a leakage-free evaluation."
    )

    st.caption(
        "Model: saved Part B GridSearchCV Random Forest pipeline. "
        "Predictions use the original deterministic held-out split. "
        "No model retraining is performed by this dashboard."
    )

    st.stop()


# Customer Intelligence Page
# --------------------------------------------------

if page == "Customer Intelligence":

    st.title("Customer Intelligence")
    st.caption(
        "Explore customer behaviour, RFM metrics and customer segments."
    )

    # Combine customer demographics with existing RFM segmentation
    customer_view = segments.merge(
        customers,
        on="customer_id",
        how="left",
        validate="one_to_one",
    )

    # Sidebar filters
    st.sidebar.divider()
    st.sidebar.subheader("Customer Filters")

    available_segments = sorted(
        customer_view["segment_name"].dropna().unique().tolist()
    )
    available_cities = sorted(
        customer_view["city"].dropna().unique().tolist()
    )

    selected_segments = st.sidebar.multiselect(
        "Customer Segment",
        options=available_segments,
        default=available_segments,
    )

    selected_cities = st.sidebar.multiselect(
        "City",
        options=available_cities,
        default=available_cities,
    )

    filtered = customer_view[
        customer_view["segment_name"].isin(selected_segments)
        & customer_view["city"].isin(selected_cities)
    ].copy()

    # Summary metrics
    st.subheader("Customer Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Customers in View", f"{len(filtered):,}")

    c2.metric(
        "Average Recency",
        f"{filtered['recency'].mean():.1f} days"
        if not filtered.empty else "N/A",
    )

    c3.metric(
        "Average Frequency",
        f"{filtered['frequency'].mean():.1f}"
        if not filtered.empty else "N/A",
    )

    c4.metric(
        "Average Monetary Value",
        f"\u20B9{filtered['monetary'].mean():,.2f}"
        if not filtered.empty else "N/A",
    )

    st.divider()

    if filtered.empty:
        st.warning("No customers match the selected filters.")
        st.stop()

    # Segment distribution and RFM analysis
    left, right = st.columns(2)

    with left:
        st.subheader("Customer Segment Distribution")

        segment_chart = (
            filtered["segment_name"]
            .value_counts()
            .rename_axis("Segment")
            .reset_index(name="Customers")
        )

        fig_segment = px.pie(
            segment_chart,
            names="Segment",
            values="Customers",
            hole=0.55,
            title="Customers by segment",
        )

        st.plotly_chart(fig_segment, width="stretch")

    with right:
        st.subheader("RFM Behaviour")

        fig_rfm = px.scatter(
            filtered,
            x="recency",
            y="monetary",
            color="segment_name",
            size="frequency",
            hover_data=["customer_id", "city"],
            title="Recency vs Monetary Value",
            labels={
                "recency": "Recency (days)",
                "monetary": "Monetary Value (?)",
                "segment_name": "Segment",
                "frequency": "Order Frequency",
            },
        )

        st.plotly_chart(fig_rfm, width="stretch")

    st.divider()

    # Segment profile
    st.subheader("RFM Segment Summary")

    segment_summary = (
        filtered.groupby("segment_name")
        .agg(
            Customers=("customer_id", "nunique"),
            Avg_Recency=("recency", "mean"),
            Avg_Frequency=("frequency", "mean"),
            Avg_Monetary=("monetary", "mean"),
        )
        .reset_index()
        .rename(columns={"segment_name": "Segment"})
    )

    st.dataframe(
        segment_summary.round(2),
        width="stretch",
        hide_index=True,
    )

    st.divider()

    # Customer demographics
    st.subheader("Customer Demographics")

    demo_left, demo_right = st.columns(2)

    with demo_left:
        city_summary = (
            filtered["city"]
            .value_counts()
            .rename_axis("City")
            .reset_index(name="Customers")
        )

        fig_city = px.bar(
            city_summary,
            x="City",
            y="Customers",
            color="City",
            text="Customers",
            title="Customers by city",
        )

        fig_city.update_layout(showlegend=False)
        st.plotly_chart(fig_city, width="stretch")

    with demo_right:
        gender_summary = (
            filtered["gender"]
            .fillna("Unknown")
            .value_counts()
            .rename_axis("Gender")
            .reset_index(name="Customers")
        )

        fig_gender = px.pie(
            gender_summary,
            names="Gender",
            values="Customers",
            hole=0.45,
            title="Customer gender distribution",
        )

        st.plotly_chart(fig_gender, width="stretch")

    st.divider()

    # Customer-level details
    st.subheader("Customer Details")

    display_columns = [
        "customer_id",
        "city",
        "age",
        "gender",
        "segment_name",
        "recency",
        "frequency",
        "monetary",
        "is_premium",
    ]

    available_columns = [
        col for col in display_columns if col in filtered.columns
    ]

    customer_table = filtered[available_columns].sort_values(
        "monetary",
        ascending=False,
    )

    st.dataframe(
        customer_table,
        width="stretch",
        hide_index=True,
    )

    csv_data = customer_table.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Filtered Customer Report",
        data=csv_data,
        file_name="instamart_filtered_customers.csv",
        mime="text/csv",
    )

    st.caption(
        "RFM values are sourced from the existing Part B customer "
        "segmentation output. Filters affect all charts and the report."
    )

    st.stop()


# Executive Dashboard
# --------------------------------------------------

st.title("Instamart Customer Intelligence")
st.caption(
    "Executive overview of customer behaviour, orders, "
    "retention and customer support."
)

# KPI calculations
total_customers = customers["customer_id"].nunique()
total_orders = orders["order_id"].nunique()

completed_orders = orders.loc[
    orders["status"].eq("completed")
]
cancelled_orders = orders.loc[
    orders["status"].eq("cancelled")
]

completed_revenue = completed_orders["order_amount"].sum()
recorded_order_amount = orders["order_amount"].sum()

total_tickets = len(tickets)
classified_tickets = tickets["status"].isin(
    ["classified_json", "classified_keyword"]
).sum()

# KPI cards
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers", f"{total_customers:,}")
col2.metric("Total Orders", f"{total_orders:,}")
col3.metric("Completed Order Value", f"₹{completed_revenue:,.0f}")
col4.metric("Support Tickets", f"{total_tickets:,}")

st.divider()

# Secondary metrics
st.subheader("Business Performance")

m1, m2, m3, m4 = st.columns(4)

m1.metric("Completed Orders", f"{len(completed_orders):,}")
m2.metric("Cancelled Orders", f"{len(cancelled_orders):,}")
m3.metric(
    "Average Order Amount",
    f"₹{orders['order_amount'].mean():,.2f}",
)
m4.metric(
    "Classified Tickets",
    f"{classified_tickets:,}",
    help="Includes JSON-classified and keyword-classified tickets.",
)

st.caption(
    f"Total recorded order amount across all statuses: "
    f"₹{recorded_order_amount:,.2f}. "
    "Completed order value excludes cancelled orders."
)

st.divider()

# Charts
left, right = st.columns(2)

with left:
    st.subheader("Customer Segments")

    segment_counts = (
        segments["segment_name"]
        .value_counts()
        .rename_axis("Segment")
        .reset_index(name="Customers")
    )

    fig_segments = px.pie(
        segment_counts,
        names="Segment",
        values="Customers",
        hole=0.55,
        title="Customer distribution by segment",
    )

    fig_segments.update_layout(
        legend_title_text="Segment",
        margin=dict(t=55, b=15, l=10, r=10),
    )

    st.plotly_chart(fig_segments, width="stretch")

with right:
    st.subheader("Order Status")

    status_counts = (
        orders["status"]
        .value_counts()
        .rename_axis("Status")
        .reset_index(name="Orders")
    )

    fig_status = px.bar(
        status_counts,
        x="Status",
        y="Orders",
        color="Status",
        text="Orders",
        title="Completed versus cancelled orders",
    )

    fig_status.update_layout(
        showlegend=False,
        margin=dict(t=55, b=15, l=10, r=10),
    )

    st.plotly_chart(fig_status, width="stretch")

st.divider()

# Monthly order trend
st.subheader("Monthly Order Trend")

monthly_orders = orders.dropna(subset=["order_ts"]).copy()

if not monthly_orders.empty:
    monthly_orders["Month"] = (
        monthly_orders["order_ts"].dt.to_period("M").astype(str)
    )

    monthly_summary = (
        monthly_orders.groupby("Month")
        .agg(
            Orders=("order_id", "nunique"),
            Order_Amount=("order_amount", "sum"),
        )
        .reset_index()
    )

    fig_monthly = px.line(
        monthly_summary,
        x="Month",
        y="Orders",
        markers=True,
        title="Number of orders by month",
    )

    fig_monthly.update_layout(
        xaxis_title="Month",
        yaxis_title="Orders",
        margin=dict(t=55, b=15, l=10, r=10),
    )

    st.plotly_chart(fig_monthly, width="stretch")
else:
    st.info("No valid order timestamps are available for this chart.")

st.divider()

# Support overview
st.subheader("Customer Support Overview")

support_left, support_right = st.columns(2)

with support_left:
    category_data = (
        tickets["category"]
        .fillna("Unclassified")
        .value_counts()
        .rename_axis("Category")
        .reset_index(name="Tickets")
    )

    fig_categories = px.bar(
        category_data,
        x="Category",
        y="Tickets",
        color="Category",
        text="Tickets",
        title="Tickets by category",
    )

    fig_categories.update_layout(
        showlegend=False,
        margin=dict(t=55, b=15, l=10, r=10),
    )

    st.plotly_chart(fig_categories, width="stretch")

with support_right:
    route_data = (
        tickets["route"]
        .value_counts()
        .rename_axis("Route")
        .reset_index(name="Tickets")
    )

    fig_routes = px.bar(
        route_data,
        x="Route",
        y="Tickets",
        color="Route",
        text="Tickets",
        title="Support routing distribution",
    )

    fig_routes.update_layout(
        showlegend=False,
        margin=dict(t=55, b=15, l=10, r=10),
    )

    st.plotly_chart(fig_routes, width="stretch")

st.divider()

# Refund agent evaluation
summary_path = OUTPUT_DIR / "agent_scenario_summary.json"

if summary_path.exists():
    with summary_path.open("r", encoding="utf-8") as file:
        scenarios = json.load(file)

    total_scenarios = len(scenarios)
    matched_scenarios = sum(
        bool(item.get("outcome_match", False))
        for item in scenarios
    )

    total_tokens = sum(
        int(item.get("tokens_used", 0))
        for item in scenarios
    )

    st.subheader("Refund Agent Evaluation")

    a, b, c = st.columns(3)
    a.metric("Scenarios Evaluated", total_scenarios)
    b.metric("Expected Outcome Matches", f"{matched_scenarios}/{total_scenarios}")
    c.metric("Total Tokens Used", f"{total_tokens:,}")

    st.caption(
        "These are results from the recorded project scenarios, "
        "not a guarantee of performance on new customer requests."
    )

# Footer
st.divider()
st.caption(
    "Instamart Customer Intelligence & Support Copilot | "
    "Portfolio Edition | Built with Python, Streamlit, Pandas and Plotly"
)
