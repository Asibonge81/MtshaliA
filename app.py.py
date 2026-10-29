
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from pathlib import Path
import re

# Configure the application
st.set_page_config(
    page_title="Ekurhuleni Women's Safety",
    page_icon="🛡️",
    layout="wide"
)

# Dashboard heading
st.title("Ekurhuleni Women's Safety Dashboard")

st.write(
    "An interactive machine learning solution for "
    "identifying reported-crime hotspots and analysing "
    "safety-related crime patterns in Ekurhuleni."
)

st.info(
    "This dashboard uses reported crime statistics "
    "to support safety planning and prevention-resource "
    "prioritisation. These are not women-only victim counts "
    "or precise crime locations."
)

BASE = Path(__file__).resolve().parent
DATA_FOLDER = BASE / "datasets"
MODEL_FOLDER = BASE / "models"

# Individual crime categories from the SAPS datasets.
# Aggregate categories are excluded to avoid double counting.
CATEGORIES = [
    "Abduction",
    "Assault with the intent to inflict grievous bodily harm",
    "Attempted murder",
    "Attempted sexual offences",
    "Common assault",
    "Common robbery",
    "Contact sexual offences",
    "Kidnapping",
    "Murder",
    "Rape",
    "Robbery with aggravating circumstances",
    "Sexual assault",
]

WORKBOOKS = [
    (
        "2025-2026_-_4th_Quarter_WEB.xlsx",
        "Jan-Mar",
        "January",
        "March"
    ),
    (
        "2026-2027_-_1st_Quarter_WEB.xlsm",
        "Apr-Jun",
        "April",
        "June"
    ),
]



@st.cache_data(show_spinner="Loading SAPS crime data...")
def load_data():
    parts = []

    for filename, quarter, start_month, end_month in WORKBOOKS:
        path = DATA_FOLDER / filename

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # The RAW Data worksheet uses Excel row 3 as its header.
        # Columns M, Q, U, Y and AC contain quarterly totals.
        df = pd.read_excel(
            path,
            sheet_name="RAW Data",
            header=2,
            usecols="E:H,M,Q,U,Y,AC",
            engine="openpyxl"
        )

        df.columns = df.columns.astype(str).str.strip()

        required = {
            "Station",
            "District",
            "Province",
            "Crime_Category"
        }

        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            raise ValueError(
                f"{filename} is missing {sorted(missing)}"
            )

        # Filter for Ekurhuleni and selected crime categories
        df = df.loc[
            (
                df["District"]
                .astype(str)
                .str.strip()
                .eq("Ekurhuleni District")
            )
            &
            (
                df["Crime_Category"]
                .astype(str)
                .str.strip()
                .isin(CATEGORIES)
            )
        ].copy()

        # Identify quarterly total columns
        year_cols = {}

        pattern = re.compile(
            rf"^{start_month}\s+(20\d{{2}})\s+to\s+{end_month}\s+\1$",
            re.IGNORECASE
        )

        for col in df.columns:
            normalized = " ".join(str(col).split())
            match = pattern.fullmatch(normalized)

            if match:
                year_cols[col] = int(match.group(1))

        if len(year_cols) != 5:
            raise ValueError(
                f"{filename}: expected 5 quarterly total columns, "
                f"found {len(year_cols)}"
            )

        # Convert wide Excel data into long format
        tidy = df.melt(
            id_vars=["Station", "Crime_Category"],
            value_vars=list(year_cols),
            var_name="SourcePeriod",
            value_name="Cases"
        )

        tidy["Year"] = (
            tidy["SourcePeriod"]
            .map(year_cols)
            .astype(int)
        )

        tidy["Quarter"] = quarter

        tidy["Cases"] = pd.to_numeric(
            tidy["Cases"],
            errors="coerce"
        )

        tidy = tidy.dropna(
            subset=["Station", "Crime_Category", "Cases"]
        )

        tidy["Station"] = (
            tidy["Station"].astype(str).str.strip()
        )

        tidy["Crime_Category"] = (
            tidy["Crime_Category"].astype(str).str.strip()
        )

        parts.append(
            tidy[
                [
                    "Station",
                    "Crime_Category",
                    "Year",
                    "Quarter",
                    "Cases"
                ]
            ]
        )

    result = pd.concat(parts, ignore_index=True)

    return result.groupby(
        ["Station", "Crime_Category", "Year", "Quarter"],
        as_index=False
    )["Cases"].sum()


try:
    data = load_data()

except (OSError, ValueError, ImportError, KeyError) as error:
    st.error(
        f"Could not load the original SAPS datasets: {error}"
    )
    st.stop()


if data.empty:
    st.error(
        "The SAPS workbooks contain no matching Ekurhuleni records."
    )
    st.stop()



st.sidebar.header("Filter Crime Data")

all_stations = sorted(data["Station"].unique())
all_categories = sorted(data["Crime_Category"].unique())

stations = st.sidebar.multiselect(
    "Police Station",
    all_stations,
    default=all_stations
)

crime_types = st.sidebar.multiselect(
    "Crime Category",
    all_categories,
    default=all_categories
)

min_year = int(data["Year"].min())
max_year = int(data["Year"].max())

years = st.sidebar.slider(
    "Year",
    min_year,
    max_year,
    (min_year, max_year)
)

quarters = st.sidebar.multiselect(
    "Quarter",
    ["Jan-Mar", "Apr-Jun"],
    default=["Jan-Mar", "Apr-Jun"]
)

filtered = data.loc[
    data["Station"].isin(stations)
    &
    data["Crime_Category"].isin(crime_types)
    &
    data["Year"].between(*years)
    &
    data["Quarter"].isin(quarters)
].copy()

st.sidebar.caption(
    f"{len(filtered):,} matching data records"
)

st.sidebar.download_button(
    "Download filtered CSV",
    filtered.to_csv(index=False).encode("utf-8"),
    file_name="ekurhuleni_filtered_data.csv",
    mime="text/csv"
)



tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",
    "Crime Patterns",
    "Hotspot Classification",
    "Compare 3 Hotspots"
])



# SIMPLE  TAB 1: OVERVIEW

with tab1:
    st.header("Crime Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Reported Cases",
        f"{filtered['Cases'].sum():,.0f}"
    )

    col2.metric(
        "Police Stations",
        filtered["Station"].nunique()
    )

    col3.metric(
        "Crime Categories",
        filtered["Crime_Category"].nunique()
    )

    if filtered.empty:
        st.warning(
            "No records match the current filters."
        )

    else:
        station_totals = (
            filtered.groupby("Station", as_index=False)["Cases"]
            .sum()
            .sort_values("Cases", ascending=False)
        )

        top_n = st.slider(
            "Show top police stations",
            3,
            min(20, len(station_totals)),
            min(10, len(station_totals)),
            key="overview_top"
        )

        fig = px.bar(
            station_totals.head(top_n),
            x="Station",
            y="Cases",
            title="Police stations with the most reported cases"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        category_totals = (
            filtered.groupby(
                "Crime_Category",
                as_index=False
            )["Cases"]
            .sum()
            .sort_values("Cases", ascending=False)
        )

        fig = px.bar(
            category_totals,
            x="Cases",
            y="Crime_Category",
            orientation="h",
            title="Reported cases by crime category"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader("Explore the Dataset")

        st.dataframe(
            filtered,
            use_container_width=True,
            hide_index=True
        )



# TAB 2: WIL SHOW CRIME PATTERNS

with tab2:
    st.header("Crime Patterns")

    if filtered.empty:
        st.warning(
            "Choose stations and crime categories to view trends."
        )

    else:
        breakdown = st.radio(
            "Analyse patterns by",
            [
                "All selected data",
                "One police station",
                "One crime category"
            ],
            horizontal=True
        )

        pattern_data = filtered

        if breakdown == "One police station":
            chosen_station = st.selectbox(
                "Select police station",
                sorted(filtered["Station"].unique()),
                key="pattern_station"
            )

            pattern_data = filtered[
                filtered["Station"] == chosen_station
            ]

        elif breakdown == "One crime category":
            chosen_crime = st.selectbox(
                "Select crime category",
                sorted(filtered["Crime_Category"].unique()),
                key="pattern_crime"
            )

            pattern_data = filtered[
                filtered["Crime_Category"] == chosen_crime
            ]

        # Crime trends over time
        trend = pattern_data.groupby(
            ["Year", "Quarter"],
            as_index=False
        )["Cases"].sum()

        trend["QuarterOrder"] = trend["Quarter"].map({
            "Jan-Mar": 1,
            "Apr-Jun": 2
        })

        trend = trend.sort_values(
            ["Year", "QuarterOrder"]
        )

        trend["Period"] = (
            trend["Year"].astype(str)
            + " "
            + trend["Quarter"]
        )

        fig = px.line(
            trend,
            x="Period",
            y="Cases",
            markers=True,
            title="Reported crime over time"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # Year and quarter comparison
        by_quarter = pattern_data.groupby(
            ["Year", "Quarter"],
            as_index=False
        )["Cases"].sum()

        fig = px.bar(
            by_quarter,
            x="Year",
            y="Cases",
            color="Quarter",
            barmode="group",
            title="Yearly comparison by quarter"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # Crime heatmap
        matrix = pattern_data.pivot_table(
            index="Station",
            columns="Crime_Category",
            values="Cases",
            aggfunc="sum",
            fill_value=0
        )

        fig = px.imshow(
            matrix,
            aspect="auto",
            color_continuous_scale="Blues",
            title="Crime-category heatmap by police station"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# -------------------------------------------------
# TAB 3: HOTSPOT CLASSIFICATION
# -------------------------------------------------

with tab3:
    st.header("Hotspot Classification")

    st.write(
        "A station is classified as an observed hotspot when "
        "its total reported cases are in the highest 25% of "
        "stations for the selected year and quarter. "
        "This is a relative station ranking."
    )

    if filtered.empty:
        st.warning(
            "Select at least one police station, "
            "crime category and quarter."
        )

    else:
        year_choice = st.selectbox(
            "Select Year",
            sorted(
                filtered["Year"].unique(),
                reverse=True
            ),
            key="hotspot_year"
        )

        quarter_options = [
            q for q in ["Jan-Mar", "Apr-Jun"]
            if q in filtered.loc[
                filtered["Year"] == year_choice,
                "Quarter"
            ].unique()
        ]

        quarter_choice = st.selectbox(
            "Select Quarter",
            quarter_options,
            key="hotspot_quarter"
        )

        selected = filtered.loc[
            (filtered["Year"] == year_choice)
            &
            (filtered["Quarter"] == quarter_choice)
        ]

        ranking = (
            selected.groupby(
                "Station",
                as_index=False
            )["Cases"]
            .sum()
            .sort_values(
                "Cases",
                ascending=False
            )
        )

        cutoff = ranking["Cases"].quantile(0.75)

        ranking["Hotspot"] = (
            ranking["Cases"] >= cutoff
        )

        ranking["Classification"] = (
            ranking["Hotspot"].map({
                True: "Hotspot",
                False: "Other station"
            })
        )

        m1, m2 = st.columns(2)

        m1.metric(
            "Observed Hotspots",
            int(ranking["Hotspot"].sum())
        )

        m2.metric(
            "Top-quartile threshold",
            f"{cutoff:,.0f} cases"
        )

        fig = px.bar(
            ranking,
            x="Station",
            y="Cases",
            color="Classification",
            title=(
                "Observed hotspot classification — "
                f"{quarter_choice} {year_choice}"
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.dataframe(
            ranking[
                [
                    "Station",
                    "Cases",
                    "Classification"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

    # Previously evaluated model performance
    st.subheader(
        "Previously Evaluated Random Forest Model"
    )

    st.caption(
        "These are previously reported test-set results, "
        "not model scores recomputed using dashboard filters."
    )

    a, b, c = st.columns(3)

    a.metric("Test Accuracy", "92.31%")
    b.metric("Hotspot Recall", "85%")
    c.metric("Hotspot F1", "85%")

    confusion = pd.DataFrame(
        [
            [55, 3],
            [3, 17]
        ],
        index=[
            "Actual Other",
            "Actual Hotspot"
        ],
        columns=[
            "Predicted Other",
            "Predicted Hotspot"
        ]
    )

    fig = px.imshow(
        confusion,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Random Forest test-set confusion matrix"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    model_path = (
        MODEL_FOLDER / "ekurhuleni_hotspot_model.pkl"
    )

    features_path = (
        MODEL_FOLDER / "model_features.pkl"
    )

    if model_path.exists() and features_path.exists():
        st.success(
            "Saved model and feature files were found "
            "in the models folder."
        )

    else:
        st.caption(
            "Saved model files are optional for viewing "
            "the historical analysis."
        )


# -------------------------------------------------
# TAB 4: COMPARE 3 HOTSPOTS
# -------------------------------------------------

with tab4:
    st.header("Three Hotspot Comparison")

    if filtered["Station"].nunique() < 3:
        st.warning(
            "Select at least three stations using the sidebar."
        )

    else:
        totals = (
            filtered.groupby("Station")["Cases"]
            .sum()
            .sort_values(ascending=False)
        )

        chosen = st.multiselect(
            "Choose exactly three police stations",
            options=sorted(totals.index),
            default=totals.head(3).index.tolist(),
            max_selections=3
        )

        if len(chosen) != 3:
            st.warning(
                "Choose three stations to compare."
            )

        else:
            comparison = filtered[
                filtered["Station"].isin(chosen)
            ]

            summary = (
                comparison.groupby(
                    "Station",
                    as_index=False
                )["Cases"]
                .sum()
                .sort_values(
                    "Cases",
                    ascending=False
                )
            )

            # Compare total cases
            st.plotly_chart(
                px.bar(
                    summary,
                    x="Station",
                    y="Cases",
                    title="Total reported cases by station"
                ),
                use_container_width=True
            )

            # Compare crime trends
            timeline = comparison.groupby(
                ["Station", "Year", "Quarter"],
                as_index=False
            )["Cases"].sum()

            timeline["QuarterOrder"] = (
                timeline["Quarter"].map({
                    "Jan-Mar": 1,
                    "Apr-Jun": 2
                })
            )

            timeline = timeline.sort_values(
                ["Year", "QuarterOrder"]
            )

            timeline["Period"] = (
                timeline["Year"].astype(str)
                + " "
                + timeline["Quarter"]
            )

            st.plotly_chart(
                px.line(
                    timeline,
                    x="Period",
                    y="Cases",
                    color="Station",
                    markers=True,
                    title=(
                        "Reported crime trends: "
                        "three-station comparison"
                    )
                ),
                use_container_width=True
            )

            # Compare crime categories
            by_category = comparison.groupby(
                ["Station", "Crime_Category"],
                as_index=False
            )["Cases"].sum()

            st.plotly_chart(
                px.bar(
                    by_category,
                    x="Station",
                    y="Cases",
                    color="Crime_Category",
                    barmode="stack",
                    title="Crime-category comparison"
                ),
                use_container_width=True
            )

            st.dataframe(
                summary,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "Download comparison",
                summary.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="three_hotspots_comparison.csv",
                mime="text/csv"
            )
