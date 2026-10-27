"""Brand Traders – Sales Analytics Dashboard (Streamlit)

Run:  streamlit run app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import analytics as an

st.set_page_config(page_title="Brand Traders – Sales Analytics", page_icon="📊", layout="wide")
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = px.colors.qualitative.Safe

# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _stretch_supported() -> bool:
    try:
        major, minor = (int(x) for x in st.__version__.split(".")[:2])
        return (major, minor) >= (1, 50)
    except Exception:
        return False


_STRETCH = _stretch_supported()
_chart_n = 0


def show(fig, height=None):
    """Render a plotly figure full-width, compatible across Streamlit versions."""
    global _chart_n
    _chart_n += 1
    if height:
        fig.update_layout(height=height)
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    if _STRETCH:
        st.plotly_chart(fig, width="stretch", key=f"chart_{_chart_n}")
    else:
        st.plotly_chart(fig, use_container_width=True, key=f"chart_{_chart_n}")


def table(data, **kwargs):
    if _STRETCH:
        st.dataframe(data, width="stretch", hide_index=True, **kwargs)
    else:
        st.dataframe(data, use_container_width=True, hide_index=True, **kwargs)


def money(x):
    return f"${x:,.0f}"


def money_axis(fig, axis="y"):
    getattr(fig, f"update_{axis}axes")(tickprefix="$", tickformat=",.0f")
    return fig


# --------------------------------------------------------------------------- #
# Cached data layer
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading CSV files…")
def cached_folder(folder: str):
    return an.load_folder(folder)


@st.cache_data(show_spinner="Reading uploads…")
def cached_uploads(files: dict):
    return an.load_uploads(files)


@st.cache_data(show_spinner="Joining tables…")
def cached_master(tables: dict):
    return an.build_master(tables)


def default_folder() -> str:
    here = Path(__file__).parent
    for p in (here / "data", here, Path("/mnt/user-data/uploads")):
        if all((p / f"{n}.csv").exists() for n in an.TABLE_NAMES):
            return str(p)
    return str(here / "data")


# --------------------------------------------------------------------------- #
# Sidebar – data source
# --------------------------------------------------------------------------- #
st.sidebar.title("📊 Brand Traders")
source = st.sidebar.radio("Data source", ["CSV folder", "Upload CSV files"], horizontal=True)

tables = None
try:
    if source == "CSV folder":
        folder = st.sidebar.text_input("Folder containing the 7 CSVs", default_folder())
        tables = cached_folder(folder)
    else:
        ups = st.sidebar.file_uploader(
            "Upload all 7 files (" + ", ".join(an.TABLE_NAMES) + ")",
            type="csv", accept_multiple_files=True)
        if ups:
            tables = cached_uploads({u.name: u.getvalue() for u in ups})
except FileNotFoundError as e:
    st.sidebar.error(str(e))

if tables is None:
    st.title("Brand Traders – Sales Analytics")
    st.info("Point the sidebar to the folder with your CSVs (or upload them) to begin.")
    st.stop()

master = cached_master(tables)

# --------------------------------------------------------------------------- #
# Sidebar – filters
# --------------------------------------------------------------------------- #
st.sidebar.divider()
st.sidebar.subheader("Filters")

dmin, dmax = master["orderDate"].min().date(), master["orderDate"].max().date()
date_range = st.sidebar.date_input("Order date range", (dmin, dmax), min_value=dmin, max_value=dmax)
if not (isinstance(date_range, (tuple, list)) and len(date_range) == 2):
    st.info("Pick an end date to apply the range.")
    st.stop()
d0, d1 = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])


def multi(label, col):
    opts = sorted(master[col].dropna().unique())
    return st.sidebar.multiselect(label, opts, default=opts)


sel_country = multi("Customer country", "customer_country")
sel_cat = multi("Product category", "categoryName")
sel_emp = multi("Sales employee", "employeeName")
sel_ship = multi("Shipper", "shipper")
grain = st.sidebar.radio("Trend granularity", ["Month", "Quarter"], horizontal=True)
gcol = "month" if grain == "Month" else "quarter"

mask = (
    master["orderDate"].between(d0, d1)
    & master["customer_country"].isin(sel_country)
    & master["categoryName"].isin(sel_cat)
    & master["employeeName"].isin(sel_emp)
    & master["shipper"].isin(sel_ship)
)
df = master[mask].copy()
if df.empty:
    st.warning("No data matches the current filters.")
    st.stop()

ol = an.order_level(df)
K = an.kpis(df, ol)

st.sidebar.caption(
    f"{len(df):,} order lines · {K['orders']:,} orders\n\n"
    "ℹ️ Freight is stored per order; if you filter by category, order revenue "
    "only includes the selected lines while freight stays whole-order."
)

# --------------------------------------------------------------------------- #
# Header + KPIs
# --------------------------------------------------------------------------- #
st.title("Brand Traders – Sales Analytics")
st.caption(f"Orders from {d0:%d %b %Y} to {d1:%d %b %Y} · all 7 tables joined into a single order-line model")

r1 = st.columns(5)
r1[0].metric("Net revenue", money(K["revenue"]))
r1[1].metric("Orders", f"{K['orders']:,}")
r1[2].metric("Customers", f"{K['customers']:,}")
r1[3].metric("Avg order value", money(K["aov"]))
r1[4].metric("Units sold", f"{K['units']:,}")
r2 = st.columns(4)
r2[0].metric("Freight cost", money(K["freight"]))
r2[1].metric("Avg discount rate", f"{K['discount_rate']:.1%}")
r2[2].metric("On-time shipments", "n/a" if np.isnan(K["on_time"]) else f"{K['on_time']:.1%}")
r2[3].metric("Avg days to ship", "n/a" if np.isnan(K["avg_delivery"]) else f"{K['avg_delivery']:.1f}")

tabs = st.tabs([
    "🏠 Overview", "📦 Products & Categories", "👥 Customers & Markets",
    "🧑‍💼 Employees", "🚚 Shipping", "🏷️ Discounts", "🗂️ Data Model & Explorer",
])

# =========================================================================== #
# 1. Overview
# =========================================================================== #
with tabs[0]:
    st.subheader("Key insights")
    for line in an.build_insights(df, ol):
        st.markdown(f"- {line}")

    st.subheader("Revenue trend")
    trend = df.groupby(gcol).agg(revenue=("revenue", "sum"), orders=("orderID", "nunique")).reset_index()
    trend["aov"] = trend["revenue"] / trend["orders"]
    trend["rolling"] = trend["revenue"].rolling(3, min_periods=1).mean()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=trend[gcol], y=trend["revenue"], name="Revenue", marker_color="#4C78A8", opacity=0.75)
    fig.add_scatter(x=trend[gcol], y=trend["rolling"], name="3-period average",
                    mode="lines", line=dict(color="#F58518", width=3))
    fig.add_scatter(x=trend[gcol], y=trend["orders"], name="Orders", mode="lines+markers",
                    line=dict(color="#54A24B", dash="dot"), secondary_y=True)
    fig.update_yaxes(title_text="Revenue ($)", tickprefix="$", tickformat=",.0f", secondary_y=False)
    fig.update_yaxes(title_text="Orders", secondary_y=True)
    fig.update_layout(title=f"Revenue & orders by {grain.lower()}", legend=dict(orientation="h", y=-0.15))
    show(fig, 420)
    st.caption("The first and last periods in the dataset are partial (data starts 4 Jul 2013 and ends 6 May 2015).")

    c1, c2 = st.columns(2)
    with c1:
        yr = df.assign(m=df["orderDate"].dt.month).groupby(["year", "m"])["revenue"].sum().reset_index()
        yr["year"] = yr["year"].astype(str)
        f = px.line(yr, x="m", y="revenue", color="year", markers=True,
                    title="Seasonality – revenue by calendar month", labels={"m": "Month", "revenue": "Revenue"})
        f.update_xaxes(dtick=1)
        show(money_axis(f), 380)
    with c2:
        wd_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        wd = ol.assign(weekday=ol["orderDate"].dt.day_name()).groupby("weekday").agg(
            orders=("orderID", "count")).reindex(wd_order).dropna().reset_index()
        f = px.bar(wd, x="weekday", y="orders", title="Orders placed by weekday", text="orders")
        show(f, 380)

    c3, c4 = st.columns(2)
    with c3:
        f = px.line(trend, x=gcol, y="aov", markers=True, title=f"Average order value by {grain.lower()}")
        show(money_axis(f), 350)
    with c4:
        ys = df.groupby("year").agg(revenue=("revenue", "sum"), orders=("orderID", "nunique")).reset_index()
        ys["year"] = ys["year"].astype(str)
        f = px.bar(ys, x="year", y="revenue", text_auto=",.0f", title="Revenue by year (partial years)")
        show(money_axis(f), 350)

# =========================================================================== #
# 2. Products & Categories
# =========================================================================== #
with tabs[1]:
    c1, c2 = st.columns(2)
    cat = df.groupby("categoryName").agg(
        revenue=("revenue", "sum"), units=("quantity", "sum"),
        orders=("orderID", "nunique")).reset_index().sort_values("revenue", ascending=False)
    with c1:
        f = px.pie(cat, names="categoryName", values="revenue", hole=0.45, title="Revenue share by category")
        f.update_traces(textposition="inside", textinfo="percent+label")
        show(f, 420)
    with c2:
        f = px.bar(cat, x="categoryName", y="revenue", color="categoryName", text_auto=",.0f",
                   title="Revenue by category")
        f.update_layout(showlegend=False)
        show(money_axis(f), 420)

    st.markdown("#### Category performance over time")
    ct = df.groupby([gcol, "categoryName"])["revenue"].sum().reset_index()
    f = px.area(ct, x=gcol, y="revenue", color="categoryName", title=f"Revenue by category per {grain.lower()}")
    show(money_axis(f), 400)

    st.markdown("#### Product ranking")
    n = st.slider("Number of products to show", 5, 30, 10)
    prod = df.groupby(["productName", "categoryName", "status"]).agg(
        revenue=("revenue", "sum"), units=("quantity", "sum"),
        orders=("orderID", "nunique"), avg_discount=("discount", "mean")).reset_index()
    c1, c2 = st.columns(2)
    with c1:
        top = prod.nlargest(n, "revenue").sort_values("revenue")
        f = px.bar(top, x="revenue", y="productName", color="categoryName", orientation="h",
                   title=f"Top {n} products by revenue")
        show(money_axis(f, "x"), 120 + 28 * n)
    with c2:
        bot = prod.nsmallest(n, "revenue").sort_values("revenue", ascending=False)
        f = px.bar(bot, x="revenue", y="productName", color="categoryName", orientation="h",
                   title=f"Bottom {n} products by revenue")
        show(money_axis(f, "x"), 120 + 28 * n)

    c1, c2 = st.columns(2)
    with c1:
        f = px.treemap(df, path=[px.Constant("All"), "categoryName", "productName"], values="revenue",
                       title="Revenue treemap – category → product")
        show(f, 480)
    with c2:
        f = px.scatter(prod, x="units", y="revenue", size="orders", color="categoryName",
                       hover_name="productName", title="Units vs revenue (bubble = # orders)",
                       labels={"units": "Units sold", "revenue": "Revenue"})
        show(money_axis(f), 480)

    c1, c2 = st.columns(2)
    with c1:
        st_rev = df.groupby("status")["revenue"].sum().reset_index()
        f = px.pie(st_rev, names="status", values="revenue", hole=0.5,
                   title="Revenue: active vs discontinued products")
        show(f, 340)
    with c2:
        st.markdown("**Product table**")
        table(prod.sort_values("revenue", ascending=False).round(2),
              column_config={"revenue": st.column_config.NumberColumn(format="$%.0f"),
                             "avg_discount": st.column_config.NumberColumn("avg_discount", format="%.3f")})

# =========================================================================== #
# 3. Customers & Markets
# =========================================================================== #
with tabs[2]:
    country = df.groupby(["customer_country", "customer_country_map"]).agg(
        revenue=("revenue", "sum"), orders=("orderID", "nunique"),
        customers=("customerID", "nunique")).reset_index()
    country["aov"] = country["revenue"] / country["orders"]

    c1, c2 = st.columns([3, 2])
    with c1:
        f = px.choropleth(country, locations="customer_country_map", locationmode="country names",
                          color="revenue", hover_name="customer_country",
                          hover_data={"customer_country_map": False, "orders": True, "customers": True},
                          color_continuous_scale="Blues", title="Revenue by customer country")
        f.update_geos(fitbounds="locations", showframe=False, showcountries=True)
        show(f, 450)
    with c2:
        f = px.bar(country.nlargest(12, "revenue").sort_values("revenue"), x="revenue",
                   y="customer_country", orientation="h", title="Top countries by revenue", text_auto=",.0f")
        show(money_axis(f, "x"), 450)

    cust = df.groupby(["customerID", "customer", "customer_country"]).agg(
        revenue=("revenue", "sum"), orders=("orderID", "nunique"),
        units=("quantity", "sum"), last_order=("orderDate", "max")).reset_index()
    cust["aov"] = cust["revenue"] / cust["orders"]

    c1, c2 = st.columns(2)
    with c1:
        n = st.slider("Top customers to show", 5, 25, 10, key="topc")
        top = cust.nlargest(n, "revenue").sort_values("revenue")
        f = px.bar(top, x="revenue", y="customer", color="customer_country", orientation="h",
                   title=f"Top {n} customers")
        show(money_axis(f, "x"), 120 + 28 * n)
    with c2:
        p = cust.sort_values("revenue", ascending=False).reset_index(drop=True)
        p["cum_pct"] = p["revenue"].cumsum() / p["revenue"].sum()
        p["rank"] = np.arange(1, len(p) + 1)
        n80 = int((p["cum_pct"] < 0.8).sum() + 1)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_bar(x=p["rank"], y=p["revenue"], name="Revenue", marker_color="#4C78A8")
        fig.add_scatter(x=p["rank"], y=p["cum_pct"], name="Cumulative %", mode="lines",
                        line=dict(color="#E45756", width=3), secondary_y=True)
        fig.update_yaxes(tickformat=".0%", range=[0, 1.05], secondary_y=True)
        fig.update_yaxes(tickprefix="$", tickformat=",.0f", secondary_y=False)
        fig.update_xaxes(title_text="Customers ranked by revenue")
        fig.update_layout(title=f"Pareto – {n80} of {len(p)} customers drive 80% of revenue",
                          legend=dict(orientation="h", y=-0.2))
        show(fig, 400)

    st.markdown("#### Customer segmentation (RFM)")
    seg = an.rfm(df)
    if seg is None:
        st.info("Not enough customers in the current selection for RFM segmentation.")
    else:
        order = ["Champions", "Loyal", "Promising", "At Risk", "Lost"]
        sg = seg.groupby("segment").agg(customers=("customerID", "count"), revenue=("monetary", "sum")).reindex(
            order).dropna().reset_index()
        c1, c2 = st.columns(2)
        with c1:
            f = px.bar(sg, x="segment", y="customers", color="segment", text="customers",
                       title="Customers per segment", category_orders={"segment": order})
            f.update_layout(showlegend=False)
            show(f, 360)
        with c2:
            f = px.bar(sg, x="segment", y="revenue", color="segment", text_auto=",.0f",
                       title="Revenue per segment", category_orders={"segment": order})
            f.update_layout(showlegend=False)
            show(money_axis(f), 360)
        st.caption("Recency / Frequency / Monetary are scored 1–4 by quartile; the sum (3–12) maps to "
                   "Lost (≤4), At Risk (5–6), Promising (7–8), Loyal (9–10), Champions (11–12). "
                   "Recency is measured relative to the last order in the filtered data.")

    st.markdown("#### Customer table")
    q = st.text_input("Search customer / country", "")
    ct = cust.sort_values("revenue", ascending=False)
    if q:
        ct = ct[ct["customer"].str.contains(q, case=False) | ct["customer_country"].str.contains(q, case=False)]
    table(ct.round(2), column_config={
        "revenue": st.column_config.NumberColumn(format="$%.0f"),
        "aov": st.column_config.NumberColumn("Avg order", format="$%.0f"),
        "last_order": st.column_config.DateColumn(format="YYYY-MM-DD")})

# =========================================================================== #
# 4. Employees
# =========================================================================== #
with tabs[3]:
    emp = df.groupby("employeeName").agg(
        revenue=("revenue", "sum"), orders=("orderID", "nunique"),
        customers=("customerID", "nunique"), units=("quantity", "sum")).reset_index()
    emp["aov"] = emp["revenue"] / emp["orders"]
    emp = emp.sort_values("revenue", ascending=False)

    c1, c2 = st.columns(2)
    with c1:
        f = px.bar(emp.sort_values("revenue"), x="revenue", y="employeeName", orientation="h",
                   text_auto=",.0f", title="Revenue by employee")
        show(money_axis(f, "x"), 420)
    with c2:
        f = px.scatter(emp, x="orders", y="aov", size="revenue", color="employeeName",
                       title="Orders handled vs average order value (bubble = revenue)",
                       labels={"orders": "Orders", "aov": "Avg order value"})
        show(money_axis(f), 420)

    et = df.groupby([gcol, "employeeName"])["revenue"].sum().reset_index()
    f = px.line(et, x=gcol, y="revenue", color="employeeName", title=f"Employee revenue per {grain.lower()}")
    show(money_axis(f), 420)

    ec = df.pivot_table(index="employeeName", columns="categoryName", values="revenue", aggfunc="sum", fill_value=0)
    f = px.imshow(ec, text_auto=",.0f", aspect="auto", color_continuous_scale="Blues",
                  title="Employee × category revenue heatmap")
    show(f, 460)

    st.markdown("#### Team table (with reporting line)")
    e_raw = tables["employees"][["employeeID", "employeeName", "title", "city", "country", "reportsTo"]]
    mgr = e_raw.set_index("employeeID")["employeeName"]
    e_raw = e_raw.assign(manager=e_raw["reportsTo"].map(mgr)).drop(columns="reportsTo")
    team = e_raw.merge(emp, on="employeeName", how="left").sort_values("revenue", ascending=False)
    table(team.round(2), column_config={
        "revenue": st.column_config.NumberColumn(format="$%.0f"),
        "aov": st.column_config.NumberColumn("Avg order", format="$%.0f")})

# =========================================================================== #
# 5. Shipping
# =========================================================================== #
with tabs[4]:
    shipped = ol[ol["is_shipped"]]
    ship = ol.groupby("shipper").agg(
        orders=("orderID", "count"), revenue=("revenue", "sum"), freight=("freight", "sum"),
        avg_freight=("freight", "mean")).reset_index()
    perf = shipped.groupby("shipper").agg(
        avg_days=("delivery_days", "mean"), late_rate=("is_late", "mean")).reset_index()
    ship = ship.merge(perf, on="shipper", how="left")
    ship["freight_pct_rev"] = ship["freight"] / ship["revenue"]

    c1, c2, c3 = st.columns(3)
    with c1:
        f = px.pie(ship, names="shipper", values="orders", hole=0.45, title="Order share by shipper")
        show(f, 360)
    with c2:
        f = px.bar(ship, x="shipper", y="avg_freight", color="shipper", text_auto="$.2f",
                   title="Average freight per order")
        f.update_layout(showlegend=False)
        show(f, 360)
    with c3:
        f = px.bar(ship, x="shipper", y="late_rate", color="shipper", text_auto=".1%",
                   title="Late-shipment rate")
        f.update_layout(showlegend=False)
        f.update_yaxes(tickformat=".0%")
        show(f, 360)

    c1, c2 = st.columns(2)
    with c1:
        f = px.box(shipped, x="shipper", y="delivery_days", color="shipper", points="outliers",
                   title="Days from order to shipment", labels={"delivery_days": "Days"})
        f.update_layout(showlegend=False)
        show(f, 400)
    with c2:
        fc = ol.groupby("customer_country").agg(avg_freight=("freight", "mean"),
                                                orders=("orderID", "count")).reset_index()
        fc = fc[fc["orders"] >= 3].nlargest(15, "avg_freight").sort_values("avg_freight")
        f = px.bar(fc, x="avg_freight", y="customer_country", orientation="h", text_auto="$.0f",
                   title="Avg freight per order – top 15 destinations (≥3 orders)")
        show(f, 400)

    ft = ol.assign(month=ol["orderDate"].dt.to_period("M").dt.to_timestamp()).groupby(
        ["month", "shipper"])["freight"].sum().reset_index()
    f = px.line(ft, x="month", y="freight", color="shipper", title="Monthly freight cost by shipper")
    show(money_axis(f), 380)

    st.markdown("#### Shipper scorecard")
    table(ship.round(3), column_config={
        "revenue": st.column_config.NumberColumn(format="$%.0f"),
        "freight": st.column_config.NumberColumn(format="$%.0f"),
        "avg_freight": st.column_config.NumberColumn(format="$%.2f"),
        "late_rate": st.column_config.NumberColumn("late_rate", format="%.3f"),
        "freight_pct_rev": st.column_config.NumberColumn("freight / revenue", format="%.3f")})

    late = ol[ol["is_late"]].sort_values("orderDate", ascending=False)
    with st.expander(f"Late orders ({len(late)})"):
        table(late[["orderID", "orderDate", "shippedDate", "customer", "customer_country",
                    "shipper", "delivery_days", "freight"]])

# =========================================================================== #
# 6. Discounts
# =========================================================================== #
with tabs[5]:
    d = df.assign(band=an.discount_band(df["discount"]))
    band_order = list(an.discount_band(pd.Series([0.0, 0.03, 0.08, 0.12, 0.18, 0.25])).cat.categories)
    bd = d.groupby("band", observed=True).agg(
        lines=("orderID", "count"), avg_qty=("quantity", "mean"),
        revenue=("revenue", "sum"), discount_given=("discount_amt", "sum")).reindex(band_order).dropna().reset_index()

    c1, c2 = st.columns(2)
    with c1:
        f = px.bar(bd, x="band", y="lines", text="lines", title="Order lines by discount band")
        show(f, 380)
    with c2:
        f = px.bar(bd, x="band", y="avg_qty", text_auto=".1f", title="Average quantity per line by discount band")
        show(f, 380)

    c1, c2 = st.columns(2)
    with c1:
        f = px.bar(bd, x="band", y="revenue", text_auto=",.0f", title="Net revenue by discount band")
        show(money_axis(f), 380)
    with c2:
        dc = df.groupby("categoryName").agg(given=("discount_amt", "sum"), gross=("gross", "sum")).reset_index()
        dc["rate"] = dc["given"] / dc["gross"]
        f = px.bar(dc.sort_values("rate"), x="rate", y="categoryName", orientation="h",
                   text_auto=".1%", title="Effective discount rate by category")
        f.update_xaxes(tickformat=".0%")
        show(f, 380)

    dm = df.groupby(gcol).agg(given=("discount_amt", "sum"), gross=("gross", "sum")).reset_index()
    dm["rate"] = dm["given"] / dm["gross"]
    f = px.line(dm, x=gcol, y="rate", markers=True, title=f"Effective discount rate per {grain.lower()}")
    f.update_yaxes(tickformat=".1%")
    show(f, 340)

# =========================================================================== #
# 7. Data model & explorer
# =========================================================================== #
with tabs[6]:
    st.markdown("### How the 7 tables are related")
    st.graphviz_chart(an.ER_DOT)
    st.caption("Orange = transactional tables, blue = dimensions. The dashboard joins everything into one "
               "order-line table (grain: order × product).")
    table(an.RELATIONSHIPS)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Table sizes")
        sizes = pd.DataFrame({"table": list(tables), "rows": [len(v) for v in tables.values()],
                              "columns": [v.shape[1] for v in tables.values()]})
        table(sizes)
    with c2:
        st.markdown("#### Referential integrity checks")
        table(an.data_quality(tables))

    st.markdown("### Joined order-line dataset")
    cols = ["orderID", "orderDate", "customer", "customer_country", "employeeName", "shipper",
            "categoryName", "productName", "unitPrice", "quantity", "discount", "revenue", "freight",
            "shippedDate", "delivery_days", "is_late"]
    st.caption(f"{len(df):,} rows after filters")
    table(df[cols].sort_values("orderDate", ascending=False).head(1000),
          column_config={"revenue": st.column_config.NumberColumn(format="$%.2f")})
    st.download_button("⬇️ Download filtered joined data (CSV)", df.to_csv(index=False).encode("utf-8"),
                       file_name="brand_traders_joined.csv", mime="text/csv")
