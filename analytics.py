"""Data layer for the Brand Traders sales dashboard.

Pure pandas – no Streamlit / Plotly imports – so it can be unit-tested.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

TABLE_NAMES = [
    "categories",
    "customers",
    "employees",
    "order_details",
    "orders",
    "products",
    "shippers",
]

# plotly's "country names" lookup wants full names for these two
COUNTRY_FIX = {"USA": "United States", "UK": "United Kingdom"}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def read_csv_any(src) -> pd.DataFrame:
    """Read a CSV from a path or file-like object; fall back to latin-1."""
    try:
        return pd.read_csv(src, encoding="utf-8")
    except UnicodeDecodeError:
        if hasattr(src, "seek"):
            src.seek(0)
        return pd.read_csv(src, encoding="latin1")


def load_folder(folder: str) -> dict[str, pd.DataFrame]:
    p = Path(folder)
    missing = [n for n in TABLE_NAMES if not (p / f"{n}.csv").exists()]
    if missing:
        raise FileNotFoundError(f"Missing in '{folder}': {', '.join(m + '.csv' for m in missing)}")
    return {n: read_csv_any(p / f"{n}.csv") for n in TABLE_NAMES}


def load_uploads(files: dict[str, bytes]) -> dict[str, pd.DataFrame]:
    """`files` maps original filename -> raw bytes."""
    found = {}
    for fname, raw in files.items():
        stem = Path(fname).stem.lower()
        if stem in TABLE_NAMES:
            found[stem] = read_csv_any(io.BytesIO(raw))
    missing = [n for n in TABLE_NAMES if n not in found]
    if missing:
        raise FileNotFoundError(f"Still need: {', '.join(m + '.csv' for m in missing)}")
    return found


# --------------------------------------------------------------------------- #
# Joining – one denormalised "line-item" table
# --------------------------------------------------------------------------- #
def build_master(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """order_details -> orders -> customers / employees / shippers
       order_details -> products -> categories
    Grain = one row per order line."""
    cust = t["customers"].rename(columns={
        "companyName": "customer", "contactName": "contact",
        "contactTitle": "contact_title", "city": "customer_city",
        "country": "customer_country"})
    emp = t["employees"][["employeeID", "employeeName", "title", "country"]].rename(
        columns={"title": "employee_title", "country": "employee_country"})
    ship = t["shippers"].rename(columns={"companyName": "shipper"})
    cat = t["categories"][["categoryID", "categoryName"]]
    prod = t["products"].rename(columns={"unitPrice": "listPrice"})

    orders = t["orders"].copy()
    for c in ("orderDate", "requiredDate", "shippedDate"):
        orders[c] = pd.to_datetime(orders[c], errors="coerce")

    df = (
        t["order_details"]
        .merge(orders, on="orderID", how="inner", validate="many_to_one")
        .merge(cust, on="customerID", how="left", validate="many_to_one")
        .merge(emp, on="employeeID", how="left", validate="many_to_one")
        .merge(ship, on="shipperID", how="left", validate="many_to_one")
        .merge(prod, on="productID", how="left", validate="many_to_one")
        .merge(cat, on="categoryID", how="left", validate="many_to_one")
    )

    df["gross"] = df["unitPrice"] * df["quantity"]
    df["discount_amt"] = df["gross"] * df["discount"]
    df["revenue"] = df["gross"] - df["discount_amt"]

    df["year"] = df["orderDate"].dt.year
    df["month"] = df["orderDate"].dt.to_period("M").dt.to_timestamp()
    df["quarter"] = df["orderDate"].dt.to_period("Q").dt.to_timestamp()
    df["weekday"] = df["orderDate"].dt.day_name()
    df["delivery_days"] = (df["shippedDate"] - df["orderDate"]).dt.days
    df["is_shipped"] = df["shippedDate"].notna()
    df["is_late"] = df["is_shipped"] & (df["shippedDate"] > df["requiredDate"])
    df["customer_country_map"] = df["customer_country"].replace(COUNTRY_FIX)
    df["status"] = np.where(df["discontinued"] == 1, "Discontinued", "Active")
    return df


def order_level(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one row per order (freight lives at this grain, so summing it
    on line-items would double-count)."""
    g = df.groupby("orderID").agg(
        revenue=("revenue", "sum"),
        gross=("gross", "sum"),
        units=("quantity", "sum"),
        lines=("productID", "count"),
        freight=("freight", "first"),
        orderDate=("orderDate", "first"),
        shippedDate=("shippedDate", "first"),
        customerID=("customerID", "first"),
        customer=("customer", "first"),
        customer_country=("customer_country", "first"),
        employeeName=("employeeName", "first"),
        shipper=("shipper", "first"),
        delivery_days=("delivery_days", "first"),
        is_shipped=("is_shipped", "first"),
        is_late=("is_late", "first"),
    ).reset_index()
    return g


# --------------------------------------------------------------------------- #
# Analytics helpers
# --------------------------------------------------------------------------- #
def kpis(df: pd.DataFrame, ol: pd.DataFrame) -> dict:
    revenue = df["revenue"].sum()
    orders = df["orderID"].nunique()
    shipped = ol[ol["is_shipped"]]
    return {
        "revenue": revenue,
        "orders": orders,
        "customers": df["customerID"].nunique(),
        "aov": revenue / orders if orders else 0.0,
        "units": int(df["quantity"].sum()),
        "freight": ol["freight"].sum(),
        "discount_rate": df["discount_amt"].sum() / df["gross"].sum() if df["gross"].sum() else 0.0,
        "on_time": 1 - shipped["is_late"].mean() if len(shipped) else np.nan,
        "avg_delivery": shipped["delivery_days"].mean() if len(shipped) else np.nan,
    }


def rfm(df: pd.DataFrame) -> pd.DataFrame | None:
    """Quartile-based RFM segmentation per customer."""
    ref = df["orderDate"].max() + pd.Timedelta(days=1)
    g = df.groupby(["customerID", "customer", "customer_country"]).agg(
        last_order=("orderDate", "max"),
        frequency=("orderID", "nunique"),
        monetary=("revenue", "sum"),
    ).reset_index()
    if len(g) < 4:
        return None
    g["recency"] = (ref - g["last_order"]).dt.days
    g["R"] = pd.qcut(g["recency"].rank(method="first"), 4, labels=[4, 3, 2, 1]).astype(int)
    g["F"] = pd.qcut(g["frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    g["M"] = pd.qcut(g["monetary"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    g["score"] = g["R"] + g["F"] + g["M"]
    g["segment"] = pd.cut(
        g["score"], bins=[0, 4, 6, 8, 10, 12],
        labels=["Lost", "At Risk", "Promising", "Loyal", "Champions"],
    ).astype(str)
    return g


def discount_band(s: pd.Series) -> pd.Series:
    return pd.cut(
        s, bins=[-0.001, 0, 0.05, 0.10, 0.15, 0.20, 1.0],
        labels=["No discount", "≤5%", "5–10%", "10–15%", "15–20%", ">20%"],
    )


def build_insights(df: pd.DataFrame, ol: pd.DataFrame) -> list[str]:
    out: list[str] = []
    total = df["revenue"].sum()
    if not total:
        return out

    cat = df.groupby("categoryName")["revenue"].sum().sort_values(ascending=False)
    out.append(f"**{cat.index[0]}** is the strongest category, generating "
               f"{cat.iloc[0] / total:.1%} of revenue (${cat.iloc[0]:,.0f}).")

    prod = df.groupby("productName")["revenue"].sum().sort_values(ascending=False)
    out.append(f"Best-selling product by revenue: **{prod.index[0]}** (${prod.iloc[0]:,.0f}); "
               f"the top 5 products make up {prod.head(5).sum() / total:.1%} of sales.")

    ctry = df.groupby("customer_country")["revenue"].sum().sort_values(ascending=False)
    out.append(f"**{ctry.index[0]}** is the largest market ({ctry.iloc[0] / total:.1%} of revenue); "
               f"top 3 countries account for {ctry.head(3).sum() / total:.1%}.")

    cust = df.groupby("customer")["revenue"].sum().sort_values(ascending=False)
    top10 = cust.head(10).sum() / total
    out.append(f"Top customer is **{cust.index[0]}** (${cust.iloc[0]:,.0f}); the top 10 customers "
               f"contribute {top10:.1%} of revenue.")

    emp = df.groupby("employeeName")["revenue"].sum().sort_values(ascending=False)
    out.append(f"Top sales employee: **{emp.index[0]}** with ${emp.iloc[0]:,.0f} "
               f"({emp.iloc[0] / total:.1%} of revenue).")

    m = df.groupby("month")["revenue"].sum()
    if len(m):
        out.append(f"Peak month was **{m.idxmax():%b %Y}** with ${m.max():,.0f} in revenue.")

    shipped = ol[ol["is_shipped"]]
    if len(shipped):
        late = shipped.groupby("shipper")["is_late"].mean().sort_values(ascending=False)
        out.append(f"{shipped['is_late'].mean():.1%} of shipped orders arrived after the required date; "
                   f"highest late-rate carrier: **{late.index[0]}** ({late.iloc[0]:.1%}).")

    disc = df["discount"] > 0
    out.append(f"{disc.mean():.1%} of order lines carry a discount, representing "
               f"{df.loc[disc, 'revenue'].sum() / total:.1%} of revenue "
               f"(${df['discount_amt'].sum():,.0f} given away in total).")

    unshipped = int((~ol["is_shipped"]).sum())
    if unshipped:
        out.append(f"{unshipped} order(s) have no shipped date yet (still open).")
    return out


def data_quality(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Referential-integrity checks across the 7 tables."""
    od, o, p = t["order_details"], t["orders"], t["products"]
    checks = [
        ("order_details.orderID → orders", (~od["orderID"].isin(o["orderID"])).sum()),
        ("order_details.productID → products", (~od["productID"].isin(p["productID"])).sum()),
        ("orders.customerID → customers", (~o["customerID"].isin(t["customers"]["customerID"])).sum()),
        ("orders.employeeID → employees", (~o["employeeID"].isin(t["employees"]["employeeID"])).sum()),
        ("orders.shipperID → shippers", (~o["shipperID"].isin(t["shippers"]["shipperID"])).sum()),
        ("products.categoryID → categories", (~p["categoryID"].isin(t["categories"]["categoryID"])).sum()),
        ("Duplicate orderIDs in orders", o["orderID"].duplicated().sum()),
        ("Duplicate (orderID, productID) lines", od.duplicated(["orderID", "productID"]).sum()),
        ("Orders with no shippedDate", o["shippedDate"].isna().sum()),
    ]
    res = pd.DataFrame(checks, columns=["Check", "Issues found"])
    res["Status"] = np.where(res["Issues found"] == 0, "✅ OK", "⚠️ Review")
    return res


RELATIONSHIPS = pd.DataFrame(
    [
        ("orders", "customerID", "customers", "customerID", "many-to-one"),
        ("orders", "employeeID", "employees", "employeeID", "many-to-one"),
        ("orders", "shipperID", "shippers", "shipperID", "many-to-one"),
        ("order_details", "orderID", "orders", "orderID", "many-to-one"),
        ("order_details", "productID", "products", "productID", "many-to-one"),
        ("products", "categoryID", "categories", "categoryID", "many-to-one"),
        ("employees", "reportsTo", "employees", "employeeID", "self-reference"),
    ],
    columns=["From table", "Key", "To table", "Key ", "Type"],
)

ER_DOT = r"""
digraph ER {
  rankdir=LR; bgcolor="transparent";
  node [shape=record, fontname="Helvetica", fontsize=10, style=filled, fillcolor="#EEF3FB", color="#4C78A8"];
  edge [color="#7A8CA5", arrowhead=crow, fontname="Helvetica", fontsize=9];

  customers     [label="{customers|customerID (PK)\lcompanyName\lcontactName\lcity\lcountry\l}"];
  employees     [label="{employees|employeeID (PK)\lemployeeName\ltitle\lreportsTo (FK)\l}"];
  shippers      [label="{shippers|shipperID (PK)\lcompanyName\l}"];
  orders        [label="{orders|orderID (PK)\lcustomerID (FK)\lemployeeID (FK)\lshipperID (FK)\lorderDate\lrequiredDate\lshippedDate\lfreight\l}", fillcolor="#FFF4E0", color="#F58518"];
  order_details [label="{order_details|orderID (FK)\lproductID (FK)\lunitPrice\lquantity\ldiscount\l}", fillcolor="#FFF4E0", color="#F58518"];
  products      [label="{products|productID (PK)\lproductName\lunitPrice\ldiscontinued\lcategoryID (FK)\l}"];
  categories    [label="{categories|categoryID (PK)\lcategoryName\ldescription\l}"];

  orders -> customers;
  orders -> employees;
  orders -> shippers;
  order_details -> orders;
  order_details -> products;
  products -> categories;
  employees -> employees [label="reportsTo"];
}
"""
