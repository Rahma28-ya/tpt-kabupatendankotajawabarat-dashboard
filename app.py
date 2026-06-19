"""
Dashboard Prediksi TPT (Tingkat Pengangguran Terbuka) Provinsi Jawa Barat
Algoritma: Regresi Linear Berganda & Random Forest Regressor
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from streamlit_option_menu import option_menu

warnings.filterwarnings("ignore")

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Dashboard TPT Jawa Barat",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
GEO_DIR = BASE_DIR / "geo"

FEATURES = ["Jumlah_Penduduk", "TPAK", "IPM", "PDRB"]
TARGET = "TPT"
FEATURE_LABEL = {
    "Jumlah_Penduduk": "Jumlah Penduduk (jiwa)",
    "TPAK": "TPAK - Tingkat Partisipasi Angkatan Kerja (%)",
    "IPM": "IPM - Indeks Pembangunan Manusia",
    "PDRB": "PDRB (juta rupiah)",
}
SKENARIO_LIST = ["80_20", "70_30", "60_40"]
SKENARIO_LABEL = {"80_20": "80/20", "70_30": "70/30", "60_40": "60/40"}

# ============================================================
# PALET WARNA: PINK PASTEL, UNGU PASTEL, BIRU
# ============================================================
C_PINK = "#FF8FB1"
C_PINK_SOFT = "#FFD6E8"
C_PURPLE = "#A66CFF"
C_PURPLE_SOFT = "#E0C3FC"
C_BLUE = "#4EA8DE"
C_BLUE_SOFT = "#BDE0FE"
C_TEXT = "#3A2E4D"
C_BG = "#FFF7FB"

SEQ_COLORSCALE = [
    [0.0, C_BLUE_SOFT],
    [0.25, C_BLUE],
    [0.55, C_PURPLE],
    [0.8, C_PINK],
    [1.0, "#FF4F8B"],
]
CAT_COLORS = [C_PURPLE, C_PINK, C_BLUE, "#C77DFF", "#FFADC6", "#80CFFF"]

# ============================================================
# CSS KUSTOM
# ============================================================
st.markdown(
    f"""
<style>
.stApp {{
    background: linear-gradient(160deg, {C_BG} 0%, #F3E8FF 45%, #EAF4FF 100%);
}}
h1, h2, h3 {{
    color: {C_TEXT};
    font-weight: 800;
}}
.hero-box {{
    background: linear-gradient(120deg, {C_PINK_SOFT} 0%, {C_PURPLE_SOFT} 55%, {C_BLUE_SOFT} 100%);
    padding: 1.6rem 2rem;
    border-radius: 22px;
    margin-bottom: 1.4rem;
    box-shadow: 0 8px 24px rgba(166, 108, 255, 0.15);
}}
.metric-card {{
    background: white;
    border-radius: 18px;
    padding: 1.1rem 1.3rem;
    box-shadow: 0 6px 18px rgba(166, 108, 255, 0.12);
    border: 1px solid #F0DFFF;
}}
.metric-card h3 {{
    margin: 0;
    font-size: 0.85rem;
    color: #8B7AA8;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}}
.metric-card .big {{
    font-size: 1.9rem;
    font-weight: 800;
    margin: 0.15rem 0;
}}
.metric-card .sub {{
    font-size: 0.85rem;
    color: #6B5C85;
}}
.pill {{
    display: inline-block;
    padding: 0.2rem 0.8rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    margin-right: 0.4rem;
}}
.pill-pink {{ background: {C_PINK_SOFT}; color: #C2356B; }}
.pill-purple {{ background: {C_PURPLE_SOFT}; color: #6B2FBF; }}
.pill-blue {{ background: {C_BLUE_SOFT}; color: #1B6FA8; }}
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #FBEAF5 0%, #EFE3FF 100%);
}}
div[data-testid="stMetric"] {{
    background: white;
    padding: 0.9rem 1rem;
    border-radius: 16px;
    box-shadow: 0 4px 14px rgba(166, 108, 255, 0.1);
}}
.footer-note {{
    color: #8B7AA8;
    font-size: 0.8rem;
    text-align: center;
    margin-top: 2rem;
}}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# FUNGSI LOAD DATA (CACHED)
# ============================================================
@st.cache_data
def load_integrated():
    df = pd.read_csv(DATA_DIR / "data_integrated.csv")
    df["Tahun"] = df["Tahun"].astype(int)
    return df


@st.cache_data
def load_split(rasio):
    train = pd.read_csv(DATA_DIR / f"data_train_{rasio}.csv")
    test = pd.read_csv(DATA_DIR / f"data_test_{rasio}.csv")
    return train, test


@st.cache_data
def load_geojson():
    with open(GEO_DIR / "jabar.geojson", "r") as f:
        return json.load(f)


# ============================================================
# FUNGSI MODEL (CACHED)
# ============================================================
@st.cache_resource
def train_evaluate_all_scenarios():
    """Mereplikasi pipeline asli: melatih RLB & RF pada 3 skenario split."""
    hasil_rows = []
    detail_pred = {}
    for rasio in SKENARIO_LIST:
        label = SKENARIO_LABEL[rasio]
        train, test = load_split(rasio)

        X_train, y_train = train[FEATURES], train[TARGET]
        X_test, y_test = test[FEATURES], test[TARGET]

        # Regresi Linear Berganda
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        rlb = LinearRegression().fit(X_train_s, y_train)
        pred_rlb = rlb.predict(X_test_s)

        # Random Forest
        rf = RandomForestRegressor(n_estimators=500, random_state=42)
        rf.fit(X_train, y_train)
        pred_rf = rf.predict(X_test)

        for nama, pred in [("Regresi Linear Berganda", pred_rlb), ("Random Forest Regressor", pred_rf)]:
            mae = mean_absolute_error(y_test, pred)
            rmse = np.sqrt(mean_squared_error(y_test, pred))
            r2 = r2_score(y_test, pred)
            hasil_rows.append(
                {"Skenario": label, "Model": nama, "MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}
            )

        hasil_test = test.copy()
        hasil_test["Prediksi_RLB"] = pred_rlb
        hasil_test["Prediksi_RF"] = pred_rf
        hasil_test = hasil_test.sort_values(["Kabupaten_Kota", "Tahun"]).reset_index(drop=True)
        detail_pred[label] = hasil_test

        # simpan model & feature importance utk skenario ini
        detail_pred[label].attrs["rf_importance"] = dict(zip(FEATURES, rf.feature_importances_))
        detail_pred[label].attrs["rlb_coef"] = dict(zip(FEATURES, rlb.coef_))
        detail_pred[label].attrs["rlb_intercept"] = float(rlb.intercept_)

    df_hasil = pd.DataFrame(hasil_rows)
    return df_hasil, detail_pred


@st.cache_resource
def train_full_model():
    """Melatih model RF & RLB pada SELURUH data (2018-2025) untuk prediksi langsung/live."""
    df = load_integrated()
    X, y = df[FEATURES], df[TARGET]

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    rlb = LinearRegression().fit(X_s, y)

    rf = RandomForestRegressor(n_estimators=500, random_state=42)
    rf.fit(X, y)

    return {"scaler": scaler, "rlb": rlb, "rf": rf}


def predict_tpt(models, jumlah_penduduk, tpak, ipm, pdrb, model_choice="Random Forest Regressor"):
    X_input = pd.DataFrame(
        [{"Jumlah_Penduduk": jumlah_penduduk, "TPAK": tpak, "IPM": ipm, "PDRB": pdrb}]
    )[FEATURES]
    if model_choice == "Random Forest Regressor":
        return float(models["rf"].predict(X_input)[0])
    else:
        X_s = models["scaler"].transform(X_input)
        return float(models["rlb"].predict(X_s)[0])


def proyeksi_linear(series_tahun, series_nilai, tahun_target):
    """Ekstrapolasi tren linear sederhana (least squares) ke tahun target."""
    coef = np.polyfit(series_tahun, series_nilai, 1)
    return float(np.polyval(coef, tahun_target))


# ============================================================
# LOAD DATA UTAMA
# ============================================================
df_all = load_integrated()
geojson_jabar = load_geojson()
df_eval, detail_pred = train_evaluate_all_scenarios()
full_models = train_full_model()

TAHUN_LIST = sorted(df_all["Tahun"].unique().tolist())

best_row = df_eval.loc[df_eval["R2"].idxmax()]
BEST_SKENARIO = best_row["Skenario"]
BEST_MODEL = best_row["Model"]

# ============================================================
# SIDEBAR NAVIGASI
# ============================================================
with st.sidebar:
    st.markdown(
        f"""
        <div style="text-align:center; padding: 0.6rem 0 1rem 0;">
            <div style="font-size:2.4rem;">📊</div>
            <div style="font-weight:800; font-size:1.15rem; color:{C_TEXT};">TPT Jawa Barat</div>
            <div style="font-size:0.8rem; color:#8B7AA8;">Regresi Linear Berganda × Random Forest</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    menu = option_menu(
        menu_title=None,
        options=[
            "Beranda",
            "Peta Jawa Barat",
            "Tren Provinsi",
            "Aktual vs Prediksi",
            "Evaluasi Model",
            "Prediksi TPT",
            "Data Lengkap",
        ],
        icons=["house-heart", "geo-alt", "graph-up-arrow", "bullseye", "clipboard-data", "magic", "table"],
        default_index=0,
        styles={
            "container": {"padding": "0", "background-color": "transparent"},
            "icon": {"color": C_PURPLE, "font-size": "16px"},
            "nav-link": {
                "font-size": "14px",
                "text-align": "left",
                "margin": "3px 0",
                "border-radius": "12px",
                "--hover-color": C_PURPLE_SOFT,
            },
            "nav-link-selected": {"background-color": C_PURPLE, "color": "white", "font-weight": "700"},
        },
    )
    st.markdown("---")
    st.markdown(
        f"""
        <div class="metric-card">
        <h3>Model Terbaik</h3>
        <div class="big" style="font-size:1.05rem; color:{C_PURPLE};">{BEST_MODEL}</div>
        <div class="sub">Skenario {BEST_SKENARIO} &nbsp;|&nbsp; R² = {best_row['R2']:.3f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# HALAMAN: BERANDA
# ============================================================
if menu == "Beranda":
    st.markdown(
        f"""
        <div class="hero-box">
            <h1 style="margin-bottom:0.2rem;">📊 Dashboard Prediksi TPT Jawa Barat</h1>
            <p style="color:#5C4A75; font-size:1rem; margin-bottom:0;">
            Analisis dan prediksi <b>Tingkat Pengangguran Terbuka (TPT)</b> 27 kabupaten/kota
            di Provinsi Jawa Barat tahun 2018–2025, menggunakan algoritma
            <b>Regresi Linear Berganda</b> dan <b>Random Forest Regressor</b>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tahun_pilih = st.selectbox("Pilih tahun:", TAHUN_LIST, index=len(TAHUN_LIST) - 1)
    df_tahun = df_all[df_all["Tahun"] == tahun_pilih].sort_values("TPT")

    tertinggi = df_tahun.iloc[-1]
    terendah = df_tahun.iloc[0]
    rata2 = df_tahun["TPT"].mean()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""<div class="metric-card"><h3>🔺 TPT Tertinggi</h3>
            <div class="big" style="color:{C_PINK};">{tertinggi['TPT']:.2f}%</div>
            <div class="sub">{tertinggi['Kabupaten_Kota']}</div></div>""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""<div class="metric-card"><h3>🔻 TPT Terendah</h3>
            <div class="big" style="color:{C_BLUE};">{terendah['TPT']:.2f}%</div>
            <div class="sub">{terendah['Kabupaten_Kota']}</div></div>""",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""<div class="metric-card"><h3>📐 Rata-rata Provinsi</h3>
            <div class="big" style="color:{C_PURPLE};">{rata2:.2f}%</div>
            <div class="sub">Tahun {tahun_pilih}</div></div>""",
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""<div class="metric-card"><h3>🏙️ Jumlah Wilayah</h3>
            <div class="big" style="color:{C_TEXT};">{df_tahun.shape[0]}</div>
            <div class="sub">Kabupaten/Kota</div></div>""",
            unsafe_allow_html=True,
        )

    st.write("")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("##### 🏆 5 TPT Tertinggi")
        top5 = df_tahun.sort_values("TPT", ascending=False).head(5)
        fig = px.bar(
            top5, x="TPT", y="Kabupaten_Kota", orientation="h",
            color="TPT", color_continuous_scale=SEQ_COLORSCALE, text="TPT",
        )
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig.update_layout(
            yaxis={"categoryorder": "total ascending"}, height=350, showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False,
            margin=dict(l=0, r=10, t=10, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        st.markdown("##### 🌿 5 TPT Terendah")
        bot5 = df_tahun.sort_values("TPT", ascending=True).head(5)
        fig2 = px.bar(
            bot5, x="TPT", y="Kabupaten_Kota", orientation="h",
            color="TPT", color_continuous_scale=SEQ_COLORSCALE, text="TPT",
        )
        fig2.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig2.update_layout(
            yaxis={"categoryorder": "total descending"}, height=350, showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False,
            margin=dict(l=0, r=10, t=10, b=0),
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.info(
        f"💡 Model dengan performa terbaik saat ini adalah **{BEST_MODEL}** pada skenario "
        f"**{BEST_SKENARIO}** dengan R² = {best_row['R2']:.3f}, MAE = {best_row['MAE']:.3f}, "
        f"RMSE = {best_row['RMSE']:.3f}. Lihat detail di menu **Evaluasi Model**."
    )

# ============================================================
# HALAMAN: PETA JAWA BARAT
# ============================================================
elif menu == "Peta Jawa Barat":
    st.markdown("## 🗺️ Peta Sebaran TPT Jawa Barat")
    st.caption("Choropleth TPT per kabupaten/kota — pilih tahun untuk melihat persebarannya.")

    tahun_map = st.select_slider("Tahun:", options=TAHUN_LIST, value=TAHUN_LIST[-1])
    df_map = df_all[df_all["Tahun"] == tahun_map]

    fig_map = px.choropleth(
        df_map,
        geojson=geojson_jabar,
        locations="Kabupaten_Kota",
        featureidkey="properties.Kabupaten_Kota",
        color="TPT",
        color_continuous_scale=SEQ_COLORSCALE,
        hover_name="Kabupaten_Kota",
        hover_data={"TPT": ":.2f", "IPM": ":.2f", "TPAK": ":.2f", "Kabupaten_Kota": False},
        labels={"TPT": "TPT (%)"},
    )
    fig_map.update_geos(fitbounds="locations", visible=False)
    fig_map.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        height=560,
        coloraxis_colorbar=dict(title="TPT (%)"),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_map, use_container_width=True)

    c1, c2 = st.columns(2)
    tertinggi = df_map.loc[df_map["TPT"].idxmax()]
    terendah = df_map.loc[df_map["TPT"].idxmin()]
    with c1:
        st.success(f"🔺 **TPT tertinggi {tahun_map}:** {tertinggi['Kabupaten_Kota']} — {tertinggi['TPT']:.2f}%")
    with c2:
        st.success(f"🔻 **TPT terendah {tahun_map}:** {terendah['Kabupaten_Kota']} — {terendah['TPT']:.2f}%")

# ============================================================
# HALAMAN: TREN PROVINSI
# ============================================================
elif menu == "Tren Provinsi":
    st.markdown("## 📈 Tren TPT Provinsi Jawa Barat (2018–2025)")

    tren = df_all.groupby("Tahun")["TPT"].agg(["mean", "max", "min"]).reset_index()
    tren.columns = ["Tahun", "Rata-rata", "Tertinggi", "Terendah"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tren["Tahun"], y=tren["Tertinggi"], mode="lines+markers",
                              name="Tertinggi", line=dict(color=C_PINK, width=3)))
    fig.add_trace(go.Scatter(x=tren["Tahun"], y=tren["Rata-rata"], mode="lines+markers",
                              name="Rata-rata Provinsi", line=dict(color=C_PURPLE, width=4)))
    fig.add_trace(go.Scatter(x=tren["Tahun"], y=tren["Terendah"], mode="lines+markers",
                              name="Terendah", line=dict(color=C_BLUE, width=3)))
    fig.update_layout(
        height=440, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(dtick=1), yaxis_title="TPT (%)", margin=dict(l=0, r=0, t=20, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    thn_tert = tren.loc[tren["Rata-rata"].idxmax()]
    thn_rend = tren.loc[tren["Rata-rata"].idxmin()]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"""<div class="metric-card"><h3>📅 Tahun TPT Provinsi Tertinggi</h3>
            <div class="big" style="color:{C_PINK};">{int(thn_tert['Tahun'])} → {thn_tert['Rata-rata']:.2f}%</div>
            <div class="sub">Rata-rata TPT seluruh kab/kota</div></div>""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""<div class="metric-card"><h3>📅 Tahun TPT Provinsi Terendah</h3>
            <div class="big" style="color:{C_BLUE};">{int(thn_rend['Tahun'])} → {thn_rend['Rata-rata']:.2f}%</div>
            <div class="sub">Rata-rata TPT seluruh kab/kota</div></div>""",
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown("##### 🏙️ Tren TPT per Kabupaten/Kota (pilih wilayah)")
    avg_per_wilayah = df_all.groupby("Kabupaten_Kota")["TPT"].mean().sort_values(ascending=False)
    default_wilayah = [avg_per_wilayah.index[0], avg_per_wilayah.index[-1]]
    wilayah_pilih = st.multiselect(
        "Pilih kabupaten/kota:", sorted(df_all["Kabupaten_Kota"].unique()),
        default=default_wilayah,
    )
    if wilayah_pilih:
        df_w = df_all[df_all["Kabupaten_Kota"].isin(wilayah_pilih)]
        fig_w = px.line(df_w, x="Tahun", y="TPT", color="Kabupaten_Kota", markers=True,
                         color_discrete_sequence=CAT_COLORS)
        fig_w.update_layout(height=400, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                             xaxis=dict(dtick=1), margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_w, use_container_width=True)
    else:
        st.caption("Pilih satu atau lebih kabupaten/kota di atas untuk melihat tren individualnya.")

# ============================================================
# HALAMAN: AKTUAL VS PREDIKSI
# ============================================================
elif menu == "Aktual vs Prediksi":
    st.markdown("## 🎯 Data Aktual vs Prediksi")
    st.caption("Bandingkan nilai TPT aktual dengan hasil prediksi model pada data uji (test set).")

    colf1, colf2 = st.columns(2)
    with colf1:
        skenario_pilih = st.selectbox("Skenario pembagian data:", [SKENARIO_LABEL[s] for s in SKENARIO_LIST])
    df_pred = detail_pred[skenario_pilih]
    with colf2:
        tahun_tersedia = sorted(df_pred["Tahun"].unique())
        tahun_pilih_pred = st.selectbox("Tahun:", ["Semua Tahun"] + tahun_tersedia)

    if tahun_pilih_pred != "Semua Tahun":
        df_show = df_pred[df_pred["Tahun"] == tahun_pilih_pred]
    else:
        df_show = df_pred

    df_show = df_show.copy()
    df_show["Error_RLB"] = (df_show["TPT"] - df_show["Prediksi_RLB"]).abs()
    df_show["Error_RF"] = (df_show["TPT"] - df_show["Prediksi_RF"]).abs()

    c1, c2, c3 = st.columns(3)
    c1.metric("Jumlah Data Uji", len(df_show))
    c2.metric("MAE Regresi Linear", f"{df_show['Error_RLB'].mean():.3f}")
    c3.metric("MAE Random Forest", f"{df_show['Error_RF'].mean():.3f}")

    st.markdown("##### 📊 Visualisasi Aktual vs Prediksi")
    df_melt = df_show.melt(
        id_vars=["Kabupaten_Kota", "Tahun"],
        value_vars=["TPT", "Prediksi_RLB", "Prediksi_RF"],
        var_name="Jenis", value_name="Nilai",
    )
    df_melt["Jenis"] = df_melt["Jenis"].replace(
        {"TPT": "Aktual", "Prediksi_RLB": "Prediksi (RLB)", "Prediksi_RF": "Prediksi (RF)"}
    )
    fig = px.bar(
        df_melt.sort_values(["Kabupaten_Kota", "Tahun"]),
        x="Kabupaten_Kota", y="Nilai", color="Jenis", barmode="group",
        color_discrete_map={"Aktual": C_PURPLE, "Prediksi (RLB)": C_BLUE, "Prediksi (RF)": C_PINK},
        height=480,
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis_tickangle=-45, legend_title="", margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### 🔵 Scatter Aktual vs Prediksi (Random Forest)")
    fig_sc = px.scatter(
        df_show, x="TPT", y="Prediksi_RF", color_discrete_sequence=[C_PURPLE],
        hover_name="Kabupaten_Kota", labels={"TPT": "Aktual", "Prediksi_RF": "Prediksi RF"},
    )
    min_v, max_v = df_show["TPT"].min(), df_show["TPT"].max()
    fig_sc.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode="lines",
                                 line=dict(dash="dash", color=C_PINK), name="Garis Ideal"))
    fig_sc.update_layout(height=420, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown("##### 📋 Tabel Aktual vs Prediksi")
    tabel_show = df_show[["Kabupaten_Kota", "Tahun", "TPT", "Prediksi_RLB", "Prediksi_RF", "Error_RLB", "Error_RF"]]
    tabel_show.columns = ["Kabupaten/Kota", "Tahun", "Aktual", "Prediksi RLB", "Prediksi RF", "Error RLB", "Error RF"]
    st.dataframe(
        tabel_show.style.format(
            {"Aktual": "{:.2f}", "Prediksi RLB": "{:.2f}", "Prediksi RF": "{:.2f}",
             "Error RLB": "{:.2f}", "Error RF": "{:.2f}"}
        ).background_gradient(cmap="RdPu", subset=["Error RLB", "Error RF"]),
        use_container_width=True, height=400,
    )

# ============================================================
# HALAMAN: EVALUASI MODEL
# ============================================================
elif menu == "Evaluasi Model":
    st.markdown("## 📐 Evaluasi Error Model")
    st.caption("Perbandingan metrik MAE, RMSE, dan R² untuk setiap model & skenario pembagian data.")

    st.dataframe(
        df_eval.style.format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "R2": "{:.4f}"})
        .background_gradient(cmap="PuBu", subset=["R2"])
        .background_gradient(cmap="RdPu_r", subset=["MAE", "RMSE"]),
        use_container_width=True,
    )

    metrik_pilih = st.radio("Pilih metrik untuk visualisasi:", ["MAE", "RMSE", "R2"], horizontal=True)
    fig = px.bar(
        df_eval, x="Skenario", y=metrik_pilih, color="Model", barmode="group",
        color_discrete_map={"Regresi Linear Berganda": C_BLUE, "Random Forest Regressor": C_PINK},
        text=metrik_pilih, height=420,
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                       legend_title="", margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

    best = df_eval.loc[df_eval["R2"].idxmax()]
    st.success(
        f"🏆 **Model & skenario terbaik:** {best['Model']} — Skenario {best['Skenario']} "
        f"(MAE={best['MAE']:.4f}, RMSE={best['RMSE']:.4f}, R²={best['R2']:.4f})"
    )

    st.markdown("##### 🌲 Feature Importance — Random Forest (per skenario)")
    skenario_imp = st.selectbox("Skenario:", [SKENARIO_LABEL[s] for s in SKENARIO_LIST], key="imp_skenario")
    imp_dict = detail_pred[skenario_imp].attrs.get("rf_importance", {})
    if imp_dict:
        df_imp = pd.DataFrame({"Fitur": list(imp_dict.keys()), "Importance": list(imp_dict.values())})
        df_imp["Fitur"] = df_imp["Fitur"].map(FEATURE_LABEL)
        fig_imp = px.bar(
            df_imp.sort_values("Importance"), x="Importance", y="Fitur", orientation="h",
            color="Importance", color_continuous_scale=SEQ_COLORSCALE,
        )
        fig_imp.update_layout(height=320, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                               coloraxis_showscale=False, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_imp, use_container_width=True)

# ============================================================
# HALAMAN: PREDIKSI TPT (FAKTOR YANG MEMPENGARUHI + TPT 2026)
# ============================================================
elif menu == "Prediksi TPT":
    st.markdown("## 🔮 Prediksi TPT Berdasarkan Faktor Pengaruh")
    tab1, tab2 = st.tabs(["🧮 Prediksi Manual (Input Faktor)", "📅 Proyeksi TPT 2026 (Seluruh Wilayah)"])

    # ---------- TAB 1: PREDIKSI MANUAL ----------
    with tab1:
        st.caption(
            "Masukkan nilai faktor yang memengaruhi TPT (Jumlah Penduduk, TPAK, IPM, PDRB) "
            "untuk melihat estimasi TPT-nya."
        )
        model_choice = st.radio(
            "Pilih model:", ["Random Forest Regressor", "Regresi Linear Berganda"], horizontal=True
        )

        with st.form("form_prediksi"):
            colp1, colp2 = st.columns(2)
            with colp1:
                jumlah_penduduk = st.number_input(
                    FEATURE_LABEL["Jumlah_Penduduk"], min_value=0, value=2_000_000, step=10_000
                )
                tpak = st.slider(FEATURE_LABEL["TPAK"], 40.0, 90.0, 65.0, 0.1)
            with colp2:
                ipm = st.slider(FEATURE_LABEL["IPM"], 50.0, 90.0, 72.0, 0.1)
                pdrb = st.number_input(FEATURE_LABEL["PDRB"], min_value=0, value=5_000_000, step=10_000)
            submit = st.form_submit_button("🔮 Prediksi TPT", use_container_width=True)

        if submit:
            hasil = predict_tpt(full_models, jumlah_penduduk, tpak, ipm, pdrb, model_choice)
            st.markdown(
                f"""
                <div class="hero-box" style="text-align:center;">
                    <div style="font-size:1rem; color:#5C4A75;">Estimasi TPT</div>
                    <div style="font-size:3rem; font-weight:900; color:{C_PURPLE};">{hasil:.2f}%</div>
                    <div style="font-size:0.85rem; color:#8B7AA8;">menggunakan {model_choice}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if hasil >= df_all["TPT"].quantile(0.75):
                st.warning("⚠️ Estimasi TPT tergolong **tinggi** dibanding rata-rata historis Jawa Barat.")
            elif hasil <= df_all["TPT"].quantile(0.25):
                st.success("✅ Estimasi TPT tergolong **rendah** dibanding rata-rata historis Jawa Barat.")
            else:
                st.info("ℹ️ Estimasi TPT berada pada kisaran **menengah** dibanding rata-rata historis.")

    # ---------- TAB 2: PROYEKSI 2026 SELURUH WILAYAH ----------
    with tab2:
        st.caption(
            "Karena belum ada data aktual 2026, faktor (Jumlah Penduduk, TPAK, IPM, PDRB) tiap "
            "kabupaten/kota diproyeksikan dengan ekstrapolasi tren linear 2018–2025, lalu TPT 2026 "
            "diprediksi menggunakan model yang sudah dilatih pada seluruh data historis."
        )
        model_choice_26 = st.radio(
            "Pilih model untuk proyeksi:", ["Random Forest Regressor", "Regresi Linear Berganda"],
            horizontal=True, key="model26",
        )

        if st.button("🚀 Hitung Proyeksi TPT 2026 — Seluruh Kab/Kota", use_container_width=True):
            rows = []
            for kab, grp in df_all.groupby("Kabupaten_Kota"):
                grp = grp.sort_values("Tahun")
                proj = {"Kabupaten_Kota": kab}
                for feat in FEATURES:
                    proj[feat] = proyeksi_linear(grp["Tahun"].values, grp[feat].values, 2026)
                proj["Jumlah_Penduduk"] = max(proj["Jumlah_Penduduk"], 0)
                proj["PDRB"] = max(proj["PDRB"], 0)
                proj["TPAK"] = float(np.clip(proj["TPAK"], 0, 100))
                proj["IPM"] = float(np.clip(proj["IPM"], 0, 100))
                proj["TPT_2026"] = predict_tpt(
                    full_models, proj["Jumlah_Penduduk"], proj["TPAK"], proj["IPM"], proj["PDRB"], model_choice_26
                )
                rows.append(proj)
            df_2026 = pd.DataFrame(rows).sort_values("TPT_2026", ascending=False).reset_index(drop=True)
            st.session_state["df_2026"] = df_2026

        if "df_2026" in st.session_state:
            df_2026 = st.session_state["df_2026"]
            tertinggi26 = df_2026.iloc[0]
            terendah26 = df_2026.iloc[-1]

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(
                    f"""<div class="metric-card"><h3>🔺 TPT 2026 Tertinggi</h3>
                    <div class="big" style="color:{C_PINK};">{tertinggi26['TPT_2026']:.2f}%</div>
                    <div class="sub">{tertinggi26['Kabupaten_Kota']}</div></div>""",
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f"""<div class="metric-card"><h3>🔻 TPT 2026 Terendah</h3>
                    <div class="big" style="color:{C_BLUE};">{terendah26['TPT_2026']:.2f}%</div>
                    <div class="sub">{terendah26['Kabupaten_Kota']}</div></div>""",
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(
                    f"""<div class="metric-card"><h3>📐 Rata-rata Provinsi 2026</h3>
                    <div class="big" style="color:{C_PURPLE};">{df_2026['TPT_2026'].mean():.2f}%</div>
                    <div class="sub">Proyeksi seluruh wilayah</div></div>""",
                    unsafe_allow_html=True,
                )

            st.write("")
            fig26 = px.bar(
                df_2026, x="TPT_2026", y="Kabupaten_Kota", orientation="h",
                color="TPT_2026", color_continuous_scale=SEQ_COLORSCALE,
            )
            fig26.update_layout(
                height=650, yaxis={"categoryorder": "total ascending"},
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False, margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig26, use_container_width=True)

            fig_map26 = px.choropleth(
                df_2026, geojson=geojson_jabar, locations="Kabupaten_Kota",
                featureidkey="properties.Kabupaten_Kota", color="TPT_2026",
                color_continuous_scale=SEQ_COLORSCALE, hover_name="Kabupaten_Kota",
            )
            fig_map26.update_geos(fitbounds="locations", visible=False)
            fig_map26.update_layout(height=500, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_map26, use_container_width=True)

            st.markdown("##### 📋 Tabel Proyeksi 2026")
            tampil = df_2026[["Kabupaten_Kota", "Jumlah_Penduduk", "TPAK", "IPM", "PDRB", "TPT_2026"]].copy()
            tampil.columns = ["Kabupaten/Kota", "Jumlah Penduduk", "TPAK", "IPM", "PDRB", "Prediksi TPT 2026"]
            st.dataframe(
                tampil.style.format(
                    {"Jumlah Penduduk": "{:,.0f}", "TPAK": "{:.2f}", "IPM": "{:.2f}",
                     "PDRB": "{:,.0f}", "Prediksi TPT 2026": "{:.2f}"}
                ).background_gradient(cmap="RdPu", subset=["Prediksi TPT 2026"]),
                use_container_width=True, height=420,
            )
            st.caption(
                "⚠️ Catatan: proyeksi 2026 bersifat estimasi berbasis tren historis, bukan data resmi BPS."
            )
        else:
            st.info("Klik tombol di atas untuk menghitung proyeksi TPT 2026 seluruh kabupaten/kota.")

# ============================================================
# HALAMAN: DATA LENGKAP
# ============================================================
elif menu == "Data Lengkap":
    st.markdown("## 📋 Tabel Data TPT Jawa Barat (2018–2025)")

    colf1, colf2 = st.columns(2)
    with colf1:
        wilayah_filter = st.multiselect(
            "Filter Kabupaten/Kota:", sorted(df_all["Kabupaten_Kota"].unique())
        )
    with colf2:
        tahun_filter = st.multiselect("Filter Tahun:", TAHUN_LIST)

    df_filtered = df_all.copy()
    if wilayah_filter:
        df_filtered = df_filtered[df_filtered["Kabupaten_Kota"].isin(wilayah_filter)]
    if tahun_filter:
        df_filtered = df_filtered[df_filtered["Tahun"].isin(tahun_filter)]

    df_filtered = df_filtered.sort_values(["Kabupaten_Kota", "Tahun"]).reset_index(drop=True)
    st.dataframe(
        df_filtered.style.format(
            {"IPM": "{:.2f}", "TPT": "{:.2f}", "Jumlah_Penduduk": "{:,.0f}", "PDRB": "{:,.0f}", "TPAK": "{:.2f}"}
        ).background_gradient(cmap="PuRd", subset=["TPT"]),
        use_container_width=True, height=520,
    )

    st.download_button(
        "⬇️ Unduh data (CSV)",
        df_filtered.to_csv(index=False).encode("utf-8"),
        file_name="data_tpt_jabar.csv",
        mime="text/csv",
    )

    st.write("")
    st.markdown("##### 📊 Statistik Deskriptif")
    st.dataframe(
        df_filtered[FEATURES + [TARGET]].describe().T.style.format("{:.2f}"),
        use_container_width=True,
    )

# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """
    <div class="footer-note">
    Dashboard Prediksi TPT Provinsi Jawa Barat · Regresi Linear Berganda & Random Forest Regressor
    </div>
    """,
    unsafe_allow_html=True,
)
