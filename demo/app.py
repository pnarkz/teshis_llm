"""Presentation console for the thermal diagnosis project.

Tasarim yonu: olcum aleti / muhendislik konsolu. Monospace tipografi, keskin
koseler, ince izgara cizgileri; gradyan, golge ve yuvarlak kose yok. Yazi tipi
olarak yalnizca sistemde hazir bulunan monospace aileleri kullanilir; sunum
sirasinda internet olmasa da gorunum bozulmaz.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import (  # noqa: E402
    curves_for,
    error_galleries,
    evidence_for,
    examples_for,
    images_for,
    label_distribution_image,
    llm_response,
    llm_score,
    load_results,
    sparkline,
    train_batch_images,
    training_curve,
)

st.set_page_config(page_title="TESHIS//KONSOL", page_icon="▚", layout="wide", initial_sidebar_state="expanded")

MONO = 'ui-monospace, "Cascadia Mono", "Segoe UI Mono", Consolas, "DejaVu Sans Mono", monospace'
PHOSPHOR = "#3ee0a3"
AMBER = "#f0a45c"
RED = "#ff6b5f"
CYAN = "#5cc8f0"

st.markdown(
    f"""
    <style>
    :root {{
      --bg:#080a0b; --panel:#0f1315; --panel2:#131819; --line:#212a2c;
      --ink:#d4dedd; --dim:#68797a; --phos:{PHOSPHOR}; --amber:{AMBER}; --red:{RED};
    }}
    .stApp {{ background:var(--bg); color:var(--ink); }}
    html, body, [class*="css"], .stMarkdown, p, li, span, div, label,
    h1, h2, h3, h4, h5, h6, .stCaption {{ font-family:{MONO} !important; }}
    * {{ border-radius:0 !important; box-shadow:none !important; }}
    [data-testid="stHeader"] {{ background:transparent; }}
    .block-container {{ max-width:1420px; padding:1.4rem 2.2rem 3rem; }}

    [data-testid="stSidebar"] {{ background:#0b0e0f; border-right:1px solid var(--line); }}
    [data-testid="stSidebar"] * {{ color:var(--ink) !important; font-family:{MONO} !important; }}

    /* ---- konsol basligi ---- */
    .masthead {{ border:1px solid var(--line); border-left:3px solid var(--phos);
      background:var(--panel); padding:.85rem 1.1rem; margin-bottom:1.1rem;
      display:flex; justify-content:space-between; align-items:baseline; gap:1rem; flex-wrap:wrap; }}
    .masthead .id {{ font-size:1.35rem; font-weight:700; letter-spacing:.16em; color:var(--ink); }}
    .masthead .id b {{ color:var(--phos); font-weight:700; }}
    .masthead .meta {{ font-size:.72rem; letter-spacing:.13em; color:var(--dim); text-transform:uppercase; }}

    /* ---- bolum basliklari ---- */
    .rule {{ display:flex; align-items:center; gap:.7rem; margin:1.9rem 0 .8rem; }}
    .rule .tag {{ font-size:.7rem; letter-spacing:.2em; color:var(--phos);
      text-transform:uppercase; white-space:nowrap; }}
    .rule .bar {{ flex:1; height:1px; background:var(--line); }}
    .rule .n {{ font-size:.7rem; color:var(--dim); letter-spacing:.12em; }}

    /* ---- readout tablosu ---- */
    .readout {{ border:1px solid var(--line); background:var(--panel); }}
    .readout .row {{ display:grid; grid-template-columns:1fr auto auto;
      gap:1rem; padding:.5rem .9rem; border-bottom:1px solid var(--line); align-items:baseline; }}
    .readout .row:last-child {{ border-bottom:none; }}
    .readout .k {{ font-size:.72rem; letter-spacing:.15em; color:var(--dim); text-transform:uppercase; }}
    .readout .v {{ font-size:1.02rem; font-variant-numeric:tabular-nums; color:var(--ink); }}
    .readout .d {{ font-size:.78rem; font-variant-numeric:tabular-nums; min-width:8.5ch; text-align:right; }}
    .up {{ color:var(--phos); }} .down {{ color:var(--amber); }} .flat {{ color:var(--dim); }}

    /* ---- panel / not ---- */
    .panel {{ border:1px solid var(--line); background:var(--panel); padding:.85rem 1rem; }}
    .panel.alert {{ border-left:3px solid var(--amber); }}
    .panel.ok {{ border-left:3px solid var(--phos); }}
    .panel .lbl {{ font-size:.68rem; letter-spacing:.2em; color:var(--dim);
      text-transform:uppercase; display:block; margin-bottom:.35rem; }}
    .spark {{ font-size:.95rem; letter-spacing:.06em; color:var(--phos); line-height:1.1; }}
    .kv {{ font-size:.78rem; color:var(--dim); }}
    .kv b {{ color:var(--ink); font-weight:600; }}

    /* ---- streamlit bilesenleri ---- */
    [data-testid="stMetric"] {{ background:var(--panel); border:1px solid var(--line); padding:.6rem .8rem; }}
    [data-testid="stMetricLabel"] {{ color:var(--dim) !important; }}
    [data-testid="stMetricValue"] {{ color:var(--ink) !important; font-family:{MONO} !important; }}
    div[data-testid="stDataFrame"] {{ border:1px solid var(--line); }}
    div[data-testid="stDataFrame"] [role="gridcell"],
    div[data-testid="stDataFrame"] [role="columnheader"] {{
      color:var(--ink) !important; background:var(--panel) !important; font-family:{MONO} !important; }}
    .stCaption, [data-testid="stCaptionContainer"] {{ color:var(--dim) !important; font-size:.72rem !important; }}
    h1,h2,h3,h4,h5,h6 {{ color:var(--ink) !important; letter-spacing:.04em; }}
    .stTabs [data-baseweb="tab-list"] {{ gap:0; border-bottom:1px solid var(--line); }}
    .stTabs [data-baseweb="tab"] {{ font-family:{MONO} !important; font-size:.74rem;
      letter-spacing:.14em; text-transform:uppercase; color:var(--dim); background:transparent;
      border:1px solid transparent; border-bottom:none; padding:.5rem 1rem; }}
    .stTabs [aria-selected="true"] {{ color:var(--phos) !important;
      border-color:var(--line); background:var(--panel); }}
    details {{ border:1px solid var(--line) !important; background:var(--panel) !important; }}
    summary {{ font-family:{MONO} !important; font-size:.76rem !important; letter-spacing:.1em; }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def results_data() -> pd.DataFrame:
    return load_results()


results = results_data()
baseline = results[results["scenario"] == "Baseline"].iloc[0]
scenarios = results["scenario"].tolist()
METRICS = (("mAP50", "mAP50"), ("mAP50-95", "mAP50_95"), ("PRECISION", "precision"), ("RECALL", "recall"))
CLASS_COLUMNS = ["AP_tasit", "AP_insan", "AP_UAP", "AP_UAI"]
CLASS_NAMES = ["tasit", "insan", "UAP", "UAI"]
CHART_COLORS = [PHOSPHOR, CYAN, AMBER, RED]


def rule(tag: str, note: str = "") -> None:
    st.markdown(
        f'<div class="rule"><span class="tag">{tag}</span><span class="bar"></span>'
        f'<span class="n">{note}</span></div>',
        unsafe_allow_html=True,
    )


def readout(row: pd.Series, reference: pd.Series | None = None) -> str:
    lines = []
    for label, field in METRICS:
        value = float(row[field])
        if reference is None:
            delta_html = '<span class="d flat">&mdash;</span>'
        else:
            delta = value - float(reference[field])
            css = "flat" if abs(delta) < 5e-5 else ("up" if delta > 0 else "down")
            arrow = "─" if css == "flat" else ("▴" if delta > 0 else "▾")
            delta_html = f'<span class="d {css}">{arrow} {delta:+.4f}</span>'
        lines.append(
            f'<div class="row"><span class="k">{label}</span>'
            f'<span class="v">{value:.4f}</span>{delta_html}</div>'
        )
    return f'<div class="readout">{"".join(lines)}</div>'


VARSAYILAN_BILGI = {
    "problem": "Aciklama girilmemis",
    "change": "-",
    "signal": "-",
    "why": "Bu senaryo icin demo/app.py::scenario_info altina aciklama eklenmedi.",
    "status": "Bilinmiyor",
}

scenario_info = {
    "Baseline": {
        "problem": "Saglikli referans (fine-tune YOK)",
        "change": "main_model.pt dogrudan olculur; hicbir egitim yapilmaz.",
        "signal": "Dagitimdaki modelin oldugu gibi performansi.",
        "why": "Senaryolarla ayni protokolde egitilmedigi icin bozulma karsilastirmasinda taban olarak KULLANILMAZ; onun yerine v00 kullanilir.",
        "status": "Tamamlandi",
    },
    "v00_saglikli": {
        "problem": "Saglikli referans (ortak protokolle egitildi)",
        "change": "Veri hic bozulmaz; senaryolarla birebir ayni protokolde fine-tune edilir.",
        "signal": "Bozulmanin degil, yalnizca fine-tune'un kendi etkisini gosterir.",
        "why": "Senaryo farklarinin 'bozulma etkisi' mi yoksa 'fine-tune etkisi' mi oldugunu ayirmak icin gereken dogru taban budur.",
        "status": "Tamamlandi",
    },
    "D1": {
        "problem": "Sinif yetersizligi",
        "change": "Insan sinifini iceren egitim karelerinin buyuk bolumu azaltildi.",
        "signal": "Insan AP ve recall degerlerinde dusus beklenir.",
        "why": "Modelin az temsil edilen bir sinifi ne kadar ogrenebildigini test eder.",
        "status": "Tamamlandi",
    },
    "D2a": {
        "problem": "Lokalizasyon etiketi gurultusu",
        "change": "Train bounding box merkezleri kontrollu oranda kaydirildi.",
        "signal": "mAP50-95, mAP50'den daha fazla dusebilir.",
        "why": "Kutunun sinifi dogru olsa bile konumunun hatali olmasinin etkisini ayirir.",
        "status": "Tamamlandi",
    },
    "D2b": {
        "problem": "Eksik etiket",
        "change": "Train etiket satirlarinin %25'i silindi; val/test degistirilmedi.",
        "signal": "Precision dususu ve yanlis pozitif artisi beklenir.",
        "why": "Gercekteki eksik anotasyonlarin egitimi nasil yanilttigini olcer.",
        "status": "Tamamlandi",
    },
    "D2b final_best": {
        "problem": "Eksik etiket + farkli baslangic modeli",
        "change": "D2b protokolu final_best.pt ile tekrarlandi.",
        "signal": "Ayni veri bozulmasinda model baslangicinin etkisi gorulur.",
        "why": "Sonucun yalnizca veri bozulmasindan mi, yoksa model farkindan mi geldigini kontrol eder.",
        "status": "Tamamlandi",
    },
    "D5": {
        "problem": "Kaynak / alan kaymasi",
        "change": "Egitim seti yalnizca aaterm kaynagiyla sinirlandi (17.515 -> 11.064 kare); etiketler degismedi.",
        "signal": "Egitimde gorulmeyen kaynaklarda recall duser; aaterm korunur.",
        "why": "Tek kaynaktan veri toplamanin, dagitimda farkli sensor/sahnelerde yarattigi riski olcer.",
        "status": "Tamamlandi",
    },
    "D4": {
        "problem": "Kucuk nesne sinyal kaybi",
        "change": "Egitim etiketlerinden etkin boyutu 16 px altindaki 29.499 kutu silindi (%22,4).",
        "signal": "Yalnizca <16 px bandinda recall coker; diger bantlar degismez.",
        "why": "Toplam mAP'in gizledigi boyut-ozgu bir bozulmayi olcer; D1'den ayrimi sinif x boyut kirilimiyla yapilir.",
        "status": "Tamamlandi",
    },
    "D3b": {
        "problem": "tasit/insan sinif karisikligi (olculebilir surum)",
        "change": "Train etiketlerinde tasit (0) ve insan (1) siniflarinin %30'u karistirildi (39.393 satir).",
        "signal": "AP/precision neredeyse degismez; capraz hata yalnizca sabit esikli confusion matrix'te gorunur.",
        "why": "D3 ile ayni bozulmayi bol ornekli siniflara uygular; boylece etki 3.982 kutuyla olculebilir hale gelir.",
        "status": "Tamamlandi",
    },
    "D3": {
        "problem": "UAP/UAI sinif karisikligi",
        "change": "Train etiketlerinde UAP (2) ve UAI (3) siniflarinin %30'u kontrollu olarak birbirine karistirildi.",
        "signal": "UAP/UAI arasinda confusion matrix'te capraz hata artisi beklenir.",
        "why": "Sinif ID etiketleme hatasinin, ozellikle az orneli siniflarda etkisini olcer.",
        "status": "Tamamlandi",
    },
}

st.markdown(
    '<div class="masthead"><span class="id">TESHIS<b>//</b>KONSOL</span>'
    '<span class="meta">kontrollu bozulma diagnostigi &middot; val_diagnostic 1.056 goruntu &middot; '
    "test seti kullanilmadi</span></div>",
    unsafe_allow_html=True,
)

st.sidebar.markdown('<div class="rule"><span class="tag">bolum</span><span class="bar"></span></div>', unsafe_allow_html=True)
page = st.sidebar.radio(
    "Bolum",
    ["Genel Bakis", "Senaryo Incele", "Hata Galerisi", "Proje ve Senaryolar", "LLM Ajan"],
    label_visibility="collapsed",
)
st.sidebar.markdown(
    f'<div class="panel ok" style="margin-top:1rem"><span class="lbl">durum</span>'
    f'<span class="kv"><b>{len(scenarios)}</b> kosu yuklendi<br><b>{len(error_galleries())}</b> hata galerisi<br>'
    "kaynak: reports/ + experiments/</span></div>",
    unsafe_allow_html=True,
)
st.sidebar.caption("Bu konsol mevcut raporlari okur. Egitim veya test calistirmaz.")


if page == "Genel Bakis":
    rule("00 // yonetici ozeti", f"{len(scenarios)} kosu")
    cards = st.columns(4)
    for card, (label, value) in zip(
        cards,
        (("KOSU", str(len(scenarios))), ("DIAGNOSTIC GORUNTU", "1.056"), ("SINIF", "4"), ("TEST KULLANIMI", "YOK")),
    ):
        card.markdown(
            f'<div class="panel"><span class="lbl">{label}</span>'
            f'<span style="font-size:1.5rem">{value}</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="panel alert" style="margin-top:1rem"><span class="lbl">ana bulgu</span>'
        "<b>Her veri problemi farkli bir metrigi bozuyor.</b> D2a konum gurultusu mAP50-95'i, "
        "D2b eksik etiket precision'i, D3 sinif karisikligi ise UAP/UAI ayrimini bozdu. "
        "Bu ayrisma, bozulmanin nedenini metrik imzasindan okumayi mumkun kiliyor.</div>",
        unsafe_allow_html=True,
    )

    rule("01 // kosu kayitlari", "epoch egrisi = egitim ilerlemesi")
    for _, row in results.iterrows():
        scenario = row["scenario"]
        info = scenario_info.get(scenario, VARSAYILAN_BILGI)
        left, right = st.columns([1.05, 1])
        with left:
            st.markdown(
                f'<div class="panel"><span class="lbl">{scenario} &middot; {info.get("problem", "")}</span>'
                f'<span class="kv">run_id <b>{row["run_id"]}</b> &middot; veri <b>{row["data_version"]}</b><br>'
                f'model <b>{row["model"]}</b> &middot; seed <b>{row["seed"]}</b> &middot; '
                f'epoch <b>{row["epochs"]}</b> &middot; batch <b>{row["batch"]}</b></span></div>',
                unsafe_allow_html=True,
            )
            curve = training_curve(row)
            if not curve.empty:
                st.markdown(
                    '<div class="panel" style="border-top:none">'
                    f'<span class="lbl">mAP50 &middot; epoch 1&rarr;{len(curve)}</span>'
                    f'<div class="spark">{sparkline(curve["metrics/mAP50(B)"].tolist())}</div>'
                    f'<span class="lbl" style="margin-top:.5rem">train box_loss</span>'
                    f'<div class="spark" style="color:var(--amber)">{sparkline(curve["train/box_loss"].tolist())}</div>'
                    "</div>",
                    unsafe_allow_html=True,
                )
        with right:
            st.markdown(
                readout(row, None if scenario == "Baseline" else baseline),
                unsafe_allow_html=True,
            )

    rule("02 // karsilastirma tablosu")
    display = results[["scenario", "mAP50", "mAP50_95", "precision", "recall", *CLASS_COLUMNS]].copy()
    display.columns = ["Kosu", "mAP50", "mAP50-95", "Precision", "Recall", *CLASS_NAMES]
    st.dataframe(
        display.style.format({column: "{:.4f}" for column in display.columns[1:]}),
        width="stretch",
        hide_index=True,
    )

    rule("03 // metrik profili")
    chart = results.set_index("scenario")[["mAP50", "mAP50_95", "precision", "recall"]].astype(float)
    # stack=False: bu metrikler toplanabilir buyuklukler degil, yan yana karsilastirilir.
    st.bar_chart(chart, height=320, color=CHART_COLORS, stack=False)

    rule("04 // sinif bazli AP50", "UAP n=15 &middot; UAI n=17 &rarr; belirsizlik yuksek")
    class_table = results.set_index("scenario")[CLASS_COLUMNS].astype(float)
    class_table.columns = CLASS_NAMES
    st.bar_chart(class_table, height=320, color=CHART_COLORS, stack=False)


elif page == "Senaryo Incele":
    selected = st.sidebar.selectbox("Kosu", scenarios, index=0)
    row = results[results["scenario"] == selected].iloc[0]
    info = scenario_info.get(selected, VARSAYILAN_BILGI)

    rule(f"// {selected}", info["problem"])
    left, right = st.columns([1, 1])
    left.markdown(readout(row, None if selected == "Baseline" else baseline), unsafe_allow_html=True)
    right.markdown(
        f'<div class="panel alert"><span class="lbl">uygulanan degisiklik</span>'
        f'<span class="kv">{info["change"]}</span></div>'
        f'<div class="panel" style="border-top:none"><span class="lbl">beklenen sinyal</span>'
        f'<span class="kv">{info["signal"]}</span></div>'
        f'<div class="panel" style="border-top:none"><span class="lbl">neden</span>'
        f'<span class="kv">{info["why"]}</span></div>',
        unsafe_allow_html=True,
    )

    curve = training_curve(row)
    if not curve.empty:
        rule("01 // egitim egrisi", f"epoch 1&rarr;{len(curve)}")
        metric_tab, loss_tab = st.tabs(["Dogrulama metrikleri", "Egitim kayiplari"])
        with metric_tab:
            frame = curve.set_index("epoch")[
                ["metrics/mAP50(B)", "metrics/mAP50-95(B)", "metrics/precision(B)", "metrics/recall(B)"]
            ]
            frame.columns = ["mAP50", "mAP50-95", "precision", "recall"]
            st.line_chart(frame, height=300, color=CHART_COLORS)
            st.caption("Bu egriler egitim sirasindaki kendi val bolmesinden gelir; ust readout ise kilitli val_diagnostic olcumudur.")
        with loss_tab:
            losses = curve.set_index("epoch")[["train/box_loss", "train/cls_loss", "val/box_loss", "val/cls_loss"]]
            losses.columns = ["train box", "train cls", "val box", "val cls"]
            st.line_chart(losses, height=300, color=CHART_COLORS)

    rule("02 // sinif bazli AP50")
    class_frame = results[results["scenario"] == selected][CLASS_COLUMNS].T
    class_frame.columns = ["AP50"]
    class_frame.index = CLASS_NAMES
    chart_col, evidence_col = st.columns([1, 1.15])
    chart_col.bar_chart(class_frame, height=280, color=PHOSPHOR)
    with evidence_col:
        with st.expander("Kayitli kanit (JSON)", expanded=False):
            evidence = evidence_for(selected)
            st.json(evidence if evidence else {"durum": "rapor bulunamadi"})

    matrices = images_for(selected)
    if matrices:
        rule("03 // confusion matrix", "sabit esik &middot; argmax sinif")
        matrix_cols = st.columns(len(matrices))
        for column, image in zip(matrix_cols, matrices):
            column.image(str(image), caption=image.stem, width="stretch")

    curve_images = curves_for(selected)
    if curve_images:
        rule("04 // esik egrileri", "guven esigine gore davranis")
        curve_cols = st.columns(2)
        for index, (path, label) in enumerate(curve_images):
            curve_cols[index % 2].image(str(path), caption=label, width="stretch")

    examples = examples_for(selected)
    if examples:
        rule("05 // etiket vs tahmin", "sol: gercek etiket &middot; sag: model tahmini")
        for index in range(0, len(examples), 2):
            pair = examples[index : index + 2]
            pair_cols = st.columns(len(pair))
            for column, image in zip(pair_cols, pair):
                kind = "GERCEK ETIKET" if "labels" in image.name else "MODEL TAHMINI"
                column.image(str(image), caption=f"{kind} · {image.stem}", width="stretch")

    batches = train_batch_images(row)
    label_plot = label_distribution_image(row)
    if batches or label_plot:
        rule("06 // egitim verisi", "bozulmus veri modele bu haliyle verildi")
        if label_plot:
            st.image(str(label_plot), caption="Egitim setindeki sinif ve bbox dagilimi", width="stretch")
        if batches:
            batch_cols = st.columns(len(batches))
            for column, image in zip(batch_cols, batches):
                column.image(str(image), caption=image.stem, width="stretch")


elif page == "Hata Galerisi":
    galleries = error_galleries()
    if not galleries:
        st.markdown(
            '<div class="panel alert"><span class="lbl">galeri yok</span>'
            "reports/ altinda *_hata_galerisi klasoru bulunamadi.</div>",
            unsafe_allow_html=True,
        )
    else:
        gallery_key = st.sidebar.selectbox("Galeri", sorted(galleries), index=0)
        gallery = galleries[gallery_key]
        entries = pd.DataFrame(gallery["entries"])

        rule(f"// {gallery_key} hata galerisi", f"{len(entries)} goruntu")
        st.markdown(
            '<div class="panel alert"><span class="lbl">okuma notu</span>'
            "Bu goruntuler 1.056 diagnostic karesi icinde en yuksek hata skoruna sahip olanlardir. "
            "<b>Yesil</b> kutular gercek etiketi, <b>kirmizi</b> kutular model tahminini gosterir. "
            "Skor = yanlis negatif ve yanlis pozitif sayisinin, ortalama IoU ile agirliklandirilmis "
            "birlesimi; yuksek skor daha bozuk bir kareyi isaret eder.</div>",
            unsafe_allow_html=True,
        )

        rule("01 // hata dagilimi")
        stat_cols = st.columns(4)
        for column, (label, value) in zip(
            stat_cols,
            (
                ("TOPLAM FN", f'{int(entries["false_negatives"].sum())}'),
                ("TOPLAM FP", f'{int(entries["false_positives"].sum())}'),
                ("ORT. IoU", f'{entries["mean_iou"].mean():.3f}'),
                ("EN YUKSEK SKOR", f'{entries["score"].max():.1f}'),
            ),
        ):
            column.markdown(
                f'<div class="panel"><span class="lbl">{label}</span>'
                f'<span style="font-size:1.4rem">{value}</span></div>',
                unsafe_allow_html=True,
            )
        scatter = entries[["false_negatives", "false_positives", "mean_iou", "score"]].copy()
        st.scatter_chart(scatter, x="false_negatives", y="false_positives", size="score", height=300, color=RED)
        st.caption("Sag ust kose: hem cok kacirma hem cok yanlis pozitif ureten kareler.")

        rule("02 // kareler", "skora gore azalan")
        order = st.sidebar.radio("Siralama", ["Skor", "Yanlis negatif", "Yanlis pozitif", "Dusuk IoU"], index=0)
        sort_field = {
            "Skor": ("score", False),
            "Yanlis negatif": ("false_negatives", False),
            "Yanlis pozitif": ("false_positives", False),
            "Dusuk IoU": ("mean_iou", True),
        }[order]
        count = st.sidebar.slider("Gosterilecek kare", 4, min(50, len(entries)), 12, step=4)
        ordered = entries.sort_values(sort_field[0], ascending=sort_field[1]).head(count)

        for index in range(0, len(ordered), 2):
            chunk = ordered.iloc[index : index + 2]
            columns = st.columns(len(chunk))
            for column, (_, entry) in zip(columns, chunk.iterrows()):
                image_path = gallery["folder"] / entry["image"]
                if image_path.is_file():
                    column.image(str(image_path), width="stretch")
                column.markdown(
                    f'<div class="panel" style="border-top:none"><span class="lbl">{entry["source"]}</span>'
                    f'<span class="kv">FN <b>{int(entry["false_negatives"])}</b> &middot; '
                    f'FP <b>{int(entry["false_positives"])}</b> &middot; '
                    f'IoU <b>{entry["mean_iou"]:.3f}</b> &middot; '
                    f'skor <b>{entry["score"]:.1f}</b></span></div>',
                    unsafe_allow_html=True,
                )


elif page == "LLM Ajan":
    rule("// llm ajan denemesi", "anonim metrik yorumlama")
    st.markdown(
        '<div class="panel alert"><span class="lbl">pilot testi</span>'
        "LLM senaryo isimlerini gormeden, yalnizca anonim <b>kosu_NN</b> metriklerinden teshis, "
        "kanit, guven ve sonraki olcumu uretir. Bu bir benchmark degil, ajan gelistirme icin "
        "pilot rubriktir.</div>",
        unsafe_allow_html=True,
    )
    response = llm_response()
    score = llm_score()
    if score:
        rule("01 // puanlama", "diagnosis + evidence + limitations, esit agirlik")
        score_cols = st.columns(3)
        for column, (label, value) in zip(
            score_cols,
            (
                ("ORTALAMA SKOR", f'{score.get("mean_score", 0):.3f}'),
                ("DEGERLENDIRILEN KOSU", str(len(score.get("runs", [])))),
                ("RUBRIK ALANI", "3"),
            ),
        ):
            column.markdown(
                f'<div class="panel"><span class="lbl">{label}</span>'
                f'<span style="font-size:1.5rem">{value}</span></div>',
                unsafe_allow_html=True,
            )
        rows = pd.DataFrame(score.get("runs", []))
        if not rows.empty:
            breakdown = rows.set_index("run_id")[["diagnosis_score", "evidence_score", "limitation_score"]]
            breakdown.columns = ["teshis", "kanit", "sinirlama"]
            st.bar_chart(breakdown, height=280, color=CHART_COLORS[:3], stack=False)
            st.caption("Teshis sutunu dusuk olan kosularda LLM metrigi dogru okudu ama nedeni adlandiramadi.")

    if response:
        rule("02 // ajan ciktilari")
        for item in response if isinstance(response, list) else [response]:
            run_id = item.get("run_id", "kosu")
            with st.expander(f"{run_id} · {item.get('diagnosis', '-')}", expanded=run_id == "kosu_01"):
                st.markdown(
                    f'<div class="panel"><span class="lbl">teshis</span>'
                    f'<span class="kv"><b>{item.get("diagnosis", "-")}</b> &middot; '
                    f'guven <b>{item.get("confidence", "-")}</b></span></div>',
                    unsafe_allow_html=True,
                )
                st.markdown("**Kanitlar**")
                for evidence in item.get("evidence", []):
                    st.write(f"- {evidence}")
                st.markdown("**Sinirlamalar**")
                for limitation in item.get("limitations", []):
                    st.write(f"- {limitation}")
                st.caption(f"Sonraki olcum: {item.get('next_measurement', '-')}")
    else:
        st.info("Gemini cikti dosyasi bulunamadi.")


else:
    rule("// proje ve senaryolar", "kontrollu deney tasarimi")
    st.markdown(
        '<div class="panel alert"><span class="lbl">temel soru</span>'
        "Modelin basarisi dustugunde sorun modelin kendisi mi, veri seti mi, etiketler mi, "
        "yoksa belirli bir sinifin yetersiz temsil edilmesi mi?</div>",
        unsafe_allow_html=True,
    )

    rule("01 // deney akisi")
    steps = st.columns(4)
    for column, (number, title, text) in zip(
        steps,
        (
            ("01", "BASELINE", "Saglikli model ve kilitli diagnostic set olusturulur."),
            ("02", "TEK DEGISKEN", "Sadece bir veri problemi kontrollu olarak uygulanir."),
            ("03", "YENIDEN EGITIM", "Ayni egitim protokoluyle yeni model kosulur."),
            ("04", "KARSILASTIRMA", "Metrik ve gorsel kanit baseline ile karsilastirilir."),
        ),
    ):
        column.markdown(
            f'<div class="panel"><span class="lbl">{number} &middot; {title}</span>'
            f'<span class="kv">{text}</span></div>',
            unsafe_allow_html=True,
        )

    rule("02 // senaryo katalogu", f"{len(scenarios)} tamamlandi")
    for name in scenarios:
        info = scenario_info.get(name, VARSAYILAN_BILGI)
        with st.expander(f"{name} · {info['problem']}", expanded=name == "Baseline"):
            columns = st.columns(2)
            columns[0].markdown(
                f'<div class="panel"><span class="lbl">nasil olusturuldu</span>'
                f'<span class="kv">{info["change"]}</span></div>',
                unsafe_allow_html=True,
            )
            columns[1].markdown(
                f'<div class="panel"><span class="lbl">neden olusturuldu</span>'
                f'<span class="kv">{info["why"]}</span></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="panel ok" style="border-top:none"><span class="lbl">beklenen / gozlenen sinyal</span>'
                f'<span class="kv">{info["signal"]}</span></div>',
                unsafe_allow_html=True,
            )

    rule("03 // test seti neden kullanilmiyor")
    st.markdown(
        '<div class="panel"><span class="kv">Test seti son karari vermek icin saklanir. Senaryolar, model '
        "secimini veya final performansi test setine bakarak etkilemesin diye yalnizca kilitli "
        "val_diagnostic setinde incelenir.</span></div>",
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="rule" style="margin-top:2.4rem"><span class="bar"></span></div>'
    '<div class="kv" style="text-align:center">TESHIS//KONSOL &middot; ara sunum &middot; '
    "test seti final asamasina kadar kullanilmaz</div>",
    unsafe_allow_html=True,
)
