"""Presentation dashboard for the thermal diagnosis project."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import examples_for, evidence_for, images_for, llm_response, llm_score, load_results  # noqa: E402


st.set_page_config(page_title="Termal Teshis Ajani", page_icon="", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    :root { --ink:#e8f0eb; --muted:#a7bbb7; --cream:#0d1b1e; --orange:#e19a59; --teal:#1d5b5f; }
    .stApp { background: var(--cream); color:var(--ink); }
    [data-testid="stHeader"] { background:transparent; }
    [data-testid="stToolbar"] { color:var(--ink); }
    .block-container { max-width: 1280px; padding: 2.2rem 3rem 3rem; }
    [data-testid="stSidebar"] { background: #081417; border-right:1px solid #28464a; }
    [data-testid="stSidebar"] * { color: #edf2ed !important; }
    .hero { padding: 2rem 2.2rem; border-radius: 22px; background: linear-gradient(120deg,#173238 0%,#1d5b5f 68%,#d47732 170%); color:#f7f3e9; box-shadow: 0 14px 35px #17323822; }
    .hero h1 { margin:0; font-size:2.5rem; letter-spacing:-.04em; }
    .hero p { margin:.65rem 0 0; color:#d7e6df; font-size:1.05rem; }
    .eyebrow { text-transform:uppercase; letter-spacing:.14em; font-size:.72rem; font-weight:700; color:#e7ad79; }
    .section-title { color:var(--ink); margin-top:1.8rem; margin-bottom:.25rem; }
    .section-subtitle { color:var(--muted); margin-bottom:1rem; }
    .callout { padding:1rem 1.1rem; border-left:4px solid var(--orange); background:#173238; color:#e8f0eb; border-radius:8px; }
    .status { display:inline-block; padding:.3rem .65rem; border-radius:99px; background:#194d43; color:#bfe8d4; font-size:.78rem; font-weight:700; }
    [data-testid="stMetric"] { background:#142b30; border:1px solid #2a4b4e; padding:.65rem .8rem; border-radius:12px; }
    [data-testid="stMetricLabel"] { color:#a7bbb7 !important; }
    [data-testid="stMetricValue"] { color:#f2f7f1 !important; }
    [data-testid="stMetricDelta"] { color:#e19a59 !important; }
    div[data-testid="stDataFrame"] { border:1px solid #2a4b4e; border-radius:10px; background:#142b30; }
    div[data-testid="stDataFrame"] [role="gridcell"], div[data-testid="stDataFrame"] [role="columnheader"] { color:#e8f0eb !important; background:#142b30 !important; }
    .stCaption, [data-testid="stCaptionContainer"] { color:#a7bbb7 !important; }
    h1, h2, h3, h4, h5, h6 { color:#e8f0eb !important; }
    .stMarkdown, .stText, p, li { color:#d5e2dd; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def results_data():
    return load_results()


results = results_data()
baseline = results[results["scenario"] == "Baseline"].iloc[0]
scenarios = results["scenario"].tolist()

st.markdown(
    '<div class="hero"><div class="eyebrow">Kontrollu model diagnostigi</div><h1>Termal Teshis Ajani</h1><p>Modelin sadece ne kadar iyi oldugunu degil, neden bozuldugunu gosteren deney panosu.</p></div>',
    unsafe_allow_html=True,
)

st.sidebar.markdown("## PROJE KONSOLU")
page = st.sidebar.radio("Bolum", ["Genel Bakis", "Proje ve Senaryolar", "Senaryo Incele", "LLM Ajan"], label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.markdown("**Sistem durumu**")
st.sidebar.markdown('<span class="status">Raporlar hazir</span>', unsafe_allow_html=True)
st.sidebar.caption("Bu demo mevcut raporlari okur. Egitim ve test calistirmaz.")

scenario_info = {
    "Baseline": {
        "problem": "Saglikli referans",
        "change": "Veri veya etiketler kasitli olarak bozulmaz.",
        "signal": "Diger kosularin olcum noktasi olarak kullanilir.",
        "why": "Bir bozulmanin etkisini soyleyebilmek icin once normal performansi bilmek gerekir.",
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
        "signal": "mAP50-95, mAP50'den daha fazla dusabilir.",
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
    "D3": {
        "problem": "UAP/UAI sinif karisikligi",
        "change": "Train etiketlerinde UAP (2) ve UAI (3) siniflarinin %30'u kontrollu olarak birbirine karistirildi.",
        "signal": "UAP/UAI arasinda confusion matrix'te capraz hata artisi beklenir.",
        "why": "Sinif ID etiketleme hatasinin, ozellikle az orneli siniflarda etkisini olcer.",
        "status": "Tamamlandi",
    },
}


if page == "Genel Bakis":
    st.markdown('<h2 class="section-title">Yonetici Ozeti</h2><p class="section-subtitle">Ara sunum icin hazirlanan mevcut deneylerin tek bakista gorunumu.</p>', unsafe_allow_html=True)
    cards = st.columns(4)
    cards[0].metric("Tamamlanan kosu", str(len(scenarios)))
    cards[1].metric("Diagnostic goruntu", "1.056")
    cards[2].metric("Sinif", "4")
    cards[3].metric("Test kullanimi", "Yok")

    st.markdown('<h3 class="section-title">Ana bulgu</h3>', unsafe_allow_html=True)
    st.markdown('<div class="callout"><b>Veri problemi performansi degistiriyor.</b> D2a konum etiketi gurultusu mAP50-95 degerini dusururken, D2b eksik etiketler precision dususune ve daha fazla yanlis pozitif uretilmesine yol acti.</div>', unsafe_allow_html=True)

    st.markdown('<h3 class="section-title">Kosularin karsilastirmasi</h3>', unsafe_allow_html=True)
    display = results[["scenario", "mAP50", "mAP50_95", "precision", "recall", "AP_tasit", "AP_insan"]].copy()
    display.columns = ["Kosu", "mAP50", "mAP50-95", "Precision", "Recall", "AP tasit", "AP insan"]
    st.dataframe(display.style.format({column: "{:.4f}" for column in display.columns[1:]}), use_container_width=True, hide_index=True)
    chart = results.set_index("scenario")[["mAP50", "mAP50_95", "precision", "recall"]].astype(float)
    st.bar_chart(chart, height=330)

    st.markdown('<h3 class="section-title">Metrik trendi</h3>', unsafe_allow_html=True)
    st.line_chart(chart, height=280)

    class_columns = ["AP_tasit", "AP_insan", "AP_UAP", "AP_UAI"]
    class_table = results.set_index("scenario")[class_columns].astype(float)
    class_table.columns = ["tasit", "insan", "UAP", "UAI"]
    st.markdown('<h3 class="section-title">Sinif performans haritasi</h3>', unsafe_allow_html=True)
    # Avoid pandas Styler in Streamlit: older pandas/Streamlit combinations
    # reject Styler maps when an index is considered non-unique.
    st.dataframe(class_table.round(3), use_container_width=True)

    st.markdown('<h3 class="section-title">Deney mantigi</h3>', unsafe_allow_html=True)
    flow = st.columns(3)
    flow[0].markdown("**01 · Saglikli referans**\n\nBaseline model sabit diagnostic sette olculur.")
    flow[1].markdown("**02 · Kontrollu bozulma**\n\nTek bir veri problemi kasitli olarak uygulanir.")
    flow[2].markdown("**03 · Kanita dayali yorum**\n\nMetrik farklari ve gorsel kanit birlikte incelenir.")


elif page == "Senaryo Incele":
    selected = st.sidebar.selectbox("Kosu", scenarios, index=0)
    row = results[results["scenario"] == selected].iloc[0]
    st.markdown(f'<h2 class="section-title">{selected}</h2><p class="section-subtitle">val_diagnostic · 1.056 goruntu · 4 sinif · test seti kullanilmadi</p>', unsafe_allow_html=True)

    metric_columns = (("mAP50", "mAP50"), ("mAP50-95", "mAP50_95"), ("Precision", "precision"), ("Recall", "recall"))
    cards = st.columns(4)
    for card, (label, field) in zip(cards, metric_columns):
        value = float(row[field])
        delta = value - float(baseline[field]) if selected != "Baseline" else None
        card.metric(label, f"{value:.4f}", None if delta is None else f"{delta:+.4f} baseline")

    info = scenario_info[selected]
    st.markdown(f'<div class="callout"><b>{info["problem"]}</b><br>{info["why"]}</div>', unsafe_allow_html=True)
    detail = st.columns(3)
    detail[0].markdown(f"**Uygulanan degisiklik**\n\n{info['change']}")
    detail[1].markdown(f"**Beklenen sinyal**\n\n{info['signal']}")
    detail[2].markdown(f"**Durum**\n\n`{info['status']}`")

    left, right = st.columns([1, 1.2])
    with left:
        st.markdown('<h3 class="section-title">Sinif bazli AP50</h3>', unsafe_allow_html=True)
        class_frame = results[results["scenario"] == selected][["AP_tasit", "AP_insan", "AP_UAP", "AP_UAI"]].T
        class_frame.columns = ["AP50"]
        class_frame.index = ["tasit", "insan", "UAP", "UAI"]
        st.bar_chart(class_frame, height=300)
    with right:
        st.markdown('<h3 class="section-title">Kayitli kanit</h3>', unsafe_allow_html=True)
        evidence = evidence_for(selected)
        st.json(evidence if evidence else {"durum": "rapor bulunamadi"})

    if selected != "Baseline":
        st.markdown('<h3 class="section-title">Baseline farki</h3>', unsafe_allow_html=True)
        comparison = results[results["scenario"].isin(["Baseline", selected])].set_index("scenario")[["mAP50", "mAP50_95", "precision", "recall"]].astype(float)
        st.bar_chart(comparison, height=280)
        st.caption("Pozitif fark, secili kosunun baseline'a gore daha yuksek oldugunu gosterir.")

    st.markdown('<h3 class="section-title">Precision - recall dengesi</h3>', unsafe_allow_html=True)
    tradeoff = results[["scenario", "precision", "recall", "mAP50"]].copy()
    tradeoff["bubble"] = tradeoff["mAP50"] * 100
    st.scatter_chart(tradeoff, x="recall", y="precision", size="bubble", color="scenario", height=320)

    images = images_for(selected)
    if images:
        st.markdown('<h3 class="section-title">Gorsel kanit</h3>', unsafe_allow_html=True)
        image_cols = st.columns(len(images))
        for column, image in zip(image_cols, images):
            column.image(str(image), caption=image.name, use_container_width=True)

    examples = examples_for(selected)
    if examples:
        with st.expander("Ornek label ve tahmin gorselleri", expanded=False):
            example_cols = st.columns(len(examples))
            for column, image in zip(example_cols, examples):
                column.image(str(image), caption=image.name, use_container_width=True)


elif page == "LLM Ajan":
    st.markdown('<h2 class="section-title">LLM Ajan Denemesi</h2><p class="section-subtitle">Gemini 3.6 Flash, anonim deney metriklerini yorumluyor.</p>', unsafe_allow_html=True)
    st.markdown('<div class="callout"><b>Bu bir pilot ajan testidir.</b> LLM, senaryo isimlerini gormeden metriklerden teshis, kanit, guven ve sonraki olcumu uretir.</div>', unsafe_allow_html=True)
    response = llm_response()
    score = llm_score()
    if score:
        score_cols = st.columns(3)
        score_cols[0].metric("Pilot ortalama skoru", f"{score.get('mean_score', 0):.3f}")
        score_cols[1].metric("Degerlendirilen kosu", str(len(score.get("runs", []))))
        score_cols[2].metric("Rubrik", "3 alan")
        st.caption("Skor; teshis, sayisal kanit ve sinirlama alanlarinin esit agirlikli pilot rubrigidir.")
    if response:
        for item in response if isinstance(response, list) else [response]:
            run_id = item.get("run_id", "kosu")
            with st.expander(run_id, expanded=run_id == "kosu_01"):
                st.markdown(f"**Teshis:** `{item.get('diagnosis', '-')}`")
                st.markdown(f"**Guven:** `{item.get('confidence', '-')}`")
                st.markdown("**Kanıtlar**")
                for evidence in item.get("evidence", []):
                    st.write(f"- {evidence}")
                st.markdown("**Sinirlamalar**")
                for limitation in item.get("limitations", []):
                    st.write(f"- {limitation}")
                st.markdown(f"**Sonraki olcum:** {item.get('next_measurement', '-')}")
    else:
        st.info("Gemini cikti dosyasi bulunamadi.")

else:
    st.markdown('<h2 class="section-title">Proje ve Senaryolar</h2><p class="section-subtitle">Kontrollu deney tasariminin amaci ve senaryolarin nasil olusturuldugu.</p>', unsafe_allow_html=True)
    st.markdown('<div class="callout"><b>Temel soru:</b> Modelin basarisi dustugunde sorun modelin kendisi mi, veri seti mi, etiketler mi, yoksa belirli bir sinifin yetersiz temsil edilmesi mi?</div>', unsafe_allow_html=True)

    st.markdown('<h3 class="section-title">Senaryo nasil olusturuluyor?</h3>', unsafe_allow_html=True)
    steps = st.columns(4)
    steps[0].markdown("**01 · Baseline**\n\nSaglikli model ve kilitli diagnostic set olusturulur.")
    steps[1].markdown("**02 · Tek degisken**\n\nSadece bir veri problemi kontrollu olarak uygulanir.")
    steps[2].markdown("**03 · Yeniden egitim**\n\nAyni egitim protokoluyle yeni model kosulur.")
    steps[3].markdown("**04 · Karsilastirma**\n\nMetrik ve gorsel kanit baseline ile karsilastirilir.")

    st.markdown('<h3 class="section-title">Tamamlanan senaryolar</h3>', unsafe_allow_html=True)
    for name in scenarios:
        info = scenario_info[name]
        with st.expander(f"{name} · {info['problem']}", expanded=name == "Baseline"):
            c1, c2 = st.columns(2)
            c1.markdown(f"**Nasil olusturuldu?**\n\n{info['change']}")
            c2.markdown(f"**Neden olusturuldu?**\n\n{info['why']}")
            st.info(f"Beklenen veya gozlenen sinyal: {info['signal']}")

    st.markdown('<h3 class="section-title">Neden test seti kullanilmiyor?</h3>', unsafe_allow_html=True)
    st.write("Bu asamada test seti son karari vermek icin saklanir. Senaryolar, model secimini veya final performansi test setine bakarak etkilemesin diye yalnizca kilitli val_diagnostic setinde incelenir.")
    st.markdown('<h3 class="section-title">Ara sunumda anlatilacak hikaye</h3>', unsafe_allow_html=True)
    st.write("Once saglikli referans gosterilir. Sonra tek bir veri problemi kasitli olarak uygulanir. Metriklerdeki degisim ve confusion matrix birlikte incelenir. Son olarak LLM, bu kanitlari kullanarak veri probleminin olasi nedenini ve bir sonraki olcumu onerir.")

st.divider()
st.caption("Termal Teshis Ajani · Ara sunum demosu · Test seti final asamasina kadar kullanilmaz.")
