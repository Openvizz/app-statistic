import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px
import re

from deep_translator import GoogleTranslator
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from PIL import Image

if "translated_columns" not in st.session_state:
    st.session_state.translated_columns = None

logo = Image.open("logo.png")

st.sidebar.image(logo, use_container_width=True)
st.image("logo.png", width=250)

# =========================================
# CONFIG GERAL
# =========================================
st.set_page_config(
    page_title="Desafio de Estatística Aplicada",
    layout="wide"
)

# CSS – estilo + redução de espaços em branco
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.0rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #9ca3af;
        font-size: 0.95rem;
        margin-bottom: 1.2rem;
    }
    .card {
        background-color: #ffffff;
        color: #111827;
        padding: 0.8rem 1.0rem;
        border-radius: 0.6rem;
        box-shadow: 0 0 10px rgba(0,0,0,0.05);
        border: 1px solid #f0f2f6;
        margin-bottom: 0.7rem;
    }
    .sidebar-title {
        font-weight: 600;
        font-size: 0.9rem;
        margin-top: 0.8rem;
        margin-bottom: 0.2rem;
    }
    .sidebar-note {
        font-size: 0.8rem;
        color: #9ca3af;
    }
    /* reduce global paddings */
    .block-container {
        padding-top: 0.8rem !important;
        padding-bottom: 0.4rem !important;
    }
    .element-container {
        margin-bottom: 0.35rem !important;
    }
    .row-widget.stHorizontalBlock {
        gap: 0.7rem !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        margin-bottom: 0rem !important;
    }
    h1, h2, h3, h4 {
        margin-top: 0rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">📈 Desafio de Estatística Aplicada</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Aplicação interativa para análise estatística com suporte a CSV e Excel.</div>',
    unsafe_allow_html=True,
)

# =========================================
# NAVEGAÇÃO ENTRE "PÁGINAS"
# =========================================
st.sidebar.markdown('<div class="sidebar-title">🧭 Navegação</div>', unsafe_allow_html=True)
page = st.sidebar.radio(
    "Selecione a página",
    options=["🏠 Visão geral", "📊 Análise estatística"],
    index=0
)

# =========================================
# PÁGINA 1 – VISÃO GERAL
# =========================================
if page == "🏠 Visão geral":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🎯 Objetivo do aplicativo")
    st.write(
        """
        Este aplicativo foi desenvolvido para apoiar análises de **Estatística Aplicada** a partir de bases de dados
        fornecidas em CSV ou Excel. A ideia é fornecer uma visão rápida e amigável dos dados, com foco em:
        - Entendimento da estrutura do dataset;
        - Estatística descritiva das variáveis numéricas;
        - Relações entre variáveis via matriz de correlação;
        - Bloco configurável para aplicar as regras específicas do desafio (testes, indicadores, etc.);
        - Geração de um relatório em PDF com o resumo das principais informações.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🧱 Como o app funciona (workflow)")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Entrada de dados")
        st.write(
            """
            1. Acesse a página **📊 Análise estatística** pelo menu lateral.
            2. Escolha o tipo de arquivo: **CSV** ou **Excel**.
            3. Faça o upload do arquivo.
            4. Opcionalmente, marque a opção de **traduzir cabeçalhos EN ➜ PT**.
            """
        )
    with col2:
        st.markdown("#### Saídas geradas")
        st.write(
            """
            - Visão geral do dataset (quantidade de linhas, colunas e campos numéricos);
            - Estatísticas descritivas (média, desvio-padrão, quartis, coeficiente de variação);
            - Gráficos interativos de distribuição (histograma e boxplot);
            - Matriz de correlação com heatmap;
            - Área para aplicar regras específicas do desafio;
            - Exportação de:
              - **PDF** com resumo estatístico;
              - **CSV processado** (incluindo colunas derivadas criadas no app).
            """
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🧪 Uso típico em um estudo de Estatística Aplicada")
    st.write(
        """
        - Carregar a base de dados de interesse (por exemplo, notas de alunos, resultados de experimentos, indicadores de processo);
        - Traduzir cabeçalhos, se necessário, para facilitar leitura pelos stakeholders;
        - Explorar a distribuição das variáveis-chave;
        - Analisar correlações para identificar relações relevantes;
        - Implementar, na aba **Regras do Desafio**, os cálculos solicitados no enunciado (p.ex. testes de hipóteses, comparações entre grupos);
        - Gerar o **relatório em PDF** como evidência das análises realizadas.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.stop()  # não executa o restante do app nesta página

# =========================================
# PÁGINA 2 – ANÁLISE ESTATÍSTICA
# =========================================

# ---- Sidebar específico desta página ----
st.sidebar.markdown('<div class="sidebar-title">📂 Fonte de dados</div>', unsafe_allow_html=True)

file_type = st.sidebar.radio(
    "Tipo de arquivo",
    options=["Excel", "CSV"],
    index=0,
    horizontal=True
)

uploaded_file = st.sidebar.file_uploader(
    "Envie o arquivo de dados",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=False
)

if file_type == "CSV":
    st.sidebar.markdown('<div class="sidebar-title">⚙️ Parâmetros CSV</div>', unsafe_allow_html=True)
    sep = st.sidebar.text_input("Separador", value=",")
    decimal = st.sidebar.text_input("Separador decimal", value=".")
else:
    sep = None
    decimal = None

st.sidebar.markdown('<div class="sidebar-title">🌐 Tradução de cabeçalho</div>', unsafe_allow_html=True)
translate_headers_opt = st.sidebar.checkbox(
    "Traduzir cabeçalhos EN ➜ PT",
    value=False,
    help="Traduz nomes de colunas via serviço automático; pode impactar performance em muitos campos."
)

st.sidebar.markdown('<div class="sidebar-title">ℹ️ Observação</div>', unsafe_allow_html=True)
st.sidebar.markdown(
    '<div class="sidebar-note">Após o upload, use as abas abaixo para navegar entre dados, estatísticas, correlação, regras do desafio e relatório.</div>',
    unsafe_allow_html=True,
)

# =========================================
# FUNÇÕES AUXILIARES
# =========================================

# Tradutor global
translator = GoogleTranslator(source="en", target="pt")


def load_data(file, sep, decimal) -> pd.DataFrame:
    """
    Carrega CSV ou Excel, priorizando o sufixo do arquivo.
    Para CSV, usa separador/decimal informados ou defaults razoáveis.
    """
    name = file.name.lower()
    if name.endswith(".csv"):
        sep = sep if sep not in [None, ""] else ","
        decimal = decimal if decimal not in [None, ""] else "."
        return pd.read_csv(file, sep=sep, decimal=decimal)
    else:
        return pd.read_excel(file, engine="openpyxl")


def _normalize_header(text: str) -> str:
    """
    Normaliza o texto traduzido para nome de coluna:
      - remove espaços extras
      - troca espaços por "_"
      - remove caracteres problemáticos
    """
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\sáéíóúâêôãõçÁÉÍÓÚÂÊÔÃÕÇ]", "", text)
    text = text.replace(" ", "_")
    return text


def _prepare_source_header(col: str) -> str:
    """
    Prepara o cabeçalho original para mandar ao tradutor:
      - converte "_" e "-" em espaço
      - quebra CamelCase: HoursStudied -> Hours Studied
    """
    s = re.sub(r"[_\-]+", " ", col)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)
    return s.strip()


def translate_headers(df: pd.DataFrame, enabled: bool) -> pd.DataFrame:
    """
    Traduz automaticamente cabeçalhos (majoritariamente em inglês)
    para português usando o GoogleTranslator (deep-translator).

    Se a tradução falhar em alguma coluna, mantém o nome original.
    """
    if not enabled:
        return df

    new_cols = {}
    for col in df.columns:
        try:
            src = _prepare_source_header(col)
            if not src:
                new_cols[col] = col
                continue

            translated = translator.translate(src)
            if not translated or translated.strip().lower() == src.strip().lower():
                new_cols[col] = col
            else:
                new_cols[col] = _normalize_header(translated)
        except Exception:
            new_cols[col] = col

    return df.rename(columns=new_cols)


def numeric_columns(df: pd.DataFrame):
    return df.select_dtypes(include=[np.number]).columns.tolist()


def categorical_columns(df: pd.DataFrame):
    return df.select_dtypes(exclude=[np.number]).columns.tolist()


def generate_pdf_report(df, num_cols, cat_cols, target_col, feature_cols) -> bytes:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 2 * cm

    c.setFont("Helvetica-Bold", 16)
    c.drawString(2 * cm, y, "Relatório de Estatística Aplicada")
    y -= 1.2 * cm

    c.setFont("Helvetica", 10)
    c.drawString(2 * cm, y, f"Linhas: {df.shape[0]}  |  Colunas: {df.shape[1]}")
    y -= 0.7 * cm

    if target_col and target_col != "(nenhuma)":
        c.drawString(2 * cm, y, f"Variável resposta: {target_col}")
        y -= 0.6 * cm
    if feature_cols:
        c.drawString(2 * cm, y, "Variáveis explicativas: " + ", ".join(feature_cols))
        y -= 0.8 * cm

    # Estatísticas descritivas
    if num_cols:
        c.setFont("Helvetica-Bold", 11)
        c.drawString(2 * cm, y, "Resumo das variáveis numéricas:")
        y -= 0.7 * cm
        c.setFont("Helvetica", 9)

        desc = df[num_cols].describe().T.round(2)
        for col in desc.index:
            linha = desc.loc[col]
            text = (
                f"{col}  |  média={linha['mean']}  "
                f"desv={linha['std']}  min={linha['min']}  max={linha['max']}"
            )
            c.drawString(2 * cm, y, text[:110])
            y -= 0.5 * cm
            if y < 2 * cm:
                c.showPage()
                y = height - 2 * cm
                c.setFont("Helvetica", 9)

    # Top correlações
    if len(num_cols) > 1:
        corr = df[num_cols].corr()
        corr_pairs = (
            corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            .stack()
            .reset_index()
        )
        corr_pairs.columns = ["Var1", "Var2", "Correlacao"]
        corr_pairs["abs_corr"] = corr_pairs["Correlacao"].abs()
        top_corr = corr_pairs.sort_values("abs_corr", ascending=False).head(10)

        if y < 4 * cm:
            c.showPage()
            y = height - 2 * cm

        c.setFont("Helvetica-Bold", 11)
        c.drawString(2 * cm, y, "Top 10 correlações (em módulo):")
        y -= 0.7 * cm
        c.setFont("Helvetica", 9)

        for _, row in top_corr.iterrows():
            text = f"{row['Var1']} x {row['Var2']}  ->  r = {row['Correlacao']:.2f}"
            c.drawString(2 * cm, y, text)
            y -= 0.5 * cm
            if y < 2 * cm:
                c.showPage()
                y = height - 2 * cm
                c.setFont("Helvetica", 9)

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()

# =========================================
# CORPO PRINCIPAL DA PÁGINA 2
# =========================================
if uploaded_file is None:
    st.info("Envie um arquivo CSV ou Excel na barra lateral para iniciar a análise.")
    st.stop()

# Carrega dados
try:
    df = load_data(uploaded_file, sep=sep, decimal=decimal)

    # Traduz apenas na primeira vez que o arquivo é carregado
    if translate_headers_opt:
        if st.session_state.translated_columns is None or \
        st.session_state.translated_columns["source_name"] != uploaded_file.name:
            
            translated_df = translate_headers(df.copy(), True)
            st.session_state.translated_columns = {
                "source_name": uploaded_file.name,
                "columns": translated_df.columns.tolist()
            }
            df.columns = translated_df.columns

        else:
            # Reaplica nomes já traduzidos sem chamar API novamente
            df.columns = st.session_state.translated_columns["columns"]

    else:
        # Se a tradução estiver desativada pelo usuário
        st.session_state.translated_columns = None

except Exception as e:
    st.error(f"Erro ao carregar o arquivo: {e}")
    st.stop()

num_cols = numeric_columns(df)
cat_cols = categorical_columns(df)

# Visão geral do dataset
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### 🔎 Visão geral do dataset")
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.metric("📄 Linhas", f"{df.shape[0]:,}".replace(",", "."))
with col_b:
    st.metric("📊 Colunas", f"{df.shape[1]:,}".replace(",", "."))
with col_c:
    st.metric("🔢 Campos numéricos", f"{len(num_cols):,}".replace(",", "."))
st.markdown("</div>", unsafe_allow_html=True)

# Configuração de variáveis
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### 🧮 Configuração das variáveis")

col1, col2 = st.columns(2)
with col1:
    target_col = st.selectbox(
        "Variável resposta (opcional)",
        options=["(nenhuma)"] + num_cols + cat_cols,
        index=0,
    )
with col2:
    feature_cols = st.multiselect(
        "Variáveis explicativas / preditoras (opcional)",
        options=[c for c in df.columns if c != target_col],
        default=[]
    )
st.markdown("</div>", unsafe_allow_html=True)

# Abas de análise
tab_dados, tab_desc, tab_corr, tab_regras, tab_export = st.tabs(
    ["📊 Dados", "📐 Estatística Descritiva", "📈 Correlação", "🧩 Regras do Desafio", "📑 Relatório / Exportação"]
)

# ----- Aba Dados -----
with tab_dados:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 📊 Visualização dos dados")
    st.dataframe(df.head(200), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ----- Aba Estatística Descritiva -----
with tab_desc:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 📐 Estatísticas descritivas")

    if len(num_cols) == 0:
        st.warning("Não foram identificadas colunas numéricas para análise descritiva.")
    else:
        desc = df[num_cols].describe().T
        desc["coef_var"] = desc["std"] / desc["mean"]
        st.dataframe(desc, use_container_width=True)

        st.markdown("##### 📊 Distribuição de variável numérica")
        selected_num_for_plot = st.selectbox(
            "Escolha a variável para visualizar a distribuição",
            options=num_cols,
            key="desc_num_var"
        )

        col_hist, col_box = st.columns(2)

        with col_hist:
            st.markdown("Histograma")
            fig_hist = px.histogram(
                df,
                x=selected_num_for_plot,
                nbins=30,
                marginal="rug",
                template="simple_white"
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_box:
            st.markdown("Boxplot")
            fig_box = px.box(
                df,
                y=selected_num_for_plot,
                points="all",
                template="simple_white"
            )
            st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ----- Aba Correlação -----
with tab_corr:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 📈 Matriz de correlação")

    if len(num_cols) <= 1:
        st.info("São necessárias pelo menos duas variáveis numéricas para calcular correlação.")
    else:
        corr = df[num_cols].corr()
        st.dataframe(corr, use_container_width=True)

        st.markdown("##### 🔍 Heatmap de correlação")
        fig_corr = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            title="Matriz de correlação",
            template="simple_white",
            color_continuous_scale="RdBu_r",
        )
        fig_corr.update_xaxes(side="top")
        st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ----- Aba Regras do Desafio -----
with tab_regras:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 🧩 Regras específicas do desafio")
    st.markdown(
        "Substituir estes exemplos pelos cálculos e regras definidos no enunciado do desafio, "
        "utilizando o DataFrame `df`, a variável resposta e as variáveis explicativas."
    )

    # Exemplo 1 – média por grupo
    with st.expander("Exemplo 1: média da variável resposta por grupo (placeholder)", expanded=True):
        if target_col != "(nenhuma)" and len(cat_cols) > 0 and target_col in num_cols:
            group_col = st.selectbox(
                "Escolha uma variável categórica para agrupar",
                options=cat_cols,
                key="regras_group_col"
            )
            grouped = df.groupby(group_col)[target_col].agg(["count", "mean", "std"]).reset_index()
            st.dataframe(grouped, use_container_width=True)

            fig_group = px.bar(
                grouped,
                x=group_col,
                y="mean",
                error_y="std",
                title=f"Média de {target_col} por {group_col}",
                template="simple_white"
            )
            st.plotly_chart(fig_group, use_container_width=True)
        else:
            st.info("Para este exemplo, selecione uma variável resposta numérica e pelo menos uma variável categórica.")

    # Exemplo 2 – z-score
    with st.expander("Exemplo 2: cálculo de z-score (placeholder)", expanded=False):
        if len(num_cols) > 0:
            num_for_z = st.selectbox(
                "Escolha a variável para calcular z-score",
                options=num_cols,
                key="regras_num_for_z"
            )
            col_name_z = f"{num_for_z}_zscore"
            df[col_name_z] = (df[num_for_z] - df[num_for_z].mean()) / df[num_for_z].std(ddof=0)
            st.dataframe(df[[num_for_z, col_name_z]].head(50), use_container_width=True)

            fig_z = px.scatter(
                df,
                x=num_for_z,
                y=col_name_z,
                title=f"Z-score de {num_for_z}",
                template="simple_white"
            )
            st.plotly_chart(fig_z, use_container_width=True)
        else:
            st.info("Não há variáveis numéricas para o exemplo de z-score.")

    st.markdown("</div>", unsafe_allow_html=True)

# ----- Aba Relatório / Exportação -----
with tab_export:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 📑 Relatório e exportação")

    pdf_bytes = generate_pdf_report(df, num_cols, cat_cols, target_col, feature_cols)
    st.download_button(
        label="📑 Baixar relatório em PDF",
        data=pdf_bytes,
        file_name="relatorio_estatistica_aplicada.pdf",
        mime="application/pdf"
    )

    st.markdown("---")

    buffer_csv = io.StringIO()
    df.to_csv(buffer_csv, index=False, sep=";", decimal=",")
    buffer_csv.seek(0)
    st.download_button(
        label="📤 Baixar CSV processado",
        data=buffer_csv.getvalue(),
        file_name="dados_processados.csv",
        mime="text/csv"
    )

    st.markdown("</div>", unsafe_allow_html=True)
