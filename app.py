# app.py
import io
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
from fpdf import FPDF
import streamlit as st


# ---------------------- CONFIGURAÇÃO DA PÁGINA ----------------------
st.set_page_config(
    page_title="Simulador de Patologia Digital",
    page_icon="🧫",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------- ESTILOS GERAIS ----------------------
CUSTOM_CSS = """
<style>
    /* Deixa fundo mais clean e cartões com visual de dashboard */
    .main {
        background-color: #0f172a;
        color: #e5e7eb;
    }
    section[data-testid="stSidebar"] {
        background-color: #020617;
    }
    .metric-card {
        padding: 1rem 1.25rem;
        border-radius: 0.75rem;
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.75);
    }
    .metric-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #9ca3af;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 600;
        color: #e5e7eb;
    }
    .metric-sub {
        font-size: 0.75rem;
        color: #9ca3af;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        border-bottom: 1px solid #1f2937;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #020617;
        padding: 0.5rem 1rem;
        border-radius: 999px;
        color: #9ca3af;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1, #22c55e);
        color: #f9fafb !important;
    }
    .annotation-box {
        border-radius: 0.75rem;
        border: 1px solid #334155;
        padding: 0.75rem;
        background: rgba(15, 23, 42, 0.9);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------- FUNÇÕES AUXILIARES ----------------------
@st.cache_data
def read_image(file) -> np.ndarray:
    """Lê uma imagem enviada pelo usuário e retorna em formato OpenCV (BGR)."""
    bytes_data = file.read()
    image = Image.open(io.BytesIO(bytes_data)).convert("RGB")
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    return img_bgr


def apply_zoom(image: np.ndarray, zoom: float) -> np.ndarray:
    """Aplica zoom simples (crop central) simulando aproximação do campo."""
    if zoom == 1.0:
        return image
    h, w, _ = image.shape
    center_x, center_y = w // 2, h // 2
    new_w, new_h = int(w / zoom), int(h / zoom)
    x1 = max(center_x - new_w // 2, 0)
    y1 = max(center_y - new_h // 2, 0)
    x2 = min(center_x + new_w // 2, w)
    y2 = min(center_y + new_h // 2, h)
    cropped = image[y1:y2, x1:x2]
    resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_CUBIC)
    return resized


def draw_grid(image: np.ndarray, grid_size: int = 5, color=(0, 255, 0)) -> np.ndarray:
    """Desenha uma grade sobre a imagem para treino de navegação/contagem."""
    img = image.copy()
    h, w, _ = img.shape
    step_x = w // grid_size
    step_y = h // grid_size

    for i in range(1, grid_size):
        # linhas verticais
        cv2.line(img, (i * step_x, 0), (i * step_x, h), color, 1)
        # linhas horizontais
        cv2.line(img, (0, i * step_y), (w, i * step_y), color, 1)

    return img


def to_pil(image_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))


def generate_pdf_report(
    pil_image: Image.Image,
    student_name: str,
    case_id: str,
    comments: str,
) -> bytes:
    """Gera um PDF simples com a lâmina e o relatório do aluno."""
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Cabeçalho
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Simulador de Patologia Digital", ln=True, align="C")

    pdf.set_font("Arial", "", 11)
    pdf.ln(4)
    pdf.cell(0, 8, f"Aluno: {student_name}", ln=True)
    pdf.cell(0, 8, f"Caso: {case_id}", ln=True)
    pdf.cell(0, 8, f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}", ln=True)

    # Salvar imagem temporariamente em memória
    img_buffer = io.BytesIO()
    pil_image.save(img_buffer, format="PNG")
    img_buffer.seek(0)

    # Salvar em arquivo temporário para o FPDF (FPDF não aceita BytesIO diretamente)
    temp_path = "temp_slide.png"
    with open(temp_path, "wb") as f:
        f.write(img_buffer.read())

    # Inserir imagem centralizada
    pdf.ln(4)
    x = 10
    max_width = 190  # A4 width - 2*10mm
    pdf.image(temp_path, x=x, w=max_width)

    # Comentários do aluno
    pdf.ln(8)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Raciocínio diagnóstico / observações:", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 6, comments or "(sem comentários)")

    # Exporta para bytes
    pdf_bytes = pdf.output(dest="S").encode("latin1")
    return pdf_bytes


# ---------------------- LAYOUT PRINCIPAL ----------------------
st.title("🧫 Simulador de Análise Patológica")
st.markdown(
    "Simulador interativo de **patologia** digital para treinamento de alunos em leitura de lâminas e letramento digital."
)

with st.sidebar:
    st.header("Configurações")
    student_name = st.text_input("Nome do aluno", placeholder="Digite seu nome")
    case_id = st.text_input("Identificação do caso", placeholder="Ex.: Caso 01 - Necrose")

    st.markdown("---")
    zoom = st.slider("Zoom aproximado", min_value=1.0, max_value=4.0, value=1.5, step=0.25)
    show_grid = st.checkbox("Mostrar grade de contagem", value=False)
    grid_size = st.slider("Resolução da grade", 3, 10, 5)

    st.markdown("---")
    st.caption("Carregue uma lâmina digital (JPG, PNG ou TIFF).")
    uploaded_file = st.file_uploader(
        "Lâmina digital", type=["jpg", "jpeg", "png", "tiff"], accept_multiple_files=False
    )

# Métricas / cards superiores
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">Modo</div>
            <div class="metric-value">Treino individual</div>
            <div class="metric-sub">Exploração livre da lâmina</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_b:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Caso ativo</div>
            <div class="metric-value">{case_id or "Não definido"}</div>
            <div class="metric-sub">Defina um caso na barra lateral</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_c:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">Funcionalidades</div>
            <div class="metric-value">Zoom + Grade</div>
            <div class="metric-sub">PDF com registro do raciocínio</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("")

tab1, tab2 = st.tabs(["Visualização da lâmina", "Relatório do aluno"])

# ---------------------- TAB 1: VISUALIZAÇÃO ----------------------
with tab1:
    if uploaded_file is None:
        st.info("Carregue uma imagem de lâmina na barra lateral para iniciar o simulador.")
    else:
        # lê e processa
        img_bgr = read_image(uploaded_file)
        img_zoom = apply_zoom(img_bgr, zoom=zoom)
        if show_grid:
            img_zoom = draw_grid(img_zoom, grid_size=grid_size)

        pil_to_show = to_pil(img_zoom)

        # layout de duas colunas: imagem e painel de instruções
        img_col, info_col = st.columns([3, 2])

        with img_col:
            st.subheader("Campo de visão")
            st.image(pil_to_show, use_column_width=True)

        with info_col:
            st.subheader("Tarefas sugeridas")
            st.markdown(
                """
                - Identifique regiões de interesse (inflamação, necrose, células atípicas).  
                - Use o **zoom** para simular diferentes aumentos do microscópio.  
                - Ative a **grade** para exercícios de contagem celular ou estimativa de proporções.  
                """
            )
            st.markdown("### Observações rápidas")
            quick_notes = st.text_area(
                "Anote o que você está vendo (pontos‑chave morfológicos).",
                height=160,
                key="quick_notes",
            )

# ---------------------- TAB 2: RELATÓRIO + PDF ----------------------
with tab2:
    st.subheader("Raciocínio diagnóstico")
    comments = st.text_area(
        "Descreva seu raciocínio (padrões, diagnóstico diferencial, correlação clínico‑patológica).",
        height=260,
    )

    col_left, col_right = st.columns([1, 1])
    with col_left:
        include_image = st.checkbox("Incluir captura da lâmina no PDF", value=True)
    with col_right:
        st.caption("O PDF pode ser usado para portfólio de aprendizagem ou avaliação formativa.")

    if st.button("📄 Gerar PDF do caso", type="primary"):
        if uploaded_file is None:
            st.warning("Você precisa carregar uma lâmina antes de gerar o PDF.")
        else:
            img_bgr = read_image(uploaded_file)
            img_zoom = apply_zoom(img_bgr, zoom=zoom)
            if show_grid:
                img_zoom = draw_grid(img_zoom, grid_size=grid_size)
            pil_img = to_pil(img_zoom) if include_image else Image.new("RGB", (800, 600), "white")

            pdf_bytes = generate_pdf_report(
                pil_image=pil_img,
                student_name=student_name or "Aluno não identificado",
                case_id=case_id or "Caso sem identificação",
                comments=comments or "",
            )

            st.success("PDF gerado com sucesso. Faça o download abaixo.")
            st.download_button(
                label="⬇️ Baixar relatório em PDF",
                data=pdf_bytes,
                file_name=f"relatorio_patologia_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
            )
