# app.py
import io
import random
import time
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
from fpdf import FPDF
import streamlit as st

# ---------------------- DADOS E CENÁRIOS PEDAGÓGICOS ----------------------
# Aqui definimos os textos técnicos detalhados para cada cenário solicitado.
PATHOLOGY_SCENARIOS = {
    "Modo Livre (Aleatório)": None,
    "Músculo Cardíaco": {
        "diagnosis": "Hipertrofia Miocárdica",
        "description": (
            "A análise histopatológica revela aumento significativo do diâmetro transversal dos cardiomiócitos "
            "(hipertrofia), sem evidência de aumento numérico das células (hiperplasia). Observam-se núcleos "
            "aumentados de volume, por vezes hipercromáticos e com formas retangulares ('em vagão de trem' "
            "ou boxcar nuclei). O citoplasma apresenta-se abundante e eosinofílico. Há preservação da arquitetura "
            "fascicular, podendo haver leve fibrose intersticial secundária dependendo da cronicidade."
        )
    },
    "Epitélio Prostático": {
        "diagnosis": "Hiperplasia Prostática Benigna (HPB)",
        "description": (
            "Cortes evidenciam proliferação nodular de elementos estromais e glandulares. As glândulas apresentam "
            "arquitetura variada, algumas císticas e dilatadas, revestidas por dupla camada celular: uma camada "
            "basal de células cuboides/achatadas e uma camada luminal de células cilíndricas secretoras. "
            "Observam-se projeções papilares intra-luminais. O estroma fibromuscular exibe hipercelularidade. "
            "Ausência de atipias nucleares significativas ou invasão estromal que sugerissem malignidade."
        )
    },
    "Epitélio Brônquico": {
        "diagnosis": "Metaplasia Escamosa",
        "description": (
            "Observa-se substituição do epitélio respiratório normal (pseudoestratificado cilíndrico ciliado) "
            "por epitélio escamoso estratificado maduro. Nota-se perda dos cílios e das células caliciformes "
            "na área afetada. As células escamosas apresentam diferenciação regular, sem atipias ou disqueratose "
            "significativas neste corte. O processo representa uma resposta adaptativa reversível a irritantes crônicos "
            "(ex.: tabagismo), conferindo maior resistência mecânica em detrimento da função mucociliar."
        )
    },
    "Tecido Nervoso": {
        "diagnosis": "Atrofia Encefálica / Alterações Neurodegenerativas",
        "description": (
            "O tecido exibe redução da densidade neuronal, com neurônios remanescentes apresentando retração "
            "do corpo celular (encolhimento) e picnose nuclear. Observa-se gliose reacional (proliferação de "
            "astrócitos e micróglias) substituindo o neurópilo perdido. Pode haver dilatação dos espaços "
            "perivasculares (Virchow-Robin). Tais achados correlacionam-se macroscopicamente com o estreitamento "
            "dos giros e alargamento dos sulcos, compatíveis com processo atrófico cerebral."
        )
    },
    "Epitélio Renal": {
        "diagnosis": "Degeneração Hidrópica (Tumefação Turva)",
        "description": (
            "Os túbulos renais proximais exibem células aumentadas de volume devido ao acúmulo intracelular de "
            "água. O citoplasma apresenta-se pálido, vacuolizado e finamente granular, deslocando o núcleo "
            "centralmente, embora este permaneça viável. A luz tubular encontra-se reduzida ou ocluída pela "
            "protrusão celular. Trata-se de uma lesão celular reversível decorrente de falha nas bombas iônicas "
            "de membrana (hipóxia ou tóxicos)."
        )
    },
    "Tecido Hepático": {
        "diagnosis": "Esteatose Hepática (Degeneração Gordurosa)",
        "description": (
            "Parênquima hepático exibindo acúmulo de vacúolos lipídicos no citoplasma dos hepatócitos. "
            "Predomínio de esteatose macrovesicular, onde uma única grande gota de gordura desloca e comprime "
            "o núcleo para a periferia da célula (aspecto em 'anel de sinete'). Em menores áreas, pode haver "
            "esteatose microvesicular (múltiplos pequenos vacúolos sem deslocamento nuclear). Inflamação "
            "lobular é escassa ou ausente neste campo."
        )
    }
}

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
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------- FUNÇÕES AUXILIARES BÁSICAS ----------------------
@st.cache_data
def read_image(file) -> np.ndarray:
    """Lê uma imagem enviada pelo usuário e retorna em formato OpenCV (BGR)."""
    bytes_data = file.read()
    image = Image.open(io.BytesIO(bytes_data)).convert("RGB")
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    return img_bgr


def apply_zoom(image: np.ndarray, zoom: float) -> np.ndarray:
    """Aplica zoom simples (crop central)."""
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
        cv2.line(img, (i * step_x, 0), (i * step_x, h), color, 1)
        cv2.line(img, (0, i * step_y), (w, i * step_y), color, 1)

    return img


def to_pil(image_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))


# ---------------------- FUNÇÕES PEDAGÓGICAS ----------------------
def deidentify_slide(image_bgr: np.ndarray, border_pct: float = 0.08) -> np.ndarray:
    """Desidentificação simples: blur nas bordas onde labels costumam aparecer."""
    img = image_bgr.copy()
    h, w, _ = img.shape
    b_w = int(w * border_pct)
    b_h = int(h * border_pct)

    # regiões de borda
    top = img[0:b_h, :]
    bottom = img[h - b_h : h, :]
    left = img[:, 0:b_w]
    right = img[:, w - b_w : w]

    top_blur = cv2.GaussianBlur(top, (51, 51), 0)
    bottom_blur = cv2.GaussianBlur(bottom, (51, 51), 0)
    left_blur = cv2.GaussianBlur(left, (51, 51), 0)
    right_blur = cv2.GaussianBlur(right, (51, 51), 0)

    img[0:b_h, :] = top_blur
    img[h - b_h : h, :] = bottom_blur
    img[:, 0:b_w] = left_blur
    img[:, w - b_w : w] = right_blur

    return img


def simple_cell_count(image_bgr: np.ndarray, min_area: int = 30, max_area: int = 5000):
    """Contagem simplificada de 'células' por segmentação e contorno."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    annotated = image_bgr.copy()
    count = 0
    for c in contours:
        area = cv2.contourArea(c)
        if min_area <= area <= max_area:
            (x, y), radius = cv2.minEnclosingCircle(c)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(annotated, center, radius, (0, 255, 0), 2)
            count += 1

    return annotated, count


def simulate_ai_analysis(image_bgr: np.ndarray, selected_scenario: dict = None):
    """
    'Análise de IA' simulada.
    Se um cenário específico for passado (selecionado pelo usuário), força o diagnóstico correto.
    Caso contrário, gera resultados aleatórios genéricos.
    """
    
    if selected_scenario:
        # MODO DIRECIONADO: O diagnóstico já está definido pelo input do usuário
        top_label = selected_scenario["diagnosis"]
        narrative = selected_scenario["description"]
        
        # Gera alta confiança para o diagnóstico correto
        confidence = random.uniform(0.88, 0.99)
        
        # Cria "competidores" falsos com baixa probabilidade para preencher o gráfico
        distractors = [
            "Tecido Normal",
            "Artefato de Técnica",
            "Inflamação Inespecífica",
            "Outra Alteração"
        ]
        # Remove o label correto se estiver na lista (improvável, mas por segurança)
        if top_label in distractors:
            distractors.remove(top_label)
            
        # Distribui o restante da probabilidade (1 - confidence) entre os distratores
        remaining_prob = 1.0 - confidence
        distractor_probs = np.random.dirichlet(np.ones(len(distractors))) * remaining_prob
        
        labels_sorted = [top_label] + distractors
        probs_sorted = np.concatenate(([confidence], distractor_probs))
        
    else:
        # MODO LIVRE (ALEATÓRIO ANTIGO)
        mean_intensity = float(image_bgr.mean())
        random.seed(int(mean_intensity))

        labels = [
            "Padrão inflamatório crônico",
            "Padrão inflamatório agudo",
            "Padrão neoplásico",
            "Tecido essencialmente normal",
            "Alterações degenerativas / regressivas",
        ]
        probs = np.abs(np.random.dirichlet(np.ones(len(labels))))
        order = np.argsort(probs)[::-1]
        labels_sorted = [labels[i] for i in order]
        probs_sorted = probs[order]

        top_label = labels_sorted[0]
        confidence = probs_sorted[0]

        # Lógica narrativa genérica antiga
        if "neoplásico" in top_label:
            narrative = "O algoritmo sugere padrão neoplásico, priorizando a correlação com achados clínicos."
        elif "inflamatório crônico" in top_label:
            narrative = "O algoritmo indica predomínio de inflamação crônica e fibrose."
        elif "inflamatório agudo" in top_label:
            narrative = "O algoritmo indica padrão inflamatório agudo (exsudativo)."
        elif "normal" in top_label:
            narrative = "O algoritmo não identifica alterações significativas."
        else:
            narrative = "O algoritmo sugere alterações degenerativas genéricas."

    return labels_sorted, probs_sorted, top_label, confidence, narrative


def generate_pdf_report(
    pil_image: Image.Image,
    student_name: str,
    case_id: str,
    tissue_type: str,
    comments: str,
    ai_summary: str | None = None,
    cell_count: int | None = None,
) -> bytes:
    """Gera um PDF simples com a lâmina e o relatório do aluno."""
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Simulador de Patologia Digital", ln=True, align="C")
    
    pdf.set_font("Arial", "I", 8)
    pdf.cell(0, 5, "Desenvolvido por Prof. Rodrigo Niskier", ln=True, align="C")

    pdf.set_font("Arial", "", 11)
    pdf.ln(4)
    pdf.cell(0, 8, f"Aluno: {student_name}", ln=True)
    pdf.cell(0, 8, f"Caso/Lâmina: {case_id}", ln=True)
    pdf.cell(0, 8, f"Tecido Analisado: {tissue_type}", ln=True)
    pdf.cell(0, 8, f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}", ln=True)

    # Resumo de IA e contagem (se disponíveis)
    pdf.ln(4)
    if ai_summary:
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 8, "Resumo da análise de IA (simulada):", ln=True)
        pdf.set_font("Arial", "", 11)
        pdf.multi_cell(0, 6, ai_summary)
    if cell_count is not None:
        pdf.ln(2)
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 8, "Estimativa de contagem de células:", ln=True)
        pdf.set_font("Arial", "", 11)
        pdf.cell(0, 6, f"Total estimado: {cell_count}", ln=True)

    # Imagem
    img_buffer = io.BytesIO()
    pil_image.save(img_buffer, format="PNG")
    img_buffer.seek(0)
    temp_path = "temp_slide.png"
    with open(temp_path, "wb") as f:
        f.write(img_buffer.read())

    pdf.ln(4)
    x = 10
    max_width = 190
    pdf.image(temp_path, x=x, w=max_width)

    # Comentários do aluno
    pdf.ln(8)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Raciocínio diagnóstico / observações do aluno:", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 6, comments or "(sem comentários)")

    # Corrige a saída do FPDF (compatibilidade bytes/string)
    try:
        val = pdf.output(dest='S')
    except TypeError:
        val = pdf.output()

    if isinstance(val, str):
        return val.encode('latin1')
    return bytes(val)


# ---------------------- LAYOUT PRINCIPAL ----------------------
st.title("🧫 Simulador de Análise Patológica")

# ADIÇÃO DO CRÉDITO
st.markdown("**Desenvolvido por Prof. Rodrigo Niskier**")

st.markdown(
    "Simulador interativo de **patologia** digital para treinamento em leitura de lâminas, "
    "contagem de células e letramento digital (incluindo IA simulada)."
)

with st.sidebar:
    st.header("Configurações gerais")
    student_name = st.text_input("Nome do aluno", placeholder="Digite seu nome")
    case_id = st.text_input("Identificação do caso", placeholder="Ex.: Lâmina 05")

    st.markdown("---")
    
    # NOVO SELETOR DE TECIDO
    st.markdown("**Contexto Histológico**")
    tissue_option = st.selectbox(
        "Selecione o tecido da lâmina carregada:",
        options=list(PATHOLOGY_SCENARIOS.keys())
    )
    
    st.markdown("---")
    zoom = st.slider("Zoom aproximado", 1.0, 4.0, 1.5, 0.25)
    show_grid = st.checkbox("Mostrar grade de contagem", value=False)
    grid_size = st.slider("Resolução da grade", 3, 10, 5)

    st.markdown("---")
    deidentify = st.checkbox("Aplicar desidentificação da lâmina (blur em bordas)", value=True)

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
            <div class="metric-label">Tecido Selecionado</div>
            <div class="metric-value">{}</div>
            <div class="metric-sub">Contexto para análise</div>
        </div>
        """.format(tissue_option if tissue_option != "Modo Livre (Aleatório)" else "Modo Livre"),
        unsafe_allow_html=True,
    )
with col_b:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Caso ativo</div>
            <div class="metric-value">{case_id or "Não definido"}</div>
            <div class="metric-sub">Identificação na sidebar</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_c:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">Ferramentas</div>
            <div class="metric-value">Zoom · Contagem · IA</div>
            <div class="metric-sub">Relatório em PDF para portfólio</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("")

tab1, tab2, tab3 = st.tabs(
    ["Visualização / navegação", "Contagem de células", "IA simulada + relatório"]
)

# Variáveis compartilhadas entre abas
base_image_bgr = None
processed_for_pdf = None
ai_summary_for_pdf = None
cell_count_for_pdf = None

if uploaded_file is not None:
    base_image_bgr = read_image(uploaded_file)
    if deidentify:
        base_image_bgr = deidentify_slide(base_image_bgr)

    # imagem com zoom + grade para uso geral
    zoomed = apply_zoom(base_image_bgr, zoom=zoom)
    if show_grid:
        zoomed = draw_grid(zoomed, grid_size=grid_size)
    processed_for_pdf = zoomed.copy()
else:
    st.info("Carregue uma imagem de lâmina na barra lateral para iniciar o simulador.")


# ---------------------- TAB 1: VISUALIZAÇÃO ----------------------
with tab1:
    if base_image_bgr is None:
        st.warning("Nenhuma lâmina carregada.")
    else:
        img_col, info_col = st.columns([3, 2])
        with img_col:
            st.subheader("Campo de visão")
            st.image(to_pil(zoomed), use_column_width=True)
            if deidentify:
                st.caption("Desidentificação automática ativa (blur em bordas da lâmina).")

        with info_col:
            st.subheader("Tarefas sugeridas")
            st.markdown(
                """
                - Identifique regiões de interesse (inflamação, necrose, atipias).  
                - Use o **zoom** para simular diferentes aumentos do microscópio.  
                - Ative a **grade** para exercícios de contagem ou estimativa de proporções.  
                """
            )
            # Dica contextual baseada na seleção
            if tissue_option != "Modo Livre (Aleatório)":
                st.info(f"💡 Dica: Procure por características típicas de **{tissue_option}**.")
            
            quick_notes = st.text_area(
                "Observações rápidas (o que chama a sua atenção nesta lâmina?).",
                height=160,
                key="quick_notes",
            )


# ---------------------- TAB 2: CONTAGEM DE CÉLULAS ----------------------
with tab2:
    if base_image_bgr is None:
        st.warning("Nenhuma lâmina carregada.")
    else:
        st.subheader("Estimativa automatizada de contagem de células (didático)")
        st.caption(
            "Este módulo usa visão computacional simples para estimar o número de 'células' na imagem. "
            "Os resultados têm finalidade **pedagógica**, não diagnóstica."
        )

        c1, c2 = st.columns(2)
        with c1:
            min_area = st.slider("Área mínima (pixels)", 10, 500, 30, 5)
        with c2:
            max_area = st.slider("Área máxima (pixels)", 500, 10000, 5000, 100)

        annotated, count_cells = simple_cell_count(zoomed, min_area=min_area, max_area=max_area)
        cell_count_for_pdf = int(count_cells)

        img_col, info_col = st.columns([3, 2])
        with img_col:
            st.image(to_pil(annotated), caption=f"Células detectadas: {count_cells}", use_column_width=True)

        with info_col:
            st.markdown(
                f"""
                **Total estimado de 'células'**: {count_cells}  

                Sugestões de uso em sala de aula:  
                - Comparar a contagem automática com a estimativa visual do aluno.  
                - Discutir **fontes de erro** (células sobrepostas, artefatos, ruído de coloração).  
                - Relacionar a contagem com índices morfométricos ou escores semi-quantitativos.  
                """
            )


# ---------------------- TAB 3: IA SIMULADA + RELATÓRIO ----------------------
with tab3:
    if base_image_bgr is None:
        st.warning("Nenhuma lâmina carregada.")
    else:
        st.subheader("Análise de IA (simulada) e relatório do aluno")

        # mostra novamente a lâmina (estado atual: zoom + grade + (des)identificação)
        st.markdown("### Campo de visão para análise de IA")
        img_col, info_col = st.columns([3, 2])

        # placeholder para animação
        scan_placeholder = img_col.empty()

        # imagem base para animação do scanner
        scan_base = zoomed.copy()
        h, w, _ = scan_base.shape

        with info_col:
            st.markdown(
                "> Clique em **Gerar análise de IA** para simular o algoritmo percorrendo a lâmina.\n"
                "> A animação representa um scanner X‑Y varrendo o campo de visão."
            )
            
            if tissue_option != "Modo Livre (Aleatório)":
                st.success(f"Contexto definido: **{tissue_option}**. A IA buscará padrões específicos.")
            
            start_scan = st.button("▶️ Gerar análise de IA (simulada)")

        ai_summary_for_pdf = None  # garante reset local
        labels_sorted = probs_sorted = top_label = confidence = narrative = None

        if start_scan:
            # animação: linha percorrendo a lâmina em X e depois em Y
            n_steps_x = 25
            n_steps_y = 25

            # varredura horizontal (eixo X)
            for i in range(n_steps_x):
                frame = scan_base.copy()
                x_pos = int(w * (i / (n_steps_x - 1)))
                cv2.line(frame, (x_pos, 0), (x_pos, h), (0, 255, 0), 2)
                scan_placeholder.image(to_pil(frame), use_column_width=True)
                time.sleep(0.03)

            # varredura vertical (eixo Y)
            for j in range(n_steps_y):
                frame = scan_base.copy()
                y_pos = int(h * (j / (n_steps_y - 1)))
                cv2.line(frame, (0, y_pos), (w, y_pos), (0, 255, 0), 2)
                scan_placeholder.image(to_pil(frame), use_column_width=True)
                time.sleep(0.03)

            # Define qual cenário passar (se não for aleatório)
            scenario_data = PATHOLOGY_SCENARIOS.get(tissue_option)

            # faz a "inferência" após a animação
            labels_sorted, probs_sorted, top_label, confidence, narrative = simulate_ai_analysis(
                scan_base, selected_scenario=scenario_data
            )

            # mostra imagem final sem linha, como resultado
            scan_placeholder.image(to_pil(scan_base), use_column_width=True)

            st.success("Análise de IA simulada concluída.")
        else:
            # estado inicial: apenas imagem sem scanner
            scan_placeholder.image(to_pil(scan_base), use_column_width=True)

        # se já temos resultado (após clicar no botão)
        if labels_sorted is not None:
            ai_summary_for_pdf = (
                f"Diagnóstico sugerido: {top_label} (confiança: {confidence*100:.1f}%). "
                f"\nAchados: {narrative}"
            )

            st.markdown(
                "> Esta IA é **simulada**, construída apenas para fins didáticos, sem uso real em diagnóstico."
            )
            st.markdown("### Resultado da Análise Simulada")
            for label, prob in zip(labels_sorted, probs_sorted):
                st.write(f"- **{label}**: {prob*100:.1f}%")

            st.info(f"**Descrição Técnica:** {narrative}")

        st.markdown("---")
        st.markdown("### Raciocínio diagnóstico do aluno")
        comments = st.text_area(
            "Descreva o que você concorda ou discorda da sugestão da IA, incluindo diagnóstico diferencial e correlação clínico-patológica.",
            height=220,
        )

        include_image = st.checkbox("Incluir captura da lâmina no PDF", value=True)
        include_ai = st.checkbox("Incluir resumo da IA simulada no PDF", value=True)
        include_count = st.checkbox("Incluir contagem de células estimada no PDF", value=True)

        if st.button("📄 Gerar PDF do caso", type="primary"):
            if processed_for_pdf is None:
                st.warning("Não foi possível gerar a imagem processada.")
            else:
                pil_img = to_pil(processed_for_pdf) if include_image else Image.new(
                    "RGB", (800, 600), "white"
                )
                pdf_bytes = generate_pdf_report(
                    pil_image=pil_img,
                    student_name=student_name or "Aluno não identificado",
                    case_id=case_id or "Caso sem identificação",
                    tissue_type=tissue_option, # Adicionado ao PDF
                    comments=comments,
                    ai_summary=ai_summary_for_pdf if (include_ai and ai_summary_for_pdf) else None,
                    cell_count=cell_count_for_pdf if (include_count and cell_count_for_pdf is not None) else None,
                )

                st.success("PDF gerado com sucesso. Faça o download abaixo.")
                st.download_button(
                    label="⬇️ Baixar relatório em PDF",
                    data=pdf_bytes,
                    file_name=f"relatorio_patologia_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                )
