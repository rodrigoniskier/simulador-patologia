import streamlit as st
import cv2
import numpy as np
from PIL import Image
from fpdf import FPDF
import io
import tempfile
import random
import os

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="SimuPath AI - Prof. Rodrigo Niskier", layout="wide", page_icon="🔬")

# --- COPYRIGHT E CABEÇALHO ---
st.sidebar.markdown("---")
st.sidebar.caption("Desenvolvido para fins educativos")
st.sidebar.markdown("**© 2026 Prof. Rodrigo Niskier**")

# Título Principal
st.title("🔬 SimuPath AI: Workstation de Patologia Digital")
st.markdown("""
**Simulação de Ambiente de Diagnóstico Assistido por Computador (CAD)**
Este sistema simula a triagem de lâminas histológicas utilizando algoritmos de Deep Learning para deteção de padrões suspeitos.
""")

# --- FUNÇÕES AUXILIARES ---

def gerar_metadados():
    """Gera dados fictícios de um paciente."""
    return {
        "Nome": "Maria Silva",
        "ID_Caso": f"SP-{random.randint(20000, 99999)}-26",
        "Data_Nasc": "1980-05-20",
        "Origem": "Hospital Central - Oncologia",
        "Stain": "H&E (Hematoxilina e Eosina)"
    }

def aplicar_heatmap(img_array):
    """
    Simula um mapa de atenção (Heatmap) comum em IA médica.
    Na realidade, isto viria de uma rede neural. Aqui simulamos usando processamento de imagem
    para destacar áreas de alta densidade celular (núcleos).
    """
    # Converter para cinza
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    # Inverter (núcleos são escuros, queremos que fiquem claros para o heatmap)
    inv_gray = cv2.bitwise_not(gray)
    # Aplicar mapa de cores (JET é comum em medicina para mostrar 'intensidade')
    heatmap = cv2.applyColorMap(inv_gray, cv2.COLORMAP_JET)
    # Misturar imagem original com heatmap
    overlay = cv2.addWeighted(img_array, 0.6, heatmap, 0.4, 0)
    return overlay

def criar_pdf(imagem_pil, metadados, texto_laudo, score_ia):
    """Gera um PDF com o laudo médico simulado."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Cabeçalho
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Relatório de Análise Patológica Digital", ln=True, align='C')
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt="SimuPath AI - Educational Suite | © 2026 Prof. Rodrigo Niskier", ln=True, align='C')
    pdf.line(10, 30, 200, 30)
    pdf.ln(10)

    # Dados do Paciente
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Identificação do Caso", ln=True, align='L')
    pdf.set_font("Arial", size=11)
    for key, value in metadados.items():
        pdf.cell(200, 8, txt=f"{key}: {value}", ln=True)
    pdf.ln(5)

    # Imagem
    # Salvar imagem temporariamente para inserir no PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        imagem_pil.save(tmp_file.name)
        pdf.image(tmp_file.name, x=10, y=None, w=100)
        os.unlink(tmp_file.name) # Limpar temp

    pdf.ln(5)

    # Análise de IA
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Análise Computacional (IA Preliminar)", ln=True, align='L')
    pdf.set_font("Arial", size=11)
    pdf.cell(200, 8, txt=f"Score de Risco de Malignidade: {score_ia}", ln=True)
    pdf.cell(200, 8, txt="Obs: A IA serve apenas como suporte à triagem.", ln=True)
    pdf.ln(5)

    # Laudo Médico (Texto do aluno)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Laudo Macroscópico e Microscópico (Médico Residente)", ln=True, align='L')
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 8, txt=texto_laudo)
    
    pdf.ln(20)
    pdf.cell(200, 10, txt="_______________________________________", ln=True, align='C')
    pdf.cell(200, 5, txt="Assinatura Digital do Patologista", ln=True, align='C')

    return pdf.output(dest='S').encode('latin-1')

# --- ESTADO DA SESSÃO ---
if 'metadados' not in st.session_state:
    st.session_state['metadados'] = gerar_metadados()
if 'anonimizado' not in st.session_state:
    st.session_state['anonimizado'] = False

# --- BARRA LATERAL ---
st.sidebar.header("📁 Gestão de Casos")
uploaded_file = st.sidebar.file_uploader("Importar Lâmina Digital (WSI)", type=["jpg", "png", "jpeg"])

# --- LÓGICA PRINCIPAL ---

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # Tabs para organizar o fluxo de trabalho profissional
    tab1, tab2, tab3 = st.tabs(["🖥️ Visualização & Metadados", "🤖 Análise de IA", "📝 Laudo & Relatório"])

    # --- ABA 1: VISUALIZAÇÃO ---
    with tab1:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(image, caption="Whole-Slide Image (Vista Original)", use_container_width=True)
        with col2:
            st.subheader("Dados DICOM")
            if not st.session_state['anonimizado']:
                st.error("⚠️ Identificadores Visíveis (PHI)")
                st.json(st.session_state['metadados'])
                if st.button("🛡️ Desidentificar Dados"):
                    st.session_state['metadados']['Nome'] = "ANONIMO"
                    st.session_state['metadados']['ID_Caso'] = f"Anon-{random.randint(1000,9999)}"
                    st.session_state['metadados']['Data_Nasc'] = "****-**-**"
                    st.session_state['anonimizado'] = True
                    st.rerun()
            else:
                st.success("✅ Dados Anonimizados")
                st.json(st.session_state['metadados'])
            
            st.info("Protocolo: Assegure-se de que os dados estão anonimizados antes de iniciar a análise de IA na nuvem.")

    # --- ABA 2: ANÁLISE DE IA ---
    with tab2:
        st.subheader("Deep Learning / Pathomics")
        
        # QC Automático
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        score_foco = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        col_qc1, col_qc2, col_qc3 = st.columns(3)
        col_qc1.metric("Controlo de Qualidade (Foco)", f"{score_foco:.1f}")
        
        # Simulação de Score de Malignidade (Aleatório ponderado para fins educativos)
        # Numa app real, isto viria do modelo treinado
        risco = random.randint(10, 95) if score_foco > 100 else 0
        cor_risco = "normal" if risco < 50 else "off" # Streamlit usa "inverse" ou cores especificas para destacar
        
        if score_foco < 100:
            st.error("⛔ Imagem com qualidade insuficiente para análise algorítmica.")
        else:
            col_qc2.success("QC Aprovado")
            
            if st.button("Executar Diagnóstico Assistido"):
                with st.spinner('A processar Redes Neurais Convolucionais...'):
                    # 1. Gerar Heatmap
                    img_heatmap = aplicar_heatmap(img_array)
                    
                    # 2. Segmentação simples (Threshold)
                    gray_blur = cv2.GaussianBlur(gray, (7, 7), 0)
                    thresh = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                 cv2.THRESH_BINARY_INV, 11, 2)
                    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    n_celulas = len(contornos)
                    
                    # Exibir Resultados
                    st.markdown("### Resultados da IA")
                    c1, c2 = st.columns(2)
                    
                    with c1:
                        st.image(img_heatmap, caption="Mapa de Atenção (Saliency Map)", use_container_width=True)
                        st.caption("As áreas a vermelho/azul indicam regiões onde a IA detetou padrões de alta densidade celular ou atipia nuclear.")
                    
                    with c2:
                        st.metric("Contagem Celular Estimada", n_celulas)
                        st.markdown(f"**Score de Probabilidade de Malignidade:**")
                        st.progress(risco)
                        st.markdown(f"### {risco}%")
                        
                        if risco > 70:
                            st.warning("⚠️ Classificação Sugerida: ALTO RISCO / SUSPEITO")
                        else:
                            st.success("✅ Classificação Sugerida: BAIXO RISCO / BENIGNO")

                    st.toast("Análise concluída com sucesso!")

    # --- ABA 3: LAUDO ---
    with tab3:
        st.subheader("Emissão de Relatório Médico")
        st.markdown("Como patologista em treino, utilize os dados da IA e a sua observação para redigir o laudo.")
        
        texto_padrao = "Exame microscópico revela fragmentos de tecido com arquitetura preservada/alterada. Observa-se..."
        laudo_texto = st.text_area("Descrição Macroscópica e Microscópica", height=150, value=texto_padrao)
        
        col_down1, col_down2 = st.columns([1, 1])
        
        with col_down1:
            st.markdown("Ao confirmar, o laudo será assinado digitalmente e gerado em PDF.")
            if st.button("Assinar e Gerar PDF"):
                if not st.session_state['anonimizado']:
                    st.error("Não é possível gerar o laudo com dados identificados. Por favor anonimize na primeira aba.")
                else:
                    # Gerar PDF
                    pdf_bytes = criar_pdf(image, st.session_state['metadados'], laudo_texto, "Risco Calculado na Análise")
                    
                    st.success("📄 Relatório gerado com sucesso!")
                    st.download_button(
                        label="📥 Download Relatório Médico (PDF)",
                        data=pdf_bytes,
                        file_name=f"Laudo_{st.session_state['metadados']['ID_Caso']}.pdf",
                        mime="application/pdf"
                    )

else:
    # Ecrã de Boas-vindas
    st.info("👋 Bem-vindo à Workstation SimuPath. Carregue uma lâmina digitalizada para começar.")
    st.markdown("""
    ### Guia Rápido:
    1. **Upload:** Carregue uma imagem histológica (H&E).
    2. **Privacidade:** Verifique e anonimize os dados do paciente.
    3. **IA:** Use a aba de análise para ver o "Mapa de Calor" e a estimativa de risco.
    4. **Laudo:** Escreva a sua conclusão e baixe o PDF oficial assinado.
    """)
    st.markdown("---")
    st.caption("© 2026 Prof. Rodrigo Niskier | Ferramenta de Letramento Digital em Medicina")
