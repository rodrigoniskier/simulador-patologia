import streamlit as st
import cv2
import numpy as np
from PIL import Image
import random

# Configuração da página
st.set_page_config(page_title="SimuPath AI - Educativo", layout="wide")

# Título e Introdução baseada no contexto de Patologia Digital
st.title("🔬 SimuPath AI: Simulador de Patologia Digital")
st.markdown("""
Esta aplicação simula o fluxo de trabalho de Patologia Digital (DPI) e Inteligência Artificial,
focando-se em conceitos chave como **Controlo de Qualidade (QC)**, **Desidentificação** e **Análise de Imagem**,
conforme discutido no workshop do NCI[cite: 26, 31].
""")

# --- BARRA LATERAL: Configuração e Upload ---
st.sidebar.header("1. Digitalização e Entrada")
uploaded_file = st.sidebar.file_uploader("Carregar Lâmina Digital (Imagem .jpg ou .png)", type=["jpg", "png", "jpeg"])

# Função para gerar dados fictícios do paciente (Simulando metadados DICOM)
def gerar_metadados():
    return {
        "Nome": "Maria Silva",
        "ID_Paciente": "12345-PT",
        "Data_Nasc": "1980-05-20",
        "Tipo_Amostra": "Biópsia Pulmonar",
        "Scanner": "Scanner-X WSI"
    }

# Inicializar estado da sessão para metadados
if 'metadados' not in st.session_state:
    st.session_state['metadados'] = gerar_metadados()
if 'anonimizado' not in st.session_state:
    st.session_state['anonimizado'] = False

# --- LÓGICA PRINCIPAL ---

if uploaded_file is not None:
    # Converter o ficheiro carregado para formato que o computador entenda (Array NumPy)
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # Coluna 1: Visualização e Dados, Coluna 2: Análise
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("2. Visualizador e Metadados (DICOM)")
        st.image(image, caption="Whole-Slide Image (WSI) - Região de Interesse", use_container_width=True)
        
        st.info("Os sistemas de patologia digital usam padrões como DICOM para gerir metadados.")
        
        # Exibir Metadados
        st.markdown("### Dados do Paciente")
        if not st.session_state['anonimizado']:
            st.json(st.session_state['metadados'])
            
            # Botão de Desidentificação
            st.warning("⚠️ Atenção: Dados contêm PHI (Informação de Saúde Protegida).")
            if st.button("Executar Protocolo de Desidentificação"):
                # Simula a remoção de PHI conforme normas HIPAA/GDPR 
                st.session_state['metadados']['Nome'] = "ANONIMO"
                st.session_state['metadados']['ID_Paciente'] = f"Hash-{random.randint(1000,9999)}"
                st.session_state['metadados']['Data_Nasc'] = "####-##-##"
                st.session_state['anonimizado'] = True
                st.rerun()
        else:
            st.success("✅ Dados Desidentificados com sucesso. Pronto para partilha ou análise secundária.")
            st.json(st.session_state['metadados'])

    with col2:
        st.subheader("3. Análise Computacional")
        
        # --- MÓDULO DE CONTROLO DE QUALIDADE (QC) ---
        st.markdown("#### A. Controlo de Qualidade (QC)")
        st.markdown("O QC verifica foco, artefatos e integridade da imagem antes da análise.")
        
        # Simulação simples de detecção de desfoque (Blur) usando variação Laplaciana
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        variancia_laplaciana = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        st.metric("Índice de Nitidez (Score)", f"{variancia_laplaciana:.2f}")
        
        limiar_foco = 100.0 # Valor arbitrário para simulação
        if variancia_laplaciana < limiar_foco:
            st.error("❌ Falha no QC: Imagem muito desfocada. Re-scan necessário.")
            analise_permitida = False
        else:
            st.success("✅ QC Aprovado: Imagem nítida e adequada para diagnóstico.")
            analise_permitida = True

        st.divider()

        # --- MÓDULO DE IA ---
        st.markdown("#### B. Assistente de IA (Simulação)")
        st.markdown("""
        A IA pode ser usada para contar células, graduar tumores ou quantificar biomarcadores (ex: Ki-67, PD-L1)[cite: 41, 44].
        *Nota: Esta é uma simulação simples baseada em processamento de cor.*
        """)

        if analise_permitida:
            if st.button("Executar Análise de IA"):
                with st.spinner('A processar algoritmo de segmentação...'):
                    # Simulação: Segmentação simples por limiar (Thresholding)
                    # Converte para escala de cinza e aplica um blur suave
                    gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                    
                    # Aplica limiar adaptativo para encontrar "células" (regiões escuras)
                    thresh = cv2.adaptiveThreshold(gray_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                 cv2.THRESH_BINARY_INV, 11, 2)
                    
                    # Contar contornos (simulando contagem de células)
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contagem_celulas = len(contours)
                    
                    # Desenhar contornos na imagem original para visualização
                    img_analisada = img_array.copy()
                    cv2.drawContours(img_analisada, contours, -1, (0, 255, 0), 2)
                    
                    st.image(img_analisada, caption="Resultado da IA: Segmentação de Estruturas", use_container_width=True)
                    st.info(f"📊 A IA detetou **{contagem_celulas}** estruturas de interesse nesta região.")
                    st.markdown("> **Nota Educativa:** A IA serve como suporte à decisão. O patologista deve validar estes resultados[cite: 76, 258].")
        else:
            st.warning("A análise de IA está bloqueada até que o QC da imagem seja aprovado.")

else:
    st.info("👈 Por favor, carrega uma imagem na barra lateral para iniciar a simulação.")
    st.markdown("### Instruções:")
    st.markdown("""
    1. Carrega uma imagem de tecido (podes procurar por 'H&E histology' no Google Imagens).
    2. Observa os metadados e pratica a **Desidentificação**.
    3. Verifica se a imagem passa no **Controlo de Qualidade**.
    4. Executa a **IA** para ver uma segmentação automática básica.
    """)