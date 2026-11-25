
import cv2
import numpy as np
import streamlit as st

import base64
import html
import re

from datetime import datetime
from urllib.parse import quote


# ==============================================================================
# CONFIGURAÇÕES DA APLICAÇÃO
# ==============================================================================

st.set_page_config(
    page_title="Triagem Visual Estética",
    page_icon="✨",
    layout="wide"
)

st.title("✨ Triagem Visual Estética")
st.caption(
    "Sistema de análise visual preliminar, anamnese e recomendações estéticas."
)


# ==============================================================================
# 1. DETECÇÃO FACIAL
# ==============================================================================

@st.cache_resource
def carregar_detector():
    caminho = cv2.data.haarcascades + (
        "haarcascade_frontalface_default.xml"
    )

    detector = cv2.CascadeClassifier(caminho)

    if detector.empty():
        raise RuntimeError(
            "Não foi possível carregar o classificador facial do OpenCV."
        )

    return detector


def detectar_rosto(imagem):

    detector = carregar_detector()

    cinza = cv2.cvtColor(
        imagem,
        cv2.COLOR_BGR2GRAY
    )

    rostos = detector.detectMultiScale(
        cinza,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )

    if len(rostos) == 0:
        return None

    # Seleciona o maior rosto detectado
    rosto = max(
        rostos,
        key=lambda r: r[2] * r[3]
    )

    return tuple(map(int, rosto))


# ==============================================================================
# 2. DEFINIÇÃO DAS REGIÕES FACIAIS
# ==============================================================================

def definir_zonas(rosto, largura, altura):

    x, y, rw, rh = rosto

    def coordenadas(x1, y1, x2, y2):

        return (
            max(0, min(int(x + rw * x1), largura)),
            max(0, min(int(y + rh * y1), altura)),
            max(0, min(int(x + rw * x2), largura)),
            max(0, min(int(y + rh * y2), altura))
        )

    zonas = {

        "Testa e Glabela": coordenadas(
            0.18, 0.12, 0.82, 0.38
        ),

        "Bochecha esquerda da imagem": coordenadas(
            0.12, 0.48, 0.43, 0.75
        ),

        "Bochecha direita da imagem": coordenadas(
            0.57, 0.48, 0.88, 0.75
        )
    }

    return zonas


# ==============================================================================
# 3. ANÁLISE DOS RECORTES
# ==============================================================================

def analisar_recorte(recorte):

    if recorte is None or recorte.size == 0:
        raise ValueError("Região facial vazia.")

    cinza = cv2.cvtColor(
        recorte,
        cv2.COLOR_BGR2GRAY
    )

    cinza_suave = cv2.GaussianBlur(
        cinza,
        (3, 3),
        0
    )

    bordas = cv2.Canny(
        cinza_suave,
        25,
        80
    )

    indice_linhas = round(
        float(np.mean(bordas > 0) * 100),
        2
    )

    lab = cv2.cvtColor(
        recorte,
        cv2.COLOR_BGR2LAB
    )

    _, canal_a, _ = cv2.split(lab)

    indice_tom = round(
        float(np.std(canal_a)),
        2
    )

    return indice_linhas, indice_tom


# ==============================================================================
# 4. MOTOR DE RECOMENDAÇÕES
# ==============================================================================

def gerar_recomendacoes(
    linhas_testa,
    tom_medio,
    queixa,
    tipo_pele,
    restricao
):

    procedimentos = []

    dermocosmeticos = {
        "Limpeza": "",
        "Ativos de Tratamento": [],
        "Hidratação e Barreira": "",
        "Fotoproteção": ""
    }

    # --------------------------------------------------------------------------
    # PROCEDIMENTOS
    # --------------------------------------------------------------------------

    if restricao:

        procedimentos.append(
            "Avaliação de hidratação e recuperação da barreira cutânea, "
            "com protocolos suaves e compatíveis com a condição individual."
        )

        procedimentos.append(
            "Reavaliação profissional antes de realizar peelings, "
            "microagulhamento, procedimentos injetáveis ou tratamentos "
            "com ativos potencialmente irritantes."
        )

    else:

        if queixa == "Linhas de expressão":

            procedimentos.append(
                "Avaliar protocolo de revitalização facial e hidratação. "
                "A possibilidade de toxina botulínica, skinbooster ou "
                "microagulhamento depende de avaliação presencial, "
                "indicação e habilitação profissional."
            )

        elif queixa == "Manchas/Uniformização":

            procedimentos.append(
                "Avaliar protocolo de uniformização do tom, considerando "
                "fototipo, histórico de exposição solar e características "
                "das alterações pigmentares."
            )

            procedimentos.append(
                "Peelings superficiais, microdermoabrasão ou tecnologias "
                "de luz somente após avaliação individual e verificação "
                "de contraindicações."
            )

        elif queixa == "Hidratação/Textura":

            procedimentos.append(
                "Avaliar hidratação facial, revitalização e recuperação "
                "da barreira cutânea."
            )

            procedimentos.append(
                "Outros recursos, como microagulhamento ou tecnologias "
                "de estímulo de colágeno, dependem de indicação e "
                "avaliação profissional."
            )

    # --------------------------------------------------------------------------
    # LIMPEZA
    # --------------------------------------------------------------------------

    if tipo_pele == "Oleosa":

        dermocosmeticos["Limpeza"] = (
            "Gel de limpeza suave, adequado à pele oleosa, "
            "sem provocar ressecamento excessivo."
        )

    elif tipo_pele == "Mista":

        dermocosmeticos["Limpeza"] = (
            "Gel ou espuma de limpeza equilibrante, "
            "adequado à pele mista."
        )

    elif tipo_pele == "Seca/Ressecada":

        dermocosmeticos["Limpeza"] = (
            "Loção ou creme de limpeza suave, "
            "com componentes hidratantes."
        )

    else:

        dermocosmeticos["Limpeza"] = (
            "Produto de limpeza suave, sem fragrância, "
            "adequado à pele sensível."
        )

    # --------------------------------------------------------------------------
    # ATIVOS DE TRATAMENTO
    # --------------------------------------------------------------------------

    if restricao:

        dermocosmeticos["Ativos de Tratamento"].append(
            "Não indicar automaticamente ácidos, retinoides ou "
            "ativos clareadores. A escolha deve considerar gestação, "
            "amamentação, sensibilização e avaliação profissional."
        )

    else:

        if queixa == "Linhas de expressão":

            dermocosmeticos["Ativos de Tratamento"].append(
                "Peptídeos e ácido hialurônico em formulações tópicas, "
                "considerando tolerância e necessidade individual."
            )

        elif queixa == "Manchas/Uniformização":

            dermocosmeticos["Ativos de Tratamento"].append(
                "Niacinamida ou vitamina C em formulações adequadas "
                "ao tipo de pele, com avaliação da tolerância."
            )

        elif queixa == "Hidratação/Textura":

            dermocosmeticos["Ativos de Tratamento"].append(
                "Ácido hialurônico, pantenol e outros agentes "
                "hidratantes para auxiliar na manutenção da pele."
            )

    # --------------------------------------------------------------------------
    # HIDRATAÇÃO
    # --------------------------------------------------------------------------

    if tipo_pele in ("Oleosa", "Mista"):

        dermocosmeticos["Hidratação e Barreira"] = (
            "Sérum ou gel-creme não comedogênico, "
            "com ácido hialurônico e pantenol."
        )

    elif tipo_pele == "Sensível":

        dermocosmeticos["Hidratação e Barreira"] = (
            "Hidratante reparador sem fragrância, "
            "com componentes calmantes e restauradores da barreira."
        )

    else:

        dermocosmeticos["Hidratação e Barreira"] = (
            "Creme hidratante com ceramidas, glicerina "
            "ou outros agentes de reparação da barreira cutânea."
        )

    # --------------------------------------------------------------------------
    # FOTOPROTEÇÃO
    # --------------------------------------------------------------------------

    if queixa == "Manchas/Uniformização":

        dermocosmeticos["Fotoproteção"] = (
            "Protetor solar de amplo espectro, FPS 50 ou superior. "
            "Considerar versão com cor para auxiliar na proteção "
            "contra a luz visível."
        )

    else:

        dermocosmeticos["Fotoproteção"] = (
            "Protetor solar de amplo espectro, FPS 50 ou superior, "
            "com reaplicação conforme exposição."
        )

    return procedimentos, dermocosmeticos


# ==============================================================================
# 5. MONTAGEM DA MENSAGEM DO WHATSAPP
# ==============================================================================

def montar_link_whatsapp(
    queixa,
    tipo_pele,
    restricao,
    linhas_testa,
    tom_medio,
    procedimentos,
    dermocosmeticos,
    observacoes,
    prazo_followup,
    telefone
):

    lista_proc = "\n".join(
        f"- {p}" for p in procedimentos
    )

    lista_ativos = "\n".join(
        f"- {a}"
        for a in dermocosmeticos["Ativos de Tratamento"]
    )

    mensagem = f"""
*RESUMO DA TRIAGEM VISUAL ESTÉTICA*

*Objetivo:* {queixa}
*Tipo de pele:* {tipo_pele}
*Restrição informada:* {"Sim" if restricao else "Não"}

*Indicadores visuais experimentais:*
- Testa: {linhas_testa}%
- Variação de tonalidade das bochechas: {tom_medio}

*PROCEDIMENTOS PARA AVALIAÇÃO*
{lista_proc}

*ROTINA DERMOCOSMÉTICA*

*Limpeza:* {dermocosmeticos["Limpeza"]}

*Ativos:*
{lista_ativos}

*Hidratação:*
{dermocosmeticos["Hidratação e Barreira"]}

*Fotoproteção:*
{dermocosmeticos["Fotoproteção"]}

*Retorno sugerido:* {prazo_followup}
"""

    if observacoes.strip():

        mensagem += (
            "\n*Observações profissionais:*\n"
            + observacoes.strip()
        )

    mensagem += """

Aviso: Triagem visual educativa. Os indicadores não representam
diagnóstico dermatológico. A conduta final depende de avaliação presencial.
"""

    numero = re.sub(
        r"\D",
        "",
        telefone or ""
    )

    if len(numero) in (10, 11):
        numero = "55" + numero

    if len(numero) not in (0, 12, 13):
        numero = ""

    if numero and not numero.startswith("55"):
        numero = ""

    texto_url = quote(
        mensagem,
        safe=""
    )

    if numero:
        return f"https://wa.me/{numero}?text={texto_url}"

    return f"https://wa.me/?text={texto_url}"


# ==============================================================================
# 6. GERAÇÃO DO RELATÓRIO HTML
# ==============================================================================

def gerar_relatorio_html(
    imagem_marcada,
    resultados_regioes,
    linhas_testa,
    tom_medio,
    queixa,
    tipo_pele,
    restricao,
    procedimentos,
    dermocosmeticos,
    observacoes,
    prazo_followup
):

    sucesso, buffer_img = cv2.imencode(
        ".jpg",
        imagem_marcada,
        [cv2.IMWRITE_JPEG_QUALITY, 90]
    )

    if not sucesso:
        raise RuntimeError(
            "Não foi possível preparar a imagem do relatório."
        )

    imagem_base64 = base64.b64encode(
        buffer_img.tobytes()
    ).decode("utf-8")

    data = datetime.now().strftime(
        "%d/%m/%Y às %H:%M"
    )

    linhas_html = ""

    for nome, valores in resultados_regioes.items():

        linhas_html += f"""
        <tr>
            <td>{html.escape(nome)}</td>
            <td>{valores[0]}%</td>
            <td>{valores[1]}</td>
        </tr>
        """

    procedimentos_html = "".join(
        f"<li>{html.escape(p)}</li>"
        for p in procedimentos
    )

    ativos_html = "".join(
        f"<li>{html.escape(a)}</li>"
        for a in dermocosmeticos["Ativos de Tratamento"]
    )

    observacoes_html = (
        html.escape(observacoes.strip())
        if observacoes.strip()
        else "Nenhuma observação informada."
    )

    restricao_texto = (
        "Sim - requer avaliação individual"
        if restricao
        else "Não informada"
    )

    documento = f"""
    <!DOCTYPE html>
    <html lang="pt-BR">

    <head>

    <meta charset="UTF-8">

    <meta name="viewport" content="width=device-width, initial-scale=1">

    <title>Relatório de Triagem Visual Estética</title>

    <style>

        body {{
            font-family: Arial, sans-serif;
            max-width: 900px;
            margin: 35px auto;
            padding: 20px;
            color: #333;
            line-height: 1.6;
        }}

        h1, h2 {{
            color: #28594b;
        }}

        h1 {{
            text-align: center;
        }}

        .subtitulo {{
            text-align: center;
            color: #777;
        }}

        .secao {{
            margin-top: 30px;
            page-break-inside: avoid;
        }}

        .painel {{
            background: #f3f7f5;
            padding: 15px;
            border-radius: 8px;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}

        td, th {{
            border: 1px solid #ddd;
            padding: 10px;
            text-align: left;
        }}

        th {{
            background: #e5efea;
        }}

        img {{
            display: block;
            max-width: 500px;
            width: 100%;
            margin: 20px auto;
            border-radius: 8px;
        }}

        .aviso {{
            background: #f4f4f4;
            padding: 15px;
            margin-top: 30px;
            font-size: 13px;
        }}

        button {{
            background: #28594b;
            color: white;
            padding: 12px 20px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 15px;
        }}

        @media print {{

            body {{
                margin: 10mm;
                padding: 0;
            }}

            button {{
                display: none;
            }}

            .secao {{
                page-break-inside: avoid;
            }}

        }}

    </style>

    </head>

    <body>

    <h1>RELATÓRIO DE TRIAGEM VISUAL ESTÉTICA</h1>

    <p class="subtitulo">
        Avaliação visual preliminar e recomendações personalizadas
    </p>

    <p><strong>Data da avaliação:</strong> {data}</p>

    <div class="secao">

        <h2>1. Perfil da cliente</h2>

        <div class="painel">

            <p><strong>Objetivo:</strong> {html.escape(queixa)}</p>

            <p><strong>Tipo de pele:</strong> {html.escape(tipo_pele)}</p>

            <p><strong>Restrição:</strong> {restricao_texto}</p>

        </div>

    </div>

    <div class="secao">

        <h2>2. Análise visual da imagem</h2>

        <p>
            <strong>Indicador de bordas na testa:</strong>
            {linhas_testa}%
        </p>

        <p>
            <strong>Variação de tonalidade nas bochechas:</strong>
            {tom_medio}
        </p>

        <img
            src="data:image/jpeg;base64,{imagem_base64}"
            alt="Fotografia com regiões faciais analisadas"
        >

        <table>

            <tr>
                <th>Região facial</th>
                <th>Indicador de bordas</th>
                <th>Variação de tonalidade</th>
            </tr>

            {linhas_html}

        </table>

    </div>

    <div class="secao">

        <h2>3. Procedimentos sugeridos para avaliação</h2>

        <ul>
            {procedimentos_html}
        </ul>

    </div>

    <div class="secao">

        <h2>4. Rotina dermocosmética</h2>

        <p>
            <strong>Limpeza:</strong>
            {html.escape(dermocosmeticos["Limpeza"])}
        </p>

        <strong>Ativos de tratamento:</strong>

        <ul>
            {ativos_html}
        </ul>

        <p>
            <strong>Hidratação:</strong>
            {html.escape(dermocosmeticos["Hidratação e Barreira"])}
        </p>

        <p>
            <strong>Fotoproteção:</strong>
            {html.escape(dermocosmeticos["Fotoproteção"])}
        </p>

    </div>

    <div class="secao">

        <h2>5. Acompanhamento</h2>

        <p>
            <strong>Retorno sugerido:</strong>
            {html.escape(prazo_followup)}
        </p>

        <p>
            <strong>Observações profissionais:</strong>
            {observacoes_html}
        </p>

    </div>

    <div class="aviso">

        <strong>AVISO IMPORTANTE</strong>

        <p>
            Este documento apresenta indicadores experimentais
            de processamento digital de imagem e sugestões educativas.
            Não constitui diagnóstico dermatológico, identificação
            de doenças ou prescrição de tratamento.
        </p>

        <p>
            Os indicadores de bordas e tonalidade sofrem influência
            da iluminação, câmera, posição, maquiagem e características
            da fotografia. A escolha dos procedimentos e produtos
            depende de avaliação presencial e análise de contraindicações.
        </p>

    </div>

    <p style="text-align:center; margin-top:30px;">

        Sistema de Triagem Visual Estética

    </p>

    <div style="text-align:center; margin-top:25px;">

        <button onclick="window.print()">
            IMPRIMIR / SALVAR EM PDF
        </button>

    </div>

    </body>
    </html>
    """

    return documento


# ==============================================================================
# 7. FORMULÁRIO E CAPTURA DA IMAGEM
# ==============================================================================

st.divider()

st.header("1. Fotografia facial")

modo_captura = st.radio(
    "Escolha como deseja enviar a fotografia:",
    (
        "Carregar Foto do Dispositivo",
        "Usar Câmera"
    ),
    horizontal=True
)

if modo_captura == "Usar Câmera":

    foto = st.camera_input(
        "Centralize o rosto, retire os óculos e procure boa iluminação."
    )

else:

    foto = st.file_uploader(
        "Selecione uma fotografia frontal do rosto:",
        type=["jpg", "jpeg", "png"]
    )


# ==============================================================================
# 8. ANAMNESE
# ==============================================================================

with st.form("form_triagem"):

    st.header("2. Dados da avaliação")

    col1, col2 = st.columns(2)

    with col1:

        queixa = st.selectbox(
            "Principal objetivo:",
            (
                "Linhas de expressão",
                "Manchas/Uniformização",
                "Hidratação/Textura"
            )
        )

    with col2:

        tipo_pele = st.selectbox(
            "Como você sente sua pele?",
            (
                "Mista",
                "Oleosa",
                "Seca/Ressecada",
                "Sensível"
            )
        )

    restricao = st.checkbox(
        "Estou gestante, amamentando ou com a pele sensibilizada por ácidos."
    )

    col3, col4 = st.columns(2)

    with col3:

        prazo_followup = st.selectbox(
            "Prazo de retorno:",
            (
                "15 dias (reavaliação inicial)",
                "30 dias (ciclo de renovação)",
                "60 dias (manutenção)",
                "90 dias (acompanhamento)"
            )
        )

    with col4:

        telefone_cliente = st.text_input(
            "WhatsApp do cliente (opcional):",
            placeholder="Ex.: 61981522160"
        )

    observacoes_profissional = st.text_area(
        "Anotações do profissional:",
        placeholder=(
            "Alergias, sensibilidade, histórico de procedimentos, "
            "observações da anamnese..."
        ),
        height=100
    )

    enviado = st.form_submit_button(
        "🔍 ANALISAR IMAGEM E GERAR RELATÓRIO",
        type="primary",
        use_container_width=True
    )


# ==============================================================================
# 9. PROCESSAMENTO PRINCIPAL
# ==============================================================================

if enviado:

    if foto is None:

        st.error(
            "É necessário carregar ou capturar uma fotografia antes da análise."
        )

    else:

        try:

            with st.spinner(
                "Analisando a fotografia e preparando o relatório..."
            ):

                # ------------------------------------------------------------------
                # LEITURA DA IMAGEM
                # ------------------------------------------------------------------

                bytes_imagem = np.asarray(
                    bytearray(foto.getvalue()),
                    dtype=np.uint8
                )

                img = cv2.imdecode(
                    bytes_imagem,
                    cv2.IMREAD_COLOR
                )

                if img is None:

                    raise ValueError(
                        "Imagem inválida ou formato não reconhecido."
                    )

                altura, largura = img.shape[:2]

                if largura < 200 or altura < 200:

                    raise ValueError(
                        "Fotografia com resolução insuficiente. "
                        "Envie uma imagem com pelo menos 200 x 200 pixels."
                    )

                if largura * altura > 25000000:

                    raise ValueError(
                        "Imagem muito grande. Reduza a resolução "
                        "antes de realizar o envio."
                    )

                # ------------------------------------------------------------------
                # DETECÇÃO FACIAL
                # ------------------------------------------------------------------

                rosto = detectar_rosto(img)

                if rosto is None:

                    raise ValueError(
                        "Não foi possível identificar o rosto. "
                        "Envie uma fotografia frontal, com o rosto "
                        "visível, boa iluminação e sem óculos."
                    )

                # ------------------------------------------------------------------
                # DEFINIÇÃO DAS ZONAS
                # ------------------------------------------------------------------

                zonas = definir_zonas(
                    rosto,
                    largura,
                    altura
                )

                img_marcada = img.copy()

                resultados_regioes = {}

                # ------------------------------------------------------------------
                # ANÁLISE DAS REGIÕES
                # ------------------------------------------------------------------

                for nome, (x1, y1, x2, y2) in zonas.items():

                    recorte = img[y1:y2, x1:x2]

                    if recorte.size == 0:

                        raise ValueError(
                            f"Não foi possível analisar a região: {nome}"
                        )

                    indice_linhas, indice_tom = analisar_recorte(
                        recorte
                    )

                    resultados_regioes[nome] = (
                        indice_linhas,
                        indice_tom
                    )

                    # Retângulo de identificação
                    cv2.rectangle(
                        img_marcada,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        img_marcada,
                        nome,
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA
                    )

                # Marcação do rosto detectado
                x, y, rw, rh = rosto

                cv2.rectangle(
                    img_marcada,
                    (x, y),
                    (x + rw, y + rh),
                    (255, 150, 0),
                    2
                )

                # ------------------------------------------------------------------
                # CÁLCULO DOS INDICADORES GERAIS
                # ------------------------------------------------------------------

                linhas_testa = resultados_regioes[
                    "Testa e Glabela"
                ][0]

                valores_bochechas = [

                    resultados_regioes[
                        "Bochecha esquerda da imagem"
                    ][1],

                    resultados_regioes[
                        "Bochecha direita da imagem"
                    ][1]

                ]

                tom_medio = round(
                    float(np.mean(valores_bochechas)),
                    2
                )

                # ------------------------------------------------------------------
                # MOTOR DE RECOMENDAÇÃO
                # ------------------------------------------------------------------

                procedimentos, dermocosmeticos = gerar_recomendacoes(
                    linhas_testa,
                    tom_medio,
                    queixa,
                    tipo_pele,
                    restricao
                )

                # ------------------------------------------------------------------
                # GERAÇÃO DO RELATÓRIO
                # ------------------------------------------------------------------

                relatorio_html = gerar_relatorio_html(
                    img_marcada,
                    resultados_regioes,
                    linhas_testa,
                    tom_medio,
                    queixa,
                    tipo_pele,
                    restricao,
                    procedimentos,
                    dermocosmeticos,
                    observacoes_profissional,
                    prazo_followup
                )

                # ------------------------------------------------------------------
                # LINK WHATSAPP
                # ------------------------------------------------------------------

                link_whatsapp = montar_link_whatsapp(
                    queixa,
                    tipo_pele,
                    restricao,
                    linhas_testa,
                    tom_medio,
                    procedimentos,
                    dermocosmeticos,
                    observacoes_profissional,
                    prazo_followup,
                    telefone_cliente
                )

            # ==========================================================================
            # 10. RESULTADOS NA TELA
            # ==========================================================================

            st.success(
                "Análise concluída e relatório gerado com sucesso!"
            )

            st.divider()

            st.header("3. Resultado da análise facial")

            col_img1, col_img2 = st.columns(2)

            with col_img1:

                st.subheader("Fotografia original")

                st.image(
                    cv2.cvtColor(
                        img,
                        cv2.COLOR_BGR2RGB
                    ),
                    use_container_width=True
                )

            with col_img2:

                st.subheader("Regiões analisadas")

                st.image(
                    cv2.cvtColor(
                        img_marcada,
                        cv2.COLOR_BGR2RGB
                    ),
                    use_container_width=True
                )

            # ------------------------------------------------------------------
            # INDICADORES
            # ------------------------------------------------------------------

            st.subheader("Indicadores visuais experimentais")

            col_a, col_b = st.columns(2)

            with col_a:

                st.metric(
                    "Bordas - Testa",
                    f"{linhas_testa}%"
                )

            with col_b:

                st.metric(
                    "Variação de tonalidade - Bochechas",
                    f"{tom_medio}"
                )

            st.caption(
                "Esses números representam medidas matemáticas da imagem. "
                "Não são escalas clínicas de rugas, manchas ou envelhecimento."
            )

            # ------------------------------------------------------------------
            # TABELA
            # ------------------------------------------------------------------

            st.subheader("Detalhamento por região")

            dados = []

            for nome, valores in resultados_regioes.items():

                dados.append({
                    "Região facial": nome,
                    "Indicador de bordas (%)": valores[0],
                    "Variação de tonalidade": valores[1]
                })

            st.dataframe(
                dados,
                use_container_width=True,
                hide_index=True
            )

            # ==========================================================================
            # 11. RECOMENDAÇÕES
            # ==========================================================================

            st.divider()

            st.header("4. Procedimentos estéticos sugeridos")

            for i, procedimento in enumerate(procedimentos, 1):

                st.markdown(
                    f"**{i}. {procedimento}**"
                )

            if restricao:

                st.warning(
                    "A cliente informou uma condição que exige cautela. "
                    "Não realizar procedimentos ou iniciar ativos sem "
                    "avaliação específica de contraindicações."
                )

            # ==========================================================================
            # 12. DERMOCOSMÉTICOS
            # ==========================================================================

            st.divider()

            st.header("5. Rotina dermocosmética")

            st.markdown(
                f"**Limpeza facial:** {dermocosmeticos['Limpeza']}"
            )

            st.markdown("**Ativos de tratamento:**")

            for ativo in dermocosmeticos["Ativos de Tratamento"]:

                st.markdown(f"- {ativo}")

            st.markdown(
                "**Hidratação e barreira cutânea:** "
                + dermocosmeticos["Hidratação e Barreira"]
            )

            st.markdown(
                "**Fotoproteção:** "
                + dermocosmeticos["Fotoproteção"]
            )

            # ==========================================================================
            # 13. ACOMPANHAMENTO
            # ==========================================================================

            st.divider()

            st.header("6. Acompanhamento")

            st.write(
                f"**Retorno sugerido:** {prazo_followup}"
            )

            if observacoes_profissional.strip():

                st.write(
                    "**Observações profissionais:**"
                )

                st.write(
                    observacoes_profissional
                )

            # ==========================================================================
            # 14. DOWNLOAD DO RELATÓRIO
            # ==========================================================================

            st.divider()

            st.header("7. Gerar relatório")

            st.download_button(
                label="📄 BAIXAR RELATÓRIO COMPLETO",
                data=relatorio_html.encode("utf-8"),
                file_name=(
                    "relatorio_triagem_estetica_"
                    + datetime.now().strftime("%Y%m%d_%H%M%S")
                    + ".html"
                ),
                mime="text/html",
                use_container_width=True
            )

            st.info(
                "O relatório inclui a fotografia marcada, os indicadores, "
                "as recomendações e as informações da avaliação. "
                "Abra o arquivo baixado no navegador e utilize "
                "'IMPRIMIR / SALVAR EM PDF'."
            )

            # ==========================================================================
            # 15. WHATSAPP
            # ==========================================================================

            st.divider()

            st.header("8. Compartilhar pelo WhatsApp")

            st.link_button(
                "📲 ENVIAR RESUMO DA AVALIAÇÃO",
                link_whatsapp,
                use_container_width=True
            )

            st.caption(
                "O WhatsApp abrirá com o resumo preenchido. "
                "O envio precisa ser confirmado pelo usuário."
            )

            # ==========================================================================
            # 16. AVISO FINAL
            # ==========================================================================

            st.divider()

            st.success(
                "Processamento finalizado. Relatório disponível para download."
            )

            st.caption(
                "A triagem é educativa e experimental. Não substitui "
                "avaliação dermatológica, diagnóstico ou prescrição."
            )

        except Exception as erro:

            st.error(
                "Não foi possível concluir a análise."
            )

            st.warning(
                str(erro)
            )

            st.exception(erro)


# ==============================================================================
# FIM DO SISTEMA
# ==============================================================================
