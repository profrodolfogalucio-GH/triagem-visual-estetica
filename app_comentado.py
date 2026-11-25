# ==============================================================================
# VERSÃO DIDÁTICA — TRIAGEM VISUAL ESTÉTICA
# ==============================================================================
# Este arquivo é uma cópia explicada do app.py. Os comentários foram acrescentados
# para facilitar o estudo. As instruções executáveis do código original são
# preservadas; este arquivo não substitui o app.py usado na publicação.
# Os indicadores calculados são experimentais e não constituem diagnóstico.
# ==============================================================================


# Explicação: Importa o OpenCV, utilizado para leitura, conversão, análise e marcação de imagens.
import cv2
# Explicação: Importa o NumPy com o apelido np para operações numéricas e manipulação de matrizes.
import numpy as np
# Explicação: Importa o Streamlit com o apelido st para construir a interface web.
import streamlit as st

# Explicação: Importa ferramentas para codificar a imagem e incorporá-la no relatório HTML.
import base64
# Explicação: Importa funções para tratar textos inseridos no HTML e evitar que sejam interpretados como marcação.
import html
# Explicação: Importa expressões regulares para limpar e validar o número de telefone.
import re

# Explicação: Importa um recurso específico de um módulo da biblioteca padrão do Python.
from datetime import datetime
# Explicação: Importa um recurso específico de um módulo da biblioteca padrão do Python.
from urllib.parse import quote


# ==============================================================================
# CONFIGURAÇÕES DA APLICAÇÃO
# ==============================================================================

# Explicação: Configura a página do Streamlit: título da aba, ícone e disposição ampla do conteúdo.
st.set_page_config(
    page_title="Triagem Visual Estética",
    page_icon="✨",
    layout="wide"
)

# Explicação: Exibe o título principal da aplicação na interface.
st.title("✨ Triagem Visual Estética")
# Explicação: Exibe uma legenda explicativa em tamanho reduzido na interface.
st.caption(
    "Sistema de análise visual preliminar, anamnese e recomendações estéticas."
)


# ==============================================================================
# 1. DETECÇÃO FACIAL
# ==============================================================================

# Explicação: Aplica um decorador do Streamlit para manter o recurso carregado em cache, evitando recriar o detector a cada interação.
@st.cache_resource
# Explicação: Define a função responsável por localizar e carregar o classificador Haar do OpenCV usado na detecção facial.
def carregar_detector():
    # Explicação: Atribui o resultado da expressão à variável ou chave caminho; essa informação será reutilizada nesta etapa do programa.
    caminho = cv2.data.haarcascades + (
        "haarcascade_frontalface_default.xml"
    )

    # Explicação: Atribui o resultado da expressão à variável ou chave detector; essa informação será reutilizada nesta etapa do programa.
    detector = cv2.CascadeClassifier(caminho)

    # Explicação: Verifica uma condição. O bloco indentado abaixo só será executado quando a expressão for verdadeira.
    if detector.empty():
        # Explicação: Gera uma exceção para sinalizar uma condição inválida ou impedir que o processamento continue incorretamente.
        raise RuntimeError(
            "Não foi possível carregar o classificador facial do OpenCV."
        )

    # Explicação: Encerra a função neste ponto e devolve o valor indicado para quem a chamou.
    return detector


# Explicação: Define a função que recebe uma imagem, converte-a para tons de cinza e procura rostos nela.
def detectar_rosto(imagem):

    # Explicação: Atribui o resultado da expressão à variável ou chave detector; essa informação será reutilizada nesta etapa do programa.
    detector = carregar_detector()

    # Explicação: Atribui o resultado da expressão à variável ou chave cinza; essa informação será reutilizada nesta etapa do programa.
    cinza = cv2.cvtColor(
        imagem,
        cv2.COLOR_BGR2GRAY
    )

    # Explicação: Atribui o resultado da expressão à variável ou chave rostos; essa informação será reutilizada nesta etapa do programa.
    rostos = detector.detectMultiScale(
        cinza,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )

    # Explicação: Verifica uma condição. O bloco indentado abaixo só será executado quando a expressão for verdadeira.
    if len(rostos) == 0:
        # Explicação: Encerra a função neste ponto e devolve o valor indicado para quem a chamou.
        return None

    # Seleciona o maior rosto detectado
    # Explicação: Atribui o resultado da expressão à variável ou chave rosto; essa informação será reutilizada nesta etapa do programa.
    rosto = max(
        rostos,
        key=lambda r: r[2] * r[3]
    )

    # Explicação: Encerra a função neste ponto e devolve o valor indicado para quem a chamou.
    return tuple(map(int, rosto))


# ==============================================================================
# 2. DEFINIÇÃO DAS REGIÕES FACIAIS
# ==============================================================================

# Explicação: Define a função que calcula as coordenadas das regiões faciais a partir do retângulo do rosto.
def definir_zonas(rosto, largura, altura):

    # Explicação: Atribui o resultado da expressão à variável ou chave x, y, rw, rh; essa informação será reutilizada nesta etapa do programa.
    x, y, rw, rh = rosto

    # Explicação: Define uma função interna que transforma proporções relativas em coordenadas de pixels limitadas à imagem.
    def coordenadas(x1, y1, x2, y2):

        # Explicação: Encerra a função neste ponto e devolve o valor indicado para quem a chamou.
        return (
            max(0, min(int(x + rw * x1), largura)),
            max(0, min(int(y + rh * y1), altura)),
            max(0, min(int(x + rw * x2), largura)),
            max(0, min(int(y + rh * y2), altura))
        )

    # Explicação: Atribui o resultado da expressão à variável ou chave zonas; essa informação será reutilizada nesta etapa do programa.
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

    # Explicação: Encerra a função neste ponto e devolve o valor indicado para quem a chamou.
    return zonas


# ==============================================================================
# 3. ANÁLISE DOS RECORTES
# ==============================================================================

# Explicação: Define a função que calcula indicadores matemáticos de bordas e variação de tonalidade em um recorte.
def analisar_recorte(recorte):

    # Explicação: Verifica uma condição. O bloco indentado abaixo só será executado quando a expressão for verdadeira.
    if recorte is None or recorte.size == 0:
        # Explicação: Gera uma exceção para sinalizar uma condição inválida ou impedir que o processamento continue incorretamente.
        raise ValueError("Região facial vazia.")

    # Explicação: Atribui o resultado da expressão à variável ou chave cinza; essa informação será reutilizada nesta etapa do programa.
    cinza = cv2.cvtColor(
        recorte,
        cv2.COLOR_BGR2GRAY
    )

    # Explicação: Atribui o resultado da expressão à variável ou chave cinza_suave; essa informação será reutilizada nesta etapa do programa.
    cinza_suave = cv2.GaussianBlur(
        cinza,
        (3, 3),
        0
    )

    # Explicação: Atribui o resultado da expressão à variável ou chave bordas; essa informação será reutilizada nesta etapa do programa.
    bordas = cv2.Canny(
        cinza_suave,
        25,
        80
    )

    # Explicação: Atribui o resultado da expressão à variável ou chave indice_linhas; essa informação será reutilizada nesta etapa do programa.
    indice_linhas = round(
        float(np.mean(bordas > 0) * 100),
        2
    )

    # Explicação: Atribui o resultado da expressão à variável ou chave lab; essa informação será reutilizada nesta etapa do programa.
    lab = cv2.cvtColor(
        recorte,
        cv2.COLOR_BGR2LAB
    )

    # Explicação: Atribui o resultado da expressão à variável ou chave _, canal_a, _; essa informação será reutilizada nesta etapa do programa.
    _, canal_a, _ = cv2.split(lab)

    # Explicação: Atribui o resultado da expressão à variável ou chave indice_tom; essa informação será reutilizada nesta etapa do programa.
    indice_tom = round(
        float(np.std(canal_a)),
        2
    )

    # Explicação: Encerra a função neste ponto e devolve o valor indicado para quem a chamou.
    return indice_linhas, indice_tom


# ==============================================================================
# 4. MOTOR DE RECOMENDAÇÕES
# ==============================================================================

# Explicação: Define o motor que combina queixa, tipo de pele e restrições informadas para montar sugestões educativas.
def gerar_recomendacoes(
    linhas_testa,
    tom_medio,
    queixa,
    tipo_pele,
    restricao
):

    # Explicação: Atribui o resultado da expressão à variável ou chave procedimentos; essa informação será reutilizada nesta etapa do programa.
    procedimentos = []

    # Explicação: Atribui o resultado da expressão à variável ou chave dermocosmeticos; essa informação será reutilizada nesta etapa do programa.
    dermocosmeticos = {
        "Limpeza": "",
        "Ativos de Tratamento": [],
        "Hidratação e Barreira": "",
        "Fotoproteção": ""
    }

    # --------------------------------------------------------------------------
    # PROCEDIMENTOS
    # --------------------------------------------------------------------------

    # Explicação: Verifica uma condição. O bloco indentado abaixo só será executado quando a expressão for verdadeira.
    if restricao:

        # Explicação: Executa esta instrução dentro da etapa atual do sistema; os valores e parâmetros são utilizados conforme o contexto descrito na seção.
        procedimentos.append(
            "Avaliação de hidratação e recuperação da barreira cutânea, "
            "com protocolos suaves e compatíveis com a condição individual."
        )

        # Explicação: Executa esta instrução dentro da etapa atual do sistema; os valores e parâmetros são utilizados conforme o contexto descrito na seção.
        procedimentos.append(
            "Reavaliação profissional antes de realizar peelings, "
            "microagulhamento, procedimentos injetáveis ou tratamentos "
            "com ativos potencialmente irritantes."
        )

    # Explicação: Inicia o caminho alternativo, executado quando as condições anteriores não forem atendidas.
    else:

        # Explicação: Verifica uma condição. O bloco indentado abaixo só será executado quando a expressão for verdadeira.
        if queixa == "Linhas de expressão":

            # Explicação: Executa esta instrução dentro da etapa atual do sistema; os valores e parâmetros são utilizados conforme o contexto descrito na seção.
            procedimentos.append(
                "Avaliar protocolo de revitalização facial e hidratação. "
                "A possibilidade de toxina botulínica, skinbooster ou "
                "microagulhamento depende de avaliação presencial, "
                "indicação e habilitação profissional."
            )

        # Explicação: Verifica uma condição alternativa caso as condições anteriores não tenham sido satisfeitas.
        elif queixa == "Manchas/Uniformização":

            # Explicação: Executa esta instrução dentro da etapa atual do sistema; os valores e parâmetros são utilizados conforme o contexto descrito na seção.
            procedimentos.append(
                "Avaliar protocolo de uniformização do tom, considerando "
                "fototipo, histórico de exposição solar e características "
                "das alterações pigmentares."
            )

            # Explicação: Executa esta instrução dentro da etapa atual do sistema; os valores e parâmetros são utilizados conforme o contexto descrito na seção.
            procedimentos.append(
                "Peelings superficiais, microdermoabrasão ou tecnologias "
                "de luz somente após avaliação individual e verificação "
                "de contraindicações."
            )

        # Explicação: Verifica uma condição alternativa caso as condições anteriores não tenham sido satisfeitas.
        elif queixa == "Hidratação/Textura":

            # Explicação: Executa esta instrução dentro da etapa atual do sistema; os valores e parâmetros são utilizados conforme o contexto descrito na seção.
            procedimentos.append(
                "Avaliar hidratação facial, revitalização e recuperação "
                "da barreira cutânea."
            )

            # Explicação: Executa esta instrução dentro da etapa atual do sistema; os valores e parâmetros são utilizados conforme o contexto descrito na seção.
            procedimentos.append(
                "Outros recursos, como microagulhamento ou tecnologias "
                "de estímulo de colágeno, dependem de indicação e "
                "avaliação profissional."
            )

    # --------------------------------------------------------------------------
    # LIMPEZA
    # --------------------------------------------------------------------------

    # Explicação: Verifica uma condição. O bloco indentado abaixo só será executado quando a expressão for verdadeira.
    if tipo_pele == "Oleosa":

        # Explicação: Atribui o resultado da expressão à variável ou chave dermocosmeticos["Limpeza"]; essa informação será reutilizada nesta etapa do programa.
        dermocosmeticos["Limpeza"] = (
            "Gel de limpeza suave, adequado à pele oleosa, "
            "sem provocar ressecamento excessivo."
        )

    # Explicação: Verifica uma condição alternativa caso as condições anteriores não tenham sido satisfeitas.
    elif tipo_pele == "Mista":

        # Explicação: Atribui o resultado da expressão à variável ou chave dermocosmeticos["Limpeza"]; essa informação será reutilizada nesta etapa do programa.
        dermocosmeticos["Limpeza"] = (
            "Gel ou espuma de limpeza equilibrante, "
            "adequado à pele mista."
        )

    # Explicação: Verifica uma condição alternativa caso as condições anteriores não tenham sido satisfeitas.
    elif tipo_pele == "Seca/Ressecada":

        # Explicação: Atribui o resultado da expressão à variável ou chave dermocosmeticos["Limpeza"]; essa informação será reutilizada nesta etapa do programa.
        dermocosmeticos["Limpeza"] = (
            "Loção ou creme de limpeza suave, "
            "com componentes hidratantes."
        )

    # Explicação: Inicia o caminho alternativo, executado quando as condições anteriores não forem atendidas.
    else:

        # Explicação: Atribui o resultado da expressão à variável ou chave dermocosmeticos["Limpeza"]; essa informação será reutilizada nesta etapa do programa.
        dermocosmeticos["Limpeza"] = (
            "Produto de limpeza suave, sem fragrância, "
            "adequado à pele sensível."
        )

    # --------------------------------------------------------------------------
    # ATIVOS DE TRATAMENTO
    # --------------------------------------------------------------------------

    # Explicação: Verifica uma condição. O bloco indentado abaixo só será executado quando a expressão for verdadeira.
    if restricao:

        # Explicação: Executa esta instrução dentro da etapa atual do sistema; os valores e parâmetros são utilizados conforme o contexto descrito na seção.
        dermocosmeticos["Ativos de Tratamento"].append(
            "Não indicar automaticamente ácidos, retinoides ou "
            "ativos clareadores. A escolha deve considerar gestação, "
            "amamentação, sensibilização e avaliação profissional."
        )

    # Explicação: Inicia o caminho alternativo, executado quando as condições anteriores não forem atendidas.
    else:

        # Explicação: Verifica uma condição. O bloco indentado abaixo só será executado quando a expressão for verdadeira.
        if queixa == "Linhas de expressão":

            # Explicação: Executa esta instrução dentro da etapa atual do sistema; os valores e parâmetros são utilizados conforme o contexto descrito na seção.
            dermocosmeticos["Ativos de Tratamento"].append(
                "Peptídeos e ácido hialurônico em formulações tópicas, "
                "considerando tolerância e necessidade individual."
            )

        # Explicação: Verifica uma condição alternativa caso as condições anteriores não tenham sido satisfeitas.
        elif queixa == "Manchas/Uniformização":

            # Explicação: Executa esta instrução dentro da etapa atual do sistema; os valores e parâmetros são utilizados conforme o contexto descrito na seção.
            dermocosmeticos["Ativos de Tratamento"].append(
                "Niacinamida ou vitamina C em formulações adequadas "
                "ao tipo de pele, com avaliação da tolerância."
            )

        # Explicação: Verifica uma condição alternativa caso as condições anteriores não tenham sido satisfeitas.
        elif queixa == "Hidratação/Textura":

            # Explicação: Executa esta instrução dentro da etapa atual do sistema; os valores e parâmetros são utilizados conforme o contexto descrito na seção.
            dermocosmeticos["Ativos de Tratamento"].append(
                "Ácido hialurônico, pantenol e outros agentes "
                "hidratantes para auxiliar na manutenção da pele."
            )

    # --------------------------------------------------------------------------
    # HIDRATAÇÃO
    # --------------------------------------------------------------------------

    # Explicação: Verifica uma condição. O bloco indentado abaixo só será executado quando a expressão for verdadeira.
    if tipo_pele in ("Oleosa", "Mista"):

        # Explicação: Atribui o resultado da expressão à variável ou chave dermocosmeticos["Hidratação e Barreira"]; essa informação será reutilizada nesta etapa do programa.
        dermocosmeticos["Hidratação e Barreira"] = (
            "Sérum ou gel-creme não comedogênico, "
            "com ácido hialurônico e pantenol."
        )

    # Explicação: Verifica uma condição alternativa caso as condições anteriores não tenham sido satisfeitas.
    elif tipo_pele == "Sensível":

        # Explicação: Atribui o resultado da expressão à variável ou chave dermocosmeticos["Hidratação e Barreira"]; essa informação será reutilizada nesta etapa do programa.
        dermocosmeticos["Hidratação e Barreira"] = (
            "Hidratante reparador sem fragrância, "
            "com componentes calmantes e restauradores da barreira."
        )

    # Explicação: Inicia o caminho alternativo, executado quando as condições anteriores não forem atendidas.
    else:

        # Explicação: Atribui o resultado da expressão à variável ou chave dermocosmeticos["Hidratação e Barreira"]; essa informação será reutilizada nesta etapa do programa.
        dermocosmeticos["Hidratação e Barreira"] = (
            "Creme hidratante com ceramidas, glicerina "
            "ou outros agentes de reparação da barreira cutânea."
        )

    # --------------------------------------------------------------------------
    # FOTOPROTEÇÃO
    # --------------------------------------------------------------------------

    # Explicação: Verifica uma condição. O bloco indentado abaixo só será executado quando a expressão for verdadeira.
    if queixa == "Manchas/Uniformização":

        # Explicação: Atribui o resultado da expressão à variável ou chave dermocosmeticos["Fotoproteção"]; essa informação será reutilizada nesta etapa do programa.
        dermocosmeticos["Fotoproteção"] = (
            "Protetor solar de amplo espectro, FPS 50 ou superior. "
            "Considerar versão com cor para auxiliar na proteção "
            "contra a luz visível."
        )

    # Explicação: Inicia o caminho alternativo, executado quando as condições anteriores não forem atendidas.
    else:

        # Explicação: Atribui o resultado da expressão à variável ou chave dermocosmeticos["Fotoproteção"]; essa informação será reutilizada nesta etapa do programa.
        dermocosmeticos["Fotoproteção"] = (
            "Protetor solar de amplo espectro, FPS 50 ou superior, "
            "com reaplicação conforme exposição."
        )

    # Explicação: Encerra a função neste ponto e devolve o valor indicado para quem a chamou.
    return procedimentos, dermocosmeticos


# ==============================================================================
# 5. MONTAGEM DA MENSAGEM DO WHATSAPP
# ==============================================================================

# Explicação: Define a função que monta o texto-resumo e gera um endereço do WhatsApp com a mensagem codificada.
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

    # Explicação: Atribui o resultado da expressão à variável ou chave lista_proc; essa informação será reutilizada nesta etapa do programa.
    lista_proc = "\n".join(
        f"- {p}" for p in procedimentos
    )

    # Explicação: Atribui o resultado da expressão à variável ou chave lista_ativos; essa informação será reutilizada nesta etapa do programa.
    lista_ativos = "\n".join(
        f"- {a}"
        for a in dermocosmeticos["Ativos de Tratamento"]
    )

    # Explicação: Atribui o resultado da expressão à variável ou chave mensagem; essa informação será reutilizada nesta etapa do programa.
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

    # Explicação: Verifica uma condição. O bloco indentado abaixo só será executado quando a expressão for verdadeira.
    if observacoes.strip():

        # Explicação: Atribui o resultado da expressão à variável ou chave mensagem +; essa informação será reutilizada nesta etapa do programa.
        mensagem += (
            "\n*Observações profissionais:*\n"
            + observacoes.strip()
        )

    # Explicação: Atribui o resultado da expressão à variável ou chave mensagem +; essa informação será reutilizada nesta etapa do programa.
    mensagem += """

Aviso: Triagem visual educativa. Os indicadores não representam
diagnóstico dermatológico. A conduta final depende de avaliação presencial.
"""

    # Explicação: Atribui o resultado da expressão à variável ou chave numero; essa informação será reutilizada nesta etapa do programa.
    numero = re.sub(
        r"\D",
        "",
        telefone or ""
    )

    # Explicação: Verifica uma condição. O bloco indentado abaixo só será executado quando a expressão for verdadeira.
    if len(numero) in (10, 11):
        # Explicação: Atribui o resultado da expressão à variável ou chave numero; essa informação será reutilizada nesta etapa do programa.
        numero = "55" + numero

    # Explicação: Verifica uma condição. O bloco indentado abaixo só será executado quando a expressão for verdadeira.
    if len(numero) not in (0, 12, 13):
        # Explicação: Atribui o resultado da expressão à variável ou chave numero; essa informação será reutilizada nesta etapa do programa.
        numero = ""

    # Explicação: Verifica uma condição. O bloco indentado abaixo só será executado quando a expressão for verdadeira.
    if numero and not numero.startswith("55"):
        # Explicação: Atribui o resultado da expressão à variável ou chave numero; essa informação será reutilizada nesta etapa do programa.
        numero = ""

    # Explicação: Atribui o resultado da expressão à variável ou chave texto_url; essa informação será reutilizada nesta etapa do programa.
    texto_url = quote(
        mensagem,
        safe=""
    )

    # Explicação: Verifica uma condição. O bloco indentado abaixo só será executado quando a expressão for verdadeira.
    if numero:
        # Explicação: Encerra a função neste ponto e devolve o valor indicado para quem a chamou.
        return f"https://wa.me/{numero}?text={texto_url}"

    # Explicação: Encerra a função neste ponto e devolve o valor indicado para quem a chamou.
    return f"https://wa.me/?text={texto_url}"


# ==============================================================================
# 6. GERAÇÃO DO RELATÓRIO HTML
# ==============================================================================

# Explicação: Define a função que monta o relatório HTML com imagem, indicadores, recomendações e observações.
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

    # Explicação: Atribui o resultado da expressão à variável ou chave sucesso, buffer_img; essa informação será reutilizada nesta etapa do programa.
    sucesso, buffer_img = cv2.imencode(
        ".jpg",
        imagem_marcada,
        [cv2.IMWRITE_JPEG_QUALITY, 90]
    )

    # Explicação: Verifica uma condição. O bloco indentado abaixo só será executado quando a expressão for verdadeira.
    if not sucesso:
        # Explicação: Gera uma exceção para sinalizar uma condição inválida ou impedir que o processamento continue incorretamente.
        raise RuntimeError(
            "Não foi possível preparar a imagem do relatório."
        )

    # Explicação: Atribui o resultado da expressão à variável ou chave imagem_base64; essa informação será reutilizada nesta etapa do programa.
    imagem_base64 = base64.b64encode(
        buffer_img.tobytes()
    ).decode("utf-8")

    # Explicação: Atribui o resultado da expressão à variável ou chave data; essa informação será reutilizada nesta etapa do programa.
    data = datetime.now().strftime(
        "%d/%m/%Y às %H:%M"
    )

    # Explicação: Atribui o resultado da expressão à variável ou chave linhas_html; essa informação será reutilizada nesta etapa do programa.
    linhas_html = ""

    # Explicação: Inicia um laço que percorre os elementos da sequência indicada, executando o bloco para cada elemento.
    for nome, valores in resultados_regioes.items():

        # Explicação: Atribui o resultado da expressão à variável ou chave linhas_html +; essa informação será reutilizada nesta etapa do programa.
        linhas_html += f"""
        <tr>
            <td>{html.escape(nome)}</td>
            <td>{valores[0]}%</td>
            <td>{valores[1]}</td>
        </tr>
        """

    # Explicação: Atribui o resultado da expressão à variável ou chave procedimentos_html; essa informação será reutilizada nesta etapa do programa.
    procedimentos_html = "".join(
        f"<li>{html.escape(p)}</li>"
        for p in procedimentos
    )

    # Explicação: Atribui o resultado da expressão à variável ou chave ativos_html; essa informação será reutilizada nesta etapa do programa.
    ativos_html = "".join(
        f"<li>{html.escape(a)}</li>"
        for a in dermocosmeticos["Ativos de Tratamento"]
    )

    # Explicação: Atribui o resultado da expressão à variável ou chave observacoes_html; essa informação será reutilizada nesta etapa do programa.
    observacoes_html = (
        html.escape(observacoes.strip())
        if observacoes.strip()
        else "Nenhuma observação informada."
    )

    # Explicação: Atribui o resultado da expressão à variável ou chave restricao_texto; essa informação será reutilizada nesta etapa do programa.
    restricao_texto = (
        "Sim - requer avaliação individual"
        if restricao
        else "Não informada"
    )

    # Explicação: Atribui o resultado da expressão à variável ou chave documento; essa informação será reutilizada nesta etapa do programa.
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

    # Explicação: Encerra a função neste ponto e devolve o valor indicado para quem a chamou.
    return documento


# ==============================================================================
# 7. FORMULÁRIO E CAPTURA DA IMAGEM
# ==============================================================================

# Explicação: Insere uma linha divisória para separar visualmente as seções.
st.divider()

# Explicação: Exibe o cabeçalho da seção correspondente na interface.
st.header("1. Fotografia facial")

# Explicação: Atribui o resultado da expressão à variável ou chave modo_captura; essa informação será reutilizada nesta etapa do programa.
modo_captura = st.radio(
    "Escolha como deseja enviar a fotografia:",
    (
        "Carregar Foto do Dispositivo",
        "Usar Câmera"
    ),
    horizontal=True
)

# Explicação: Verifica uma condição. O bloco indentado abaixo só será executado quando a expressão for verdadeira.
if modo_captura == "Usar Câmera":

    # Explicação: Atribui o resultado da expressão à variável ou chave foto; essa informação será reutilizada nesta etapa do programa.
    foto = st.camera_input(
        "Centralize o rosto, retire os óculos e procure boa iluminação."
    )

# Explicação: Inicia o caminho alternativo, executado quando as condições anteriores não forem atendidas.
else:

    # Explicação: Atribui o resultado da expressão à variável ou chave foto; essa informação será reutilizada nesta etapa do programa.
    foto = st.file_uploader(
        "Selecione uma fotografia frontal do rosto:",
        type=["jpg", "jpeg", "png"]
    )


# ==============================================================================
# 8. ANAMNESE
# ==============================================================================

# Explicação: Abre um contexto de execução gerenciado; o bloco indentado utiliza o recurso ou componente indicado.
with st.form("form_triagem"):

    # Explicação: Exibe o cabeçalho da seção correspondente na interface.
    st.header("2. Dados da avaliação")

    # Explicação: Atribui o resultado da expressão à variável ou chave col1, col2; essa informação será reutilizada nesta etapa do programa.
    col1, col2 = st.columns(2)

    # Explicação: Abre um contexto de execução gerenciado; o bloco indentado utiliza o recurso ou componente indicado.
    with col1:

        # Explicação: Atribui o resultado da expressão à variável ou chave queixa; essa informação será reutilizada nesta etapa do programa.
        queixa = st.selectbox(
            "Principal objetivo:",
            (
                "Linhas de expressão",
                "Manchas/Uniformização",
                "Hidratação/Textura"
            )
        )

    # Explicação: Abre um contexto de execução gerenciado; o bloco indentado utiliza o recurso ou componente indicado.
    with col2:

        # Explicação: Atribui o resultado da expressão à variável ou chave tipo_pele; essa informação será reutilizada nesta etapa do programa.
        tipo_pele = st.selectbox(
            "Como você sente sua pele?",
            (
                "Mista",
                "Oleosa",
                "Seca/Ressecada",
                "Sensível"
            )
        )

    # Explicação: Atribui o resultado da expressão à variável ou chave restricao; essa informação será reutilizada nesta etapa do programa.
    restricao = st.checkbox(
        "Estou gestante, amamentando ou com a pele sensibilizada por ácidos."
    )

    # Explicação: Atribui o resultado da expressão à variável ou chave col3, col4; essa informação será reutilizada nesta etapa do programa.
    col3, col4 = st.columns(2)

    # Explicação: Abre um contexto de execução gerenciado; o bloco indentado utiliza o recurso ou componente indicado.
    with col3:

        # Explicação: Atribui o resultado da expressão à variável ou chave prazo_followup; essa informação será reutilizada nesta etapa do programa.
        prazo_followup = st.selectbox(
            "Prazo de retorno:",
            (
                "15 dias (reavaliação inicial)",
                "30 dias (ciclo de renovação)",
                "60 dias (manutenção)",
                "90 dias (acompanhamento)"
            )
        )

    # Explicação: Abre um contexto de execução gerenciado; o bloco indentado utiliza o recurso ou componente indicado.
    with col4:

        # Explicação: Atribui o resultado da expressão à variável ou chave telefone_cliente; essa informação será reutilizada nesta etapa do programa.
        telefone_cliente = st.text_input(
            "WhatsApp do cliente (opcional):",
            placeholder="Ex.: 61981522160"
        )

    # Explicação: Atribui o resultado da expressão à variável ou chave observacoes_profissional; essa informação será reutilizada nesta etapa do programa.
    observacoes_profissional = st.text_area(
        "Anotações do profissional:",
        placeholder=(
            "Alergias, sensibilidade, histórico de procedimentos, "
            "observações da anamnese..."
        ),
        height=100
    )

    # Explicação: Atribui o resultado da expressão à variável ou chave enviado; essa informação será reutilizada nesta etapa do programa.
    enviado = st.form_submit_button(
        "🔍 ANALISAR IMAGEM E GERAR RELATÓRIO",
        type="primary",
        use_container_width=True
    )


# ==============================================================================
# 9. PROCESSAMENTO PRINCIPAL
# ==============================================================================

# Explicação: Verifica uma condição. O bloco indentado abaixo só será executado quando a expressão for verdadeira.
if enviado:

    # Explicação: Verifica uma condição. O bloco indentado abaixo só será executado quando a expressão for verdadeira.
    if foto is None:

        # Explicação: Apresenta uma mensagem de erro na interface.
        st.error(
            "É necessário carregar ou capturar uma fotografia antes da análise."
        )

    # Explicação: Inicia o caminho alternativo, executado quando as condições anteriores não forem atendidas.
    else:

        # Explicação: Inicia um bloco protegido: erros ocorridos nele poderão ser tratados pelo bloco except correspondente.
        try:

            # Explicação: Abre um contexto de execução gerenciado; o bloco indentado utiliza o recurso ou componente indicado.
            with st.spinner(
                "Analisando a fotografia e preparando o relatório..."
            ):

                # ------------------------------------------------------------------
                # LEITURA DA IMAGEM
                # ------------------------------------------------------------------

                # Explicação: Atribui o resultado da expressão à variável ou chave bytes_imagem; essa informação será reutilizada nesta etapa do programa.
                bytes_imagem = np.asarray(
                    bytearray(foto.getvalue()),
                    dtype=np.uint8
                )

                # Explicação: Atribui o resultado da expressão à variável ou chave img; essa informação será reutilizada nesta etapa do programa.
                img = cv2.imdecode(
                    bytes_imagem,
                    cv2.IMREAD_COLOR
                )

                # Explicação: Verifica uma condição. O bloco indentado abaixo só será executado quando a expressão for verdadeira.
                if img is None:

                    # Explicação: Gera uma exceção para sinalizar uma condição inválida ou impedir que o processamento continue incorretamente.
                    raise ValueError(
                        "Imagem inválida ou formato não reconhecido."
                    )

                # Explicação: Atribui o resultado da expressão à variável ou chave altura, largura; essa informação será reutilizada nesta etapa do programa.
                altura, largura = img.shape[:2]

                # Explicação: Verifica uma condição. O bloco indentado abaixo só será executado quando a expressão for verdadeira.
                if largura < 200 or altura < 200:

                    # Explicação: Gera uma exceção para sinalizar uma condição inválida ou impedir que o processamento continue incorretamente.
                    raise ValueError(
                        "Fotografia com resolução insuficiente. "
                        "Envie uma imagem com pelo menos 200 x 200 pixels."
                    )

                # Explicação: Verifica uma condição. O bloco indentado abaixo só será executado quando a expressão for verdadeira.
                if largura * altura > 25000000:

                    # Explicação: Gera uma exceção para sinalizar uma condição inválida ou impedir que o processamento continue incorretamente.
                    raise ValueError(
                        "Imagem muito grande. Reduza a resolução "
                        "antes de realizar o envio."
                    )

                # ------------------------------------------------------------------
                # DETECÇÃO FACIAL
                # ------------------------------------------------------------------

                # Explicação: Atribui o resultado da expressão à variável ou chave rosto; essa informação será reutilizada nesta etapa do programa.
                rosto = detectar_rosto(img)

                # Explicação: Verifica uma condição. O bloco indentado abaixo só será executado quando a expressão for verdadeira.
                if rosto is None:

                    # Explicação: Gera uma exceção para sinalizar uma condição inválida ou impedir que o processamento continue incorretamente.
                    raise ValueError(
                        "Não foi possível identificar o rosto. "
                        "Envie uma fotografia frontal, com o rosto "
                        "visível, boa iluminação e sem óculos."
                    )

                # ------------------------------------------------------------------
                # DEFINIÇÃO DAS ZONAS
                # ------------------------------------------------------------------

                # Explicação: Atribui o resultado da expressão à variável ou chave zonas; essa informação será reutilizada nesta etapa do programa.
                zonas = definir_zonas(
                    rosto,
                    largura,
                    altura
                )

                # Explicação: Atribui o resultado da expressão à variável ou chave img_marcada; essa informação será reutilizada nesta etapa do programa.
                img_marcada = img.copy()

                # Explicação: Atribui o resultado da expressão à variável ou chave resultados_regioes; essa informação será reutilizada nesta etapa do programa.
                resultados_regioes = {}

                # ------------------------------------------------------------------
                # ANÁLISE DAS REGIÕES
                # ------------------------------------------------------------------

                # Explicação: Inicia um laço que percorre os elementos da sequência indicada, executando o bloco para cada elemento.
                for nome, (x1, y1, x2, y2) in zonas.items():

                    # Explicação: Atribui o resultado da expressão à variável ou chave recorte; essa informação será reutilizada nesta etapa do programa.
                    recorte = img[y1:y2, x1:x2]

                    # Explicação: Verifica uma condição. O bloco indentado abaixo só será executado quando a expressão for verdadeira.
                    if recorte.size == 0:

                        # Explicação: Gera uma exceção para sinalizar uma condição inválida ou impedir que o processamento continue incorretamente.
                        raise ValueError(
                            f"Não foi possível analisar a região: {nome}"
                        )

                    # Explicação: Atribui o resultado da expressão à variável ou chave indice_linhas, indice_tom; essa informação será reutilizada nesta etapa do programa.
                    indice_linhas, indice_tom = analisar_recorte(
                        recorte
                    )

                    # Explicação: Atribui o resultado da expressão à variável ou chave resultados_regioes[nome]; essa informação será reutilizada nesta etapa do programa.
                    resultados_regioes[nome] = (
                        indice_linhas,
                        indice_tom
                    )

                    # Retângulo de identificação
                    # Explicação: Executa uma operação da biblioteca indicada; os argumentos configuram os dados de entrada e o comportamento dessa operação.
                    cv2.rectangle(
                        img_marcada,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    # Explicação: Executa uma operação da biblioteca indicada; os argumentos configuram os dados de entrada e o comportamento dessa operação.
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
                # Explicação: Atribui o resultado da expressão à variável ou chave x, y, rw, rh; essa informação será reutilizada nesta etapa do programa.
                x, y, rw, rh = rosto

                # Explicação: Executa uma operação da biblioteca indicada; os argumentos configuram os dados de entrada e o comportamento dessa operação.
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

                # Explicação: Atribui o resultado da expressão à variável ou chave linhas_testa; essa informação será reutilizada nesta etapa do programa.
                linhas_testa = resultados_regioes[
                    "Testa e Glabela"
                ][0]

                # Explicação: Atribui o resultado da expressão à variável ou chave valores_bochechas; essa informação será reutilizada nesta etapa do programa.
                valores_bochechas = [

                    resultados_regioes[
                        "Bochecha esquerda da imagem"
                    ][1],

                    resultados_regioes[
                        "Bochecha direita da imagem"
                    ][1]

                ]

                # Explicação: Atribui o resultado da expressão à variável ou chave tom_medio; essa informação será reutilizada nesta etapa do programa.
                tom_medio = round(
                    float(np.mean(valores_bochechas)),
                    2
                )

                # ------------------------------------------------------------------
                # MOTOR DE RECOMENDAÇÃO
                # ------------------------------------------------------------------

                # Explicação: Atribui o resultado da expressão à variável ou chave procedimentos, dermocosmeticos; essa informação será reutilizada nesta etapa do programa.
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

                # Explicação: Atribui o resultado da expressão à variável ou chave relatorio_html; essa informação será reutilizada nesta etapa do programa.
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

                # Explicação: Atribui o resultado da expressão à variável ou chave link_whatsapp; essa informação será reutilizada nesta etapa do programa.
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

            # Explicação: Apresenta uma mensagem de sucesso ao usuário.
            st.success(
                "Análise concluída e relatório gerado com sucesso!"
            )

            # Explicação: Insere uma linha divisória para separar visualmente as seções.
            st.divider()

            # Explicação: Exibe o cabeçalho da seção correspondente na interface.
            st.header("3. Resultado da análise facial")

            # Explicação: Atribui o resultado da expressão à variável ou chave col_img1, col_img2; essa informação será reutilizada nesta etapa do programa.
            col_img1, col_img2 = st.columns(2)

            # Explicação: Abre um contexto de execução gerenciado; o bloco indentado utiliza o recurso ou componente indicado.
            with col_img1:

                # Explicação: Exibe um subtítulo para organizar visualmente os resultados.
                st.subheader("Fotografia original")

                # Explicação: Exibe uma imagem na interface do Streamlit.
                st.image(
                    cv2.cvtColor(
                        img,
                        cv2.COLOR_BGR2RGB
                    ),
                    use_container_width=True
                )

            # Explicação: Abre um contexto de execução gerenciado; o bloco indentado utiliza o recurso ou componente indicado.
            with col_img2:

                # Explicação: Exibe um subtítulo para organizar visualmente os resultados.
                st.subheader("Regiões analisadas")

                # Explicação: Exibe uma imagem na interface do Streamlit.
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

            # Explicação: Exibe um subtítulo para organizar visualmente os resultados.
            st.subheader("Indicadores visuais experimentais")

            # Explicação: Atribui o resultado da expressão à variável ou chave col_a, col_b; essa informação será reutilizada nesta etapa do programa.
            col_a, col_b = st.columns(2)

            # Explicação: Abre um contexto de execução gerenciado; o bloco indentado utiliza o recurso ou componente indicado.
            with col_a:

                # Explicação: Apresenta um indicador numérico em destaque na interface.
                st.metric(
                    "Bordas - Testa",
                    f"{linhas_testa}%"
                )

            # Explicação: Abre um contexto de execução gerenciado; o bloco indentado utiliza o recurso ou componente indicado.
            with col_b:

                # Explicação: Apresenta um indicador numérico em destaque na interface.
                st.metric(
                    "Variação de tonalidade - Bochechas",
                    f"{tom_medio}"
                )

            # Explicação: Exibe uma legenda explicativa em tamanho reduzido na interface.
            st.caption(
                "Esses números representam medidas matemáticas da imagem. "
                "Não são escalas clínicas de rugas, manchas ou envelhecimento."
            )

            # ------------------------------------------------------------------
            # TABELA
            # ------------------------------------------------------------------

            # Explicação: Exibe um subtítulo para organizar visualmente os resultados.
            st.subheader("Detalhamento por região")

            # Explicação: Atribui o resultado da expressão à variável ou chave dados; essa informação será reutilizada nesta etapa do programa.
            dados = []

            # Explicação: Inicia um laço que percorre os elementos da sequência indicada, executando o bloco para cada elemento.
            for nome, valores in resultados_regioes.items():

                # Explicação: Executa esta instrução dentro da etapa atual do sistema; os valores e parâmetros são utilizados conforme o contexto descrito na seção.
                dados.append({
                    "Região facial": nome,
                    "Indicador de bordas (%)": valores[0],
                    "Variação de tonalidade": valores[1]
                })

            # Explicação: Exibe os dados tabulares das regiões analisadas.
            st.dataframe(
                dados,
                use_container_width=True,
                hide_index=True
            )

            # ==========================================================================
            # 11. RECOMENDAÇÕES
            # ==========================================================================

            # Explicação: Insere uma linha divisória para separar visualmente as seções.
            st.divider()

            # Explicação: Exibe o cabeçalho da seção correspondente na interface.
            st.header("4. Procedimentos estéticos sugeridos")

            # Explicação: Inicia um laço que percorre os elementos da sequência indicada, executando o bloco para cada elemento.
            for i, procedimento in enumerate(procedimentos, 1):

                # Explicação: Exibe conteúdo formatado em Markdown na interface.
                st.markdown(
                    f"**{i}. {procedimento}**"
                )

            # Explicação: Verifica uma condição. O bloco indentado abaixo só será executado quando a expressão for verdadeira.
            if restricao:

                # Explicação: Exibe um alerta visual para chamar atenção a uma condição que exige cautela.
                st.warning(
                    "A cliente informou uma condição que exige cautela. "
                    "Não realizar procedimentos ou iniciar ativos sem "
                    "avaliação específica de contraindicações."
                )

            # ==========================================================================
            # 12. DERMOCOSMÉTICOS
            # ==========================================================================

            # Explicação: Insere uma linha divisória para separar visualmente as seções.
            st.divider()

            # Explicação: Exibe o cabeçalho da seção correspondente na interface.
            st.header("5. Rotina dermocosmética")

            # Explicação: Exibe conteúdo formatado em Markdown na interface.
            st.markdown(
                f"**Limpeza facial:** {dermocosmeticos['Limpeza']}"
            )

            # Explicação: Exibe conteúdo formatado em Markdown na interface.
            st.markdown("**Ativos de tratamento:**")

            # Explicação: Inicia um laço que percorre os elementos da sequência indicada, executando o bloco para cada elemento.
            for ativo in dermocosmeticos["Ativos de Tratamento"]:

                # Explicação: Exibe conteúdo formatado em Markdown na interface.
                st.markdown(f"- {ativo}")

            # Explicação: Exibe conteúdo formatado em Markdown na interface.
            st.markdown(
                "**Hidratação e barreira cutânea:** "
                + dermocosmeticos["Hidratação e Barreira"]
            )

            # Explicação: Exibe conteúdo formatado em Markdown na interface.
            st.markdown(
                "**Fotoproteção:** "
                + dermocosmeticos["Fotoproteção"]
            )

            # ==========================================================================
            # 13. ACOMPANHAMENTO
            # ==========================================================================

            # Explicação: Insere uma linha divisória para separar visualmente as seções.
            st.divider()

            # Explicação: Exibe o cabeçalho da seção correspondente na interface.
            st.header("6. Acompanhamento")

            # Explicação: Exibe texto ou conteúdo na interface do Streamlit.
            st.write(
                f"**Retorno sugerido:** {prazo_followup}"
            )

            # Explicação: Verifica uma condição. O bloco indentado abaixo só será executado quando a expressão for verdadeira.
            if observacoes_profissional.strip():

                # Explicação: Exibe texto ou conteúdo na interface do Streamlit.
                st.write(
                    "**Observações profissionais:**"
                )

                # Explicação: Exibe texto ou conteúdo na interface do Streamlit.
                st.write(
                    observacoes_profissional
                )

            # ==========================================================================
            # 14. DOWNLOAD DO RELATÓRIO
            # ==========================================================================

            # Explicação: Insere uma linha divisória para separar visualmente as seções.
            st.divider()

            # Explicação: Exibe o cabeçalho da seção correspondente na interface.
            st.header("7. Gerar relatório")

            # Explicação: Cria um botão para baixar o relatório gerado, com nome e tipo de arquivo definidos.
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

            # Explicação: Exibe uma informação complementar ao usuário.
            st.info(
                "O relatório inclui a fotografia marcada, os indicadores, "
                "as recomendações e as informações da avaliação. "
                "Abra o arquivo baixado no navegador e utilize "
                "'IMPRIMIR / SALVAR EM PDF'."
            )

            # ==========================================================================
            # 15. WHATSAPP
            # ==========================================================================

            # Explicação: Insere uma linha divisória para separar visualmente as seções.
            st.divider()

            # Explicação: Exibe o cabeçalho da seção correspondente na interface.
            st.header("8. Compartilhar pelo WhatsApp")

            # Explicação: Cria um botão que abre o endereço do WhatsApp preparado anteriormente.
            st.link_button(
                "📲 ENVIAR RESUMO DA AVALIAÇÃO",
                link_whatsapp,
                use_container_width=True
            )

            # Explicação: Exibe uma legenda explicativa em tamanho reduzido na interface.
            st.caption(
                "O WhatsApp abrirá com o resumo preenchido. "
                "O envio precisa ser confirmado pelo usuário."
            )

            # ==========================================================================
            # 16. AVISO FINAL
            # ==========================================================================

            # Explicação: Insere uma linha divisória para separar visualmente as seções.
            st.divider()

            # Explicação: Apresenta uma mensagem de sucesso ao usuário.
            st.success(
                "Processamento finalizado. Relatório disponível para download."
            )

            # Explicação: Exibe uma legenda explicativa em tamanho reduzido na interface.
            st.caption(
                "A triagem é educativa e experimental. Não substitui "
                "avaliação dermatológica, diagnóstico ou prescrição."
            )

        # Explicação: Captura uma exceção ocorrida no bloco try para apresentar uma mensagem de erro sem encerrar silenciosamente a aplicação.
        except Exception as erro:

            # Explicação: Apresenta uma mensagem de erro na interface.
            st.error(
                "Não foi possível concluir a análise."
            )

            # Explicação: Exibe um alerta visual para chamar atenção a uma condição que exige cautela.
            st.warning(
                str(erro)
            )

            # Explicação: Executa esta instrução dentro da etapa atual do sistema; os valores e parâmetros são utilizados conforme o contexto descrito na seção.
            st.exception(erro)


# ==============================================================================
# FIM DO SISTEMA
# ==============================================================================
