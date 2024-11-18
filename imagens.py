import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

st.title("Análise de Imagens com Streamlit")

# Carregar múltiplas imagens
uploaded_files = st.file_uploader("Escolha uma ou mais imagens", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Abrir a imagem em cores usando PIL e exibir
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Imagem carregada: {uploaded_file.name}")

        # Converter a imagem para formato numpy e BGR (usado pelo OpenCV)
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Exibir histogramas de cada canal de cor
        st.write(f"### Histogramas dos Canais de Cor - {uploaded_file.name}")
        color = ('b', 'g', 'r')
        fig, ax = plt.subplots()
        for i, col in enumerate(color):
            hist = cv2.calcHist([image_bgr], [i], None, [256], [0, 256])
            ax.plot(hist, color=col)
        ax.set_title(f"Histogramas para os Canais B, G e R - {uploaded_file.name}")
        ax.set_xlabel("Intensidade do Pixel")
        ax.set_ylabel("Frequência")
        st.pyplot(fig)

        # Redimensionamento da imagem
        resized_image = cv2.resize(image_bgr, (128, 128))
        st.image(resized_image, caption="Imagem Redimensionada (128x128)", use_container_width=True)

        # Aplicar filtro de borda na imagem em cores
        edges = cv2.Canny(resized_image, threshold1=100, threshold2=200)
        st.image(edges, caption="Detecção de Bordas", use_container_width=True)

        # Contagem de objetos
        st.write(f"### Contagem de Objetos - {uploaded_file.name}")
        gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)

        # Encontrar contornos
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Ajustar os parâmetros de área mínima e máxima dos contornos
        min_area = 20  # Área mínima para um contorno ser considerado um objeto
        max_area = 2000  # Área máxima para um contorno ser considerado um objeto
        filtered_contours = [contour for contour in contours if min_area <= cv2.contourArea(contour) <= max_area]

        # Contar os objetos detectados
        object_count = len(filtered_contours)
        st.write(f"Número de objetos detectados: {object_count}")

        # Desenhar contornos na imagem original
        image_with_contours = image_bgr.copy()
        cv2.drawContours(image_with_contours, filtered_contours, -1, (0, 255, 0), 2)
        st.image(image_with_contours, caption="Imagem com Contornos Detectados", use_container_width=True)

        # Exibir imagem binária usada na detecção
        st.image(binary_image, caption="Imagem Binária para Detecção de Objetos", use_container_width=True)

        # Calcular a cor predominante baseada na intensidade dos pixels em cada canal (R, G, B)
        mean_red = np.mean(image_bgr[:, :, 2])
        mean_green = np.mean(image_bgr[:, :, 1])
        mean_blue = np.mean(image_bgr[:, :, 0])

        if mean_red > mean_green and mean_red > mean_blue:
            predominant_color = "Vermelho"
            color_box = (255, 0, 0)
        elif mean_green > mean_red and mean_green > mean_blue:
            predominant_color = "Verde"
            color_box = (0, 255, 0)
        elif mean_blue > mean_red and mean_blue > mean_green:
            predominant_color = "Azul"
            color_box = (0, 0, 255)
        else:
            predominant_color = "Sem cor predominante clara"
            color_box = (255, 255, 255)

        st.write(f"A cor predominante da imagem é: {predominant_color}")
        st.image(np.full((100, 100, 3), color_box, dtype=np.uint8), caption="Cor Predominante", use_container_width=True)

        st.write(f"Intensidade Média - Vermelho: {mean_red:.2f}")
        st.write(f"Intensidade Média - Verde: {mean_green:.2f}")
        st.write(f"Intensidade Média - Azul: {mean_blue:.2f}")
