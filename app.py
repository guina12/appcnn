import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import numpy as np

# Carregar o modelo ao iniciar o aplicativo
@st.cache_resource
def load_my_model():
    return load_model('model_image.h5')

# Pré-processar a imagem
def preprocess_image(image):
    image = image.resize((244, 244))  # Redimensiona a imagem
    image = np.array(image)  # Converte para array numpy
    image = preprocess_input(image)  # Pré-processa a imagem para ResNet50
    image = np.expand_dims(image, axis=0)  # Adiciona dimensão de lote
    return image

def main():
    st.title("Classificação de Imagens com CNN para Deteção de Sono com Imagens.")
    st.write("Carregue uma imagem e o modelo CNN fará a previsão para deteção de sono..")

    uploaded_file = st.file_uploader("Escolha uma imagem ..", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Imagem Carregada', use_column_width=True)
        st.write("")
        st.write("Classificando...")

        model = load_my_model()
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)

        if prediction>=0.5:
            st.write(" Probabilidade {:.2f}  >  0.5  , os olhos estão fechados.. deteção de sono..".format(float(prediction)))
        else:
            st.write(" Probabilidade {:.2f}  < 0.5  , os  olhos estão  abertos, não há sono".format(float(prediction)))

if __name__ == "__main__":
    main()
