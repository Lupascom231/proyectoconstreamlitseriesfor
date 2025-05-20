import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
from datetime import timedelta

st.set_page_config(layout="wide")

# Subida de archivo por el usuario
uploaded_file = st.file_uploader("Sube tu archivo de ventas (.xlsx)", type="xlsx")
if uploaded_file is not None:
    data = pd.read_excel(uploaded_file)

    # Cargar modelo y scaler
    @st.cache_resource
    def load_model_and_scaler():
        model = tf.keras.models.load_model('modelomaspreciso.keras')
        scaler = joblib.load('scalers.pkl')
        return model, scaler

    model, scaler = load_model_and_scaler()

    st.title("Predicción de Ventas con Deep Learning")

    # Selección de Plaza y CanalVenta
    display_cols = ['Plaza', 'CanalVenta']
    col1, col2 = st.columns(2)
    with col1:
        plaza = st.selectbox('Selecciona la Plaza', sorted(data['Plaza'].unique()))
    with col2:
        canal = st.selectbox('Selecciona el CanalVenta', sorted(data['CanalVenta'].unique()))

    # Filtrar datos según selección
    filtered = data[(data['Plaza'] == plaza) & (data['CanalVenta'] == canal)].copy()
    if filtered.empty:
        st.warning('No hay datos para la combinación seleccionada.')
        st.stop()

    # Eliminar columnas de selección para análisis posterior
    filtered = filtered.drop(columns=['Plaza', 'CanalVenta'])

    # Mostrar histórico y gráfico
    date_col = 'Fecha'
    ventas_col = 'Ventas'
    filtered[date_col] = pd.to_datetime(filtered[date_col])
    filtered = filtered.sort_values(date_col)

    st.subheader('Histórico de Ventas')
    st.dataframe(filtered[[date_col, ventas_col]].tail(30))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(filtered[date_col], filtered[ventas_col], label='Histórico')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Ventas')
    ax.set_title('Ventas Históricas')
    ax.legend()
    st.pyplot(fig)

    # Prueba con últimos 15 días
    time_step = 20
    st.subheader('Prueba con últimos 15 días')
    if len(filtered) < time_step + 15:
        st.warning('No hay suficientes datos para la prueba de 15 días.')
        st.stop()

    # Normalizar ventas para el modelo
    ventas = filtered[ventas_col].values.reshape(-1, 1)
    ventas_scaled = scaler.transform(ventas)

    # Tomar los últimos 15 días para prueba
    X_test = []
    y_test = []
    # Usar índices válidos para asegurar que todos los slices sean del mismo tamaño
    for i in range(len(ventas_scaled) - time_step - 15, len(ventas_scaled) - time_step):
        X_test.append(ventas_scaled[i:i+time_step, 0])
        y_test.append(ventas_scaled[i+time_step, 0])
    X_test = np.array(X_test).reshape(-1, time_step, 1)
    y_test = np.array(y_test)

    # Predicción sobre los 15 días
    y_pred = model.predict(X_test)
    y_pred_inv = scaler.inverse_transform(y_pred)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Graficar comparación
    dates_test = filtered[date_col].iloc[-15:].values
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(dates_test, y_test_inv, label='Real', marker='o')
    ax2.plot(dates_test, y_pred_inv, label='Predicción', marker='x')
    ax2.set_title('Prueba: Últimos 15 días')
    ax2.set_xlabel('Fecha')
    ax2.set_ylabel('Ventas')
    ax2.legend()
    st.pyplot(fig2)

    # Forecast futuro
    def forecast(model, last_data, scaler, n_steps=9, time_step=20):
        input_seq = last_data[-time_step:].reshape(1, time_step, 1)
        preds = []
        for _ in range(n_steps):
            pred = model.predict(input_seq, verbose=0)
            preds.append(pred[0][0])
            input_seq = np.concatenate([input_seq[:, 1:, :], pred.reshape(1, 1, 1)], axis=1)
        preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
        return preds

    st.subheader('Pronóstico Futuro (9 días)')
    forecast_vals = forecast(model, ventas_scaled, scaler, n_steps=9, time_step=time_step)
    last_date = filtered[date_col].iloc[-1]
    forecast_dates = [last_date + timedelta(days=i+1) for i in range(9)]

    forecast_df = pd.DataFrame({'Fecha': forecast_dates, 'Predicción': forecast_vals})
    st.dataframe(forecast_df)

    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.plot(filtered[date_col], filtered[ventas_col], label='Histórico')
    ax3.plot(forecast_df['Fecha'], forecast_df['Predicción'], label='Pronóstico', marker='o')
    ax3.set_title('Pronóstico de Ventas (9 días)')
    ax3.set_xlabel('Fecha')
    ax3.set_ylabel('Ventas')
    ax3.legend()
    st.pyplot(fig3)
else:
    st.info("Por favor, sube un archivo Excel para comenzar.")
