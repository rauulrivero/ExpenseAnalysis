import streamlit as st
import pandas as pd

class StreamlitApp:
    def __init__(self, api_handler):
        self.api = api_handler

    def run(self):
        st.set_page_config(layout="wide")
        st.title("🧑‍👩‍👧‍👦 Gemelo Digital: Predicción y Simulación Fiscal")
        # Sidebar for mode selection
        mode = st.sidebar.selectbox(
            "Selecciona la funcionalidad:",
            ["Predicción de Gastos", "Simulación de Recaudación"]
        )

        # Common inputs form
        with st.expander("Datos del hogar y entorno", expanded=True):
            caprov     = st.selectbox("¿Es capital de provincia?", [0, 1], help="1 = Sí, 0 = No")
            tamamu     = st.selectbox("Tamaño del municipio", [1, 2, 3, 4, 5],
                                      help="1=<10k • 2=10–20k • 3=20–50k • 4=50–100k • 5=≥100k")
            densidad   = st.selectbox("Densidad de población", [1, 2, 3],
                                      help="1=Dispersa • 2=Intermedia • 3=Densa")
            superf     = st.number_input("Superficie de la vivienda (m²)", min_value=0, value=80)
            aguacali   = st.selectbox("¿Agua caliente?", [0, 1], help="1 = Sí, 0 = No")
            calef      = st.selectbox("¿Calefacción?", [0, 1], help="1 = Sí, 0 = No")
            zonares    = st.selectbox("Tipo de zona residencial", [1,2,3,4,5,6,7],
                                      format_func=lambda x: {
                                          1: "Rural agraria", 2: "Rural pesquera", 3: "Rural industrial",
                                          4: "Urbana inferior", 5: "Urbana media", 6: "Urbana alta", 7: "Urbana lujo"
                                      }[x])
            regten     = st.selectbox("Régimen de tenencia", [1,2,3,4,5,6],
                                      format_func=lambda x: {
                                          1:"Cesión gratuita",2:"Cesión semigratuita",3:"Renta reducida",
                                          4:"Alquiler",5:"Propiedad con hipoteca",6:"Propiedad sin hipoteca"
                                      }[x])
            numocu     = st.number_input("Miembros empleados", min_value=0, max_value=20, value=1)
            numacti    = st.number_input("Miembros activos", min_value=0, max_value=20, value=1)
            numperi    = st.number_input("Perceptores de ingresos", min_value=0, max_value=20, value=1)
            numestu    = st.number_input("Estudiantes", min_value=0, max_value=20, value=0)
            nadul_mas  = st.number_input("Adultos varones", min_value=0, max_value=20, value=1)
            nadul_fem  = st.number_input("Adultos mujeres", min_value=0, max_value=20, value=0)
            nnino_fem  = st.number_input("Niñas", min_value=0, max_value=20, value=0)
            nnino_mas  = st.number_input("Niños", min_value=0, max_value=20, value=0)
            ocusp      = st.selectbox("¿Proveedor principal empleado?", [0,1], help="1 = Sí, 0 = No")
            edadsp     = st.number_input("Edad del proveedor principal", min_value=16, max_value=85, value=40)
            nacion_esp = st.selectbox("¿Nacionalidad española?", [0,1], help="1 = Sí, 0 = No")
            educ_sup   = st.selectbox("¿Educación superior?", [0,1], help="1 = Sí, 0 = No")
            caprop     = st.selectbox("¿Ingresos cuenta propia?", [0,1], help="1 = Sí, 0 = No")
            cajena     = st.selectbox("¿Ingresos cuenta ajena?", [0,1], help="1 = Sí, 0 = No")
            disposiov  = st.selectbox("Otra vivienda últimos 12m", [0,1], help="1 = Sí, 0 = No")
            impexac    = st.number_input("Ingresos netos mensuales (€)", value=1200.0)
            tmax_max   = st.number_input("Temp. máxima anual (°C)", value=22.0)
            tmin_min   = st.number_input("Temp. mínima anual (°C)", value=6.0)
            tasa_paro  = st.number_input("Tasa de paro (%)", value=12.0)
            inflacion  = st.number_input("Inflación (%)", value=2.5)
            tipo_int   = st.number_input("Tipo de interés (%)", value=0.5)
            inputs = [
                caprov, tamamu, densidad, superf, aguacali, calef,
                zonares, regten, numocu, numacti, numperi, numestu,
                nadul_mas, nadul_fem, nnino_fem, nnino_mas, ocusp,
                edadsp, nacion_esp, educ_sup, caprop, cajena, disposiov,
                impexac, tmax_max, tmin_min, tasa_paro, inflacion, tipo_int
            ]

        # Common button
        if mode == "Predicción de Gastos":
            if st.button("🔮 Predecir Gastos"):
                payload = {"inputs": inputs}
                resp = self.api.post_request("predict", payload)
                if resp.ok:
                    preds = {int(k):v for k,v in resp.json().get("predictions",{}).items()}
                    df = pd.DataFrame.from_dict(preds, orient="index", columns=["Gasto (€)"])
                    df = df.sort_index()
                    st.success("✅ Predicciones:")
                    st.table(df)
                    st.markdown("## Gráfico de Gastos")
                    st.bar_chart(df["Gasto (€)"])
                else:
                    st.error(resp.json().get("error"))

        else:  # Simulación de Recaudación
            deduccion = st.number_input("Deducción neta mensual por hijo (€)", min_value=0.0, value=50.0)
            ccaa = st.number_input("CCAA (1–18)", min_value=1, max_value=18, value=1)
            if st.button("💰 Simular Recaudación"):
                payload = {
                    "inputs": inputs,
                    "deduction_per_child": deduccion,
                    "ccaa": int(ccaa)
                }
                resp = self.api.post_request("simulate_tax", payload)
                if resp.ok:
                    result = resp.json()
                    df_imp = pd.DataFrame.from_dict(result['impact'], orient='index', columns=['Impacto (€)'])
                    df_rev = pd.DataFrame.from_dict(result['revenue_by_category'], orient='index', columns=['Recaudación (€)'])
                    st.success(f"✅ Total Recaudación: {result['total_revenue']:.2f} €")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("### Impacto en Gastos por Categoría")
                        st.bar_chart(df_imp['Impacto (€)'].sort_index())
                    with col2:
                        st.markdown("### Recaudación por Categoría")
                        st.bar_chart(df_rev['Recaudación (€)'].sort_index())
                else:
                    st.error(resp.json().get("error"))
