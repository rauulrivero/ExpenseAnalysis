import streamlit as st
import pandas as pd

class StreamlitApp:
    def __init__(self, api_handler):
        self.api = api_handler

    def run(self):
        st.set_page_config(layout="wide")
        st.title("🏡 Gemelo Digital: Inferencia de Gastos del Hogar")

        codigos_comunidad = {
            "Andalucía": 1, "Aragón": 2, "Asturias": 3, "Islas Baleares": 4,
            "Canarias": 5, "Cantabria": 6, "Castilla-La Mancha": 7, "Castilla y León": 8,
            "Cataluña": 9, "Comunidad Valenciana": 10, "Extremadura": 11, "Galicia": 12,
            "Madrid": 13, "Murcia": 14, "Navarra": 15, "País Vasco": 16,
            "La Rioja": 17, "Ceuta": 18, "Melilla": 19
        }

        st.subheader("✍️ Introduce los datos del hogar")

        comunidad = st.selectbox("Comunidad Autónoma", list(codigos_comunidad.keys()))
        cap_prov = st.checkbox("¿Es capital de provincia?")
        tamano = st.selectbox("Tamaño del municipio", [1, 2, 3, 4, 5])
        densidad = st.selectbox("Densidad de población", [1, 2, 3])
        superficie = st.number_input("Superficie (m²)", value=100.0)
        tipo_casa_map = {
             "Casa económica": 1,
            "Casa media": 2,
            "Chalet": 3
        }
        tipo_casa_label = st.selectbox("Tipo de vivienda", list(tipo_casa_map.keys()))
        tipo_casa = tipo_casa_map[tipo_casa_label]
        agua_caliente = st.checkbox("¿Tiene agua caliente?")
        calefaccion = st.checkbox("¿Tiene calefacción?")
        zona = st.selectbox("Zona residencial", list(range(1, 8)))
        tenencia = st.selectbox("Régimen de tenencia", list(range(1, 7)))
        comidas = st.number_input("Comidas totales cada dos semanas", value=90.0)

        estudiantes = st.number_input("🎓 Miembros estudiantes", value=0, min_value=0)
        ocupados = st.number_input("👨‍💼 Miembros ocupados", value=3, min_value=0)
        activos = st.number_input("🔧 Miembros activos", value=3, min_value=0)

        # Distribución por edad y sexo
        anc_mas = st.number_input("👴 Ancianos masculinos", value=0, min_value=0)
        anc_fem = st.number_input("👵 Ancianas femeninas", value=0, min_value=0)
        adult_mas = st.number_input("🧔 Adultos masculinos", value=1, min_value=0)
        adult_fem = st.number_input("👩 Adultas femeninas", value=1, min_value=0)
        ninos_mas = st.number_input("👦 Niños", value=0, min_value=0)
        ninos_fem = st.number_input("👧 Niñas", value=0, min_value=0)

        ingresos_mas = st.number_input("💶 Con ingresos (masculino)", value=0, min_value=0)
        ingresos_fem = st.number_input("💶 Con ingresos (femenino)", value=2, min_value=0)

        # Totales y derivados
        miembros_totales = anc_mas + anc_fem + adult_mas + adult_fem + ninos_mas + ninos_fem
        no_estudiantes = max(0, miembros_totales - estudiantes)
        no_ocupados = max(0, miembros_totales - ocupados)
        no_activos = max(0, miembros_totales - activos)

        masculinos_total = anc_mas + adult_mas + ninos_mas
        femeninos_total = anc_fem + adult_fem + ninos_fem

        sin_ingresos_mas = max(0, masculinos_total - ingresos_mas)
        sin_ingresos_fem = max(0, femeninos_total - ingresos_fem)

        edadsp = st.number_input("Edad del sustentador principal", value=53)
        espanol = st.checkbox("¿Tiene nacionalidad española?")
        educ_sup = st.checkbox("¿Educación superior?")
        fuente = st.selectbox("Fuente principal de ingresos", ["asalariado", "autonomYRenta", "pension"])
        otras_viviendas = st.number_input("Nº viviendas adicionales", value=0.0)
        ingresos = st.number_input("Ingresos netos anuales (€)", value=15000.0)
        gasto_no_monetario = st.number_input("Gasto no monetario (€)", value=2700.0)
        tasa_ahorro = st.number_input("Tasa de ahorro (%)", value=-0.5)
        temp_media = st.number_input("Temperatura media anual (°C)", value=15.0)
        paro = st.number_input("Tasa de paro (%)", value=5.5)
        inflacion = st.number_input("Inflación (%)", value=2.9)
        interes = st.number_input("Tipo de interés (%)", value=3.5)
        cambio = st.number_input("Tipo de cambio EUR/USD", value=1.26)
        ipc = st.number_input("IPC", value=80.99)

        if st.button("🔮 Realizar Inferencia"):
            payload = {
                "instant": "2006-01-04T08:00:00Z",
                "capitalProvincia": cap_prov,
                "tamanoMunicipio": tamano,
                "densidad": densidad,
                "superficie": superficie,
                "tipoCasa": tipo_casa,
                "aguaCaliente": agua_caliente,
                "calefaccion": calefaccion,
                "zonaResidencial": zona,
                "regimenTenencia": tenencia,
                "comidasTotales": comidas,
                "miembros:estudiantes": estudiantes,
                "miembros:noEstudiantes": no_estudiantes,
                "miembros:ocupados": ocupados,
                "miembros:noOcupados": no_ocupados,
                "miembros:activos": activos,
                "miembros:noActivos": no_activos,
                "miembros:ancianos:masculinos": anc_mas,
                "miembros:ancianos:femeninos": anc_fem,
                "miembros:adultos:masculinos": adult_mas,
                "miembros:adultos:femeninos": adult_fem,
                "miembros:ninos:masculinos": ninos_mas,
                "miembros:ninos:femeninos": ninos_fem,
                "miembros:conIngresos:masculinos": ingresos_mas,
                "miembros:conIngresos:femeninos": ingresos_fem,
                "miembros:sinIngresos:masculinos": sin_ingresos_mas,
                "miembros:sinIngresos:femeninos": sin_ingresos_fem,
                "edadSp": edadsp,
                "espanolSp": espanol,
                "educacionSuperiorSp": educ_sup,
                "fuentePrincipalIngresos": fuente,
                "numeroViviendasAdicionales": otras_viviendas,
                "ingresosNetos": ingresos,
                "gastoNoMonetario": gasto_no_monetario,
                "tasaAhorro": tasa_ahorro,
                "temperaturaMedia": temp_media,
                "tasaParo": paro,
                "inflacion": inflacion,
                "España.tipoInteres": interes,
                "España.tasaCambioEurUsd": cambio,
                "ipc": ipc
            }

            # Añadir categorías de gasto a cero
            categorias = [
                "productosAlimenticios11", "bebidasNoAlcoholicas12", "bebidasAlcoholicas21", "tabaco22",
                "articulosDeVestir31", "calzado32", "alquileresRealesDeLaVivienda41", "mantenimientoDeLaVivienda43",
                "suministroDeAgua44", "electricidadGasOtrosCombustibles45", "mueblesRevestimientos51",
                "textilesParaElHogar52", "grandesElectrodomesticos53", "utensiliosDelHogar54",
                "herramientasCasaJardin55", "bienesServiciosParaElHogar56", "productosFarmaceuticos61",
                "serviciosMedicosAmbulatorios62", "serviciosHospitalarios63", "compraDeVehiculos71",
                "usoDeVehiculosPersonales72", "serviciosDeTransporte73", "serviciosPostales81",
                "equiposTelefonoFax82", "serviciosTelefonoFax83", "audiovisualesTecnologia91",
                "bienesDuraderosDeOcio92", "ocioJardineriaYMascotas93", "serviciosRecreativosYCulturales94",
                "prensaYPapeleria95", "paquetesTuristicos96", "educacionInfantilYPrimaria101",
                "educacionSecundariaYPostsecundaria102", "educacionSuperior103", "educacionNoFormal104",
                "restauracion111", "alojamiento112", "cuidadosPersonales121", "efectosPersonales123",
                "proteccionSocial124", "seguros125", "serviciosFinancieros126", "otrosServicios127", "remesas128"
            ]
            for cat in categorias:
                payload[f"gastoMonetario:{cat}"] = 0.0

            ccaa_codigo = codigos_comunidad[comunidad]

            resp = self.api.post_request("inferencia", {
                "payload": [payload],
                "ccaa": str(ccaa_codigo)
            })

            if resp.ok:
                resultado = pd.DataFrame(resp.json())
                st.success("✅ Resultado de la inferencia")
                st.dataframe(resultado)
                st.bar_chart(resultado.drop(columns=["subject"], errors="ignore").T)
            else:
                st.error(f"❌ Error: {resp.text}")
