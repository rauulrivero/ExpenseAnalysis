# ExpenseAnalysis


Este repositorio contiene el código y los recursos necesarios para trabajar con microdatos del Instituto Nacional de Estadística (INE) desde el año 2006 hasta 2023, junto con otros indicadores como el IPC y la tasa de paro por comunidad autónoma.

## 📥 Paso 1: Recogida de Microdatos

1. Accede al siguiente enlace del INE para descargar los microdatos:
   👉 [Encuesta de Presupuestos Familiares (EPF) - Resultados](https://www.ine.es/dyngs/INEbase/operacion.htm?c=Estadistica_C&cid=1254736176806&menu=resultados&idp=1254735976608#_tabs-1254736195147)

2. Para **cada año desde 2006 hasta 2023**, descarga los microdatos disponibles (normalmente vienen en un archivo ZIP que contiene tres ficheros):

   - **Gastos**
   - **Hogar**
   - **Miembros del hogar**

3. Descomprime los archivos y organiza la información en carpetas siguiendo esta estructura:

```
data/
├── 2006/
│ ├── gastos_2006
│ ├── hogar_2006
│ └── miembros_2006
├── 2007/
│ ├── ...
...
└── 2023/
```

---

## 📈 Paso 2: Descargar datos complementarios

### 🔸 IPC por Comunidad y Año

1. Entra en el siguiente enlace:
👉 [IPC por comunidad autónoma y año - INE](https://www.ine.es/jaxiT3/Tabla.htm?t=50913)

2. Filtra los datos para los años 2006–2023 y **exporta el resultado como archivo CSV**.

3. Guarda el archivo en el directorio raíz del proyecto con un nombre como: ipc_comunidades.csv


---

### 🔸 Tasa de Paro por Comunidad y Año

1. Accede a esta tabla:
👉 [Tasa de paro por comunidad y año - INE](https://www.ine.es/jaxiT3/Tabla.htm?t=4247)

2. Selecciona los años 2006–2023 y exporta el CSV con los datos.

3. Guarda el archivo con un nombre como: tasa_paro_comunidades.csv



---

## 📚 Otros datos

El resto de los datos utilizados ya están incluidos directamente en el notebook de trabajo y no requieren descarga adicional.

---

## ✅ Estructura esperada

```
ExpenseAnalysis/
  data/
    2006/
    2007/
    ...
    ipc_comunidades.csv
    tasa_paro_comunidades.csv
  notebooks/
    *.ipynb
  README.md
```

## 🧮 Orden de Ejecución de Notebooks

Una vez descargados y organizados los datos, se deben ejecutar los notebooks en el siguiente orden para procesar toda la información y generar los datamarts necesarios.

---

### 1️⃣ Visualización y Recolección de Datos de Entorno

**Notebook**: `notebooks/views/view_and_collect_data.ipynb`

Este notebook permite comprobar que los datos recolectados tienen un formato válido. Además, se encarga de recoger los datos de entorno faltantes y generar los correspondientes archivos CSV que se utilizarán en pasos posteriores.

---

### 2️⃣ Procesamiento y Unificación de Datos

**Notebook**: `notebooks/proceesing/proccess_data.ipynb`

Este notebook construye el **datalake principal**:

- Unifica la estructura de los microdatos de todos los años en un único dataset coherente.
- Agrega los datos de entorno (IPC, tasa de paro, etc.).
- Agrupa la información por año para facilitar el análisis.


A continuación se muestra la estructura del DataLake generada:

| Carpeta (Año) | Archivo                        | Descripción                                                                 |
|---------------|--------------------------------|-----------------------------------------------------------------------------|
| `2006`–`2023` | `family_expenses.tsv`          | Información detallada sobre los gastos familiares                          |
|               | `homes.tsv`                    | Información sociodemográfica del hogar                                     |
|               | `external_indicators.tsv`      | Indicadores externos añadidos (IPC, tasa de paro, etc.) por comunidad y año |

---


### 3️⃣ Agregación por Categorías de Gasto

**Notebook**: `notebooks/proceesing/proccess_codes.ipynb`

Este notebook transforma los datos económicos en distintos niveles de agregación:

- Añade columnas con el gasto monetario agrupado por categorías superiores.
- Genera **dos datamarts**:
  - Uno con agrupación por la **categoría de gasto más alta**.
  - Otro con agrupación por **subcategorías**.

Esto es necesario porque los datos originales contienen el gasto clasificado únicamente en el nivel más bajo de codificación.

---

### 4️⃣ Creación del Datamart Final (Formato Picota)

**Notebook**: `notebooks/datamart_builder/create_datamart_picota.ipynb`

Este notebook genera el datamart final que se usará para el entrenamiento de modelos explicativos y estimativos.  
El formato resultante es compatible con **Picota**, una herramienta de análisis de gasto y explicabilidad de modelos.

> 📊 Este es el formato de entrada requerido por Picota para aplicar técnicas avanzadas de machine learning sobre los datos procesados.



## 🧠 Modelo Explicativo

Una vez construido el datamart final, se procede a analizar qué variables explican el gasto en cada categoría mediante distintos enfoques de análisis de sensibilidad.

---

### 5️⃣ Análisis de Sensibilidad

Ejecutar los siguientes notebooks en orden:

1. **Análisis lineal de sensibilidad**  
   📘 `notebooks/sensitivity_analysis/sensitivity_analisys_advanced.ipynb`  
   Realiza un análisis de sensibilidad utilizando regresión lineal para evaluar la influencia de cada variable en el gasto por categoría.

2. **Análisis no lineal de sensibilidad**  
   📘 `notebooks/sensitivity_analysis/sensitivity_analisys_nolineal.ipynb`  
   Utiliza modelos no lineales (como redes neuronales u otros métodos avanzados) para detectar relaciones más complejas entre las variables y el gasto.

---

### 6️⃣ Comparación de Modelos

📘 `notebooks/research/linear_and_nolinear_comparison.ipynb`

Este notebook compara el rendimiento de los modelos explicativos obtenidos en los pasos anteriores:

- Muestra los valores de **R²** de cada modelo por categoría de gasto.
- Presenta los **coeficientes** (modelo lineal) y **importancias** (modelo no lineal).
- Permite identificar las variables que más influyen en el gasto en cada categoría.

> 🎯 Esta etapa es clave para entender el comportamiento del gasto en los hogares y qué factores lo determinan, tanto desde una perspectiva lineal como no lineal.

## 🔧 Paso 7️⃣: Creación del gemelo digital en Quassar/Picota

Tras construir y validar el datamart final en formato compatible con **Picota**, el siguiente paso es crear nuestro **gemelo digital** para modelizar el comportamiento de los hogares.

### ✍️ 7.1 Registro en Quassar y Picota

1️⃣ Accede a [Quassar](https://quassar.io/) y [Picota](https://picota.io/).  
2️⃣ Crea una cuenta gratuita o inicia sesión si ya tienes una.  
3️⃣ Desde el panel de usuario de **Picota**, crea un nuevo proyecto y selecciona como fuente de datos el datamart generado en el paso anterior.

---

### 🔹 7.2 Definir la `Reality` y el `DigitalSubject` en DSL

Usaremos el **DSL de Picota** (Domain-Specific Language) para definir formalmente nuestra `Reality España` y el `DigitalSubject hogar`.

A continuación, reproducir una estructura **como se muestra en la ilustración `quassar_picota_subject.png`**:

```picota
Reality España

Variable tasaCambioEurUsd is Numeric
Variable tipoInteres is Numeric

Subject hogar is Prototype(prefix = "hogar")
    Variable temperaturaMedia is Numeric
    Variable tasaParo is Numeric
    Variable inflacion is Numeric
    Variable ipc is Numeric
    Variable capitalProvincia is Boolean
    Variable tamanoMunicipio is Numeric
    Variable densidad is Numeric
    Variable superficie is Numeric
    Variable tipoCasa is Numeric
    Variable aguaCaliente is Boolean
    Variable calefaccion is Boolean
    Variable zonaResidencial is Boolean
    Variable regimenTenencia is Numeric
    Variable edadSp is Numeric
    Variable espanolSp is Boolean
    Variable educacionSuperiorSp is Boolean
    Variable numeroViviendasAdicionales is Numeric
    Variable ingresosNetos is Numeric
    Variable tasaAhorro is Numeric
    Variable gastoNoMonetario is Numeric
    Variable comidasTotales is Numeric
    Variable fuentePrincipalIngresos is Enumerated("pension" "asalariado" "autonomoYRenta")
    Variable gastoMonetario is Numeric Composite
        Components("productosAlimenticios11" "bebidasNoAlcoholicas12" "bebidasAlcoholicas21" "tabaco22" ... "remesas128")
    Variable miembros is Numeric Composite
        Components("ancianos" "adultos" "ninos")
        Components("masculinos" "femeninos")
        Components("ocupados" "noOcupados")
        Components("activos" "noActivos")
        Components("conIngresos" "sinIngresos")
        Components("estudiantes" "noEstudiantes")
```

### 🔹 7.3 Definir el `DigitalTwin`

A continuación, se define el `DigitalTwin`, siguiendo el ejemplo de la ilustración `quassar_picota_twin.png`:

```picota
DigitalTwin
    DigitalSubject
        subject = España.hogar
        Resolution(scale = Hours)
        InferenceModel
            variable = España.hogar.gastoMonetario
```

🔨 **Una vez definido el `DigitalTwin`, aún no está listo el gemelo digital:**

1️⃣ Luego, deberás pulsar el botón `Build in Picota` desde el editor de `Quassar`.  
2️⃣ En el paso siguiente, debes **comprimir todos los archivos `.tsv` ubicados dentro de la carpeta generada `/picotaData`** (⚠️ *no comprimir la carpeta completa, solo su contenido*).  
3️⃣ Después, en la interfaz de `Picota`, **subir ese archivo comprimido en la sección correspondiente y pulsar `Construir`**.

✅ Al finalizar este proceso, tendrás el **gemelo digital listo para ejecutar estimaciones y análisis avanzados sobre el gasto monetario de los hogares en España**.

---

## 8️⃣ Preparar hogares para simulación y ajustar impuestos

Una vez **entrenado el gemelo digital**, se deben ejecutar los siguientes pasos para preparar la simulación fiscal:

### 🔹 8.1 Crear datamart de hogares simulados

📓 Ejecutar el notebook:  
`notebooks/create_datamart_irpf.ipynb`  

✅ Esto generará los hogares con las datos necesarios para la simulación fiscal.

---

### 🔹 8.2 Ajustar impuestos y crear datamart de tipos impositivos

📓 Ejecutar el notebook:  
`notebooks/create_datamart_indirect_taxes.ipynb`  

🔧 Antes de ejecutarlo puedes **ajustar manualmente los tipos impositivos para reflejar la realidad fiscal deseada** (o bien dejar los valores por defecto).

✅ Al finalizar este notebook, quedarán configurados los tipos de impuestos indirectos que se utilizarán en la simulación (IVA, IGIC, IPSI).

---

---

## 9️⃣ Configurar API y ejecutar la simulación fiscal

### 🔹 9.1 Copiar API ID del gemelo digital

Una vez construido el gemelo digital en Picota:

- Accede al gemelo digital en Picota y copia el valor de `API_ID` que se ha generado.

---

### 🔹 9.2 Configurar API en el notebook de simulación

📓 Abre el notebook:  
`notebooks/irpf_simulation.ipynb`

🔧 Sustituye la línea:

```python
API_ID = 'TU_API_ID'
```

por:

```python
API_ID = 'el_api_id_de_tu_gemelo'
```

### 🔹 9.3 Ejecutar simulación

✅ Ejecuta el notebook completo `notebooks/irpf_simulation.ipynb` para simular distintos escenarios de deducción fiscal.

📝 **Opcional:**  
Puedes ajustar las **probabilidades de nacimiento en cada escenario** y el **importe a deducir** para personalizar los escenarios según tus necesidades antes de ejecutar la simulación.
