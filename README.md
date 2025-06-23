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
