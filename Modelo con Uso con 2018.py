#!/usr/bin/env python
# coding: utf-8

# ## Análisis de Datos Investigación en Deforestación 

# In[3]:


get_ipython().system(' pip install --pre pycaret')
get_ipython().system(' pip install --sweetviz')
get_ipython().system(' pip install --upgrade sweetviz')
get_ipython().system(' pip install pandas')
get_ipython().system(' pip install missingno')
get_ipython().system(' pip install --upgrade pycaret')
get_ipython().system(' pip show pycaret')
import warnings
warnings.filterwarnings('ignore', '.*do not.*', )
warnings.warn('DelftStack')
warnings.warn('Do not show this message')



# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import missingno
from pandas import Index


# In[5]:


Datos = pd.read_csv('C:\\Users\\lepid\\Documents\\Los libertadores\\Proyecto_Articulo_EstadisticaAplicada\\Datos encuestas y notebooks\\Base nueva_usos_otros_lim_mod_ret.csv')
Datos


# In[6]:


Datos.columns.to_list()


# ## Análisis descriptivo exploratorio de los datos
# 
# Los datos que aparecen con grd, han sido convertidos a grilla desde polígonos a grilla regular usando índices de proporcionalidad. Esto teniendo en cuenta que la base original los datos se dan con relación a las veredas. 
# 
# **grd_Id'**, ID de grilla
# 
#  **MGN_Id_concat**: ID concatenación información por veredas para el año 2005
#  **cod_AG**: Código asignado por el Agustin Codazzi
# 
#  **pers_edad_int**: Edad de las personas como número entero
# 
#  **MGN_Xm_U18N**: Ubicación en coordenadas X
# 
#  **MGN_Ym_U18N**: Ubicación en coordenadas y
# 
#  **SUP_cod_AG**: Código agregado Agustín Codazzi
# 
#  **idx_weightByArea**: Peso de la información para transformación por área d everedad
# 
#  **idx_count_hogares**: Conteo de hogares por vereda
# 
#  **idx_count_pers**: Total personas por vereda
# 
#  **idx_count_hombre**: Cantidad Hombres por vereda
# 
#  **idx_count_mujer**: Cantidad mujeres por vereda
# 
#  **idx_count_no_etno**: Cantidad personas no étnicas
# 
#  **idx_count_indig**: Cantidad personas indígenas
# 
#  **idx_count_negrit**: Cantidad personas negritudes
# 
#  **idx_count_0-9a**: Personas con edades entre los 0 y 9 años
#  
#  **idx_count_10-19a**: Personas con edades entre los 10 y 19 años
# 
#  **idx_count_20-29a**: Personas con edades entre los 20 y 29 años
# 
#  **idx_count_30-39a**: Personas con edades entre los 30 y 39 anos 
# 
#  **idx_count_40-49a**: Personas con edades entre los 40 y 49 años
# 
#  **idx_count_50-59a**: Personas con edades entre los 50 y 59 años
# 
#  **idx_count_60-69a**: Personas con edades entre los 60 y 69 años
# 
#  **idx_count_70-79a**: Personas con edades entre los 70 y 79 años
# 
#  **idx_count_alfab**: Personas alfabetizadas
#  
#  **idx_count_analfab**: Personas analfabetas
# 
#  **idx_count_prim18**: personas mayores de 18 años con nivel máximo de formación primaria
# 
#  **idx_count_secun18**: personas mayores de 18 años con nivel máximo de formación secundaria
# 
#  **idx_count_tecn18**: personas mayores de 18 años con nivel máximo de formación técnico
# 
#  **idx_count_psup18**: personas mayores de 18 años con nivel máximo de formación postgrado
# 
#  **idx_count_ocup**: Personas ocupadas
# 
#  **idx_count_desempl**: Personas desempleadas 
# 
#  **idx_count_ocEstud**: Ocupación Estudiantes
# 
#  **idx_count_ocHogar**: Ocupación Labores del hogar 
# 
#  **idx_count_ocRetir**: Ocupación Retirados
#  
#  **TLoss2kAvg**: Promedio perdida de vegetación (2km diámetro)
# 
#  **TCovr2kAvg**: Promedio cobertura de vegetación (2km diámetro)
# 
#  **ALOS2kAvg**: Promedio altura (2km diámetro)
# 
#  **Smp2kxvect**: componente vectorial en x
# 
#  **Smp2kyvect**: componente vectorial en y
# 
#  **Weigh2kAvg**: Peso por centroide 

# In[7]:


missingno.matrix(Datos)


# In[8]:


import sweetviz as sv

reporte = sv.analyze(Datos, target_feat='2018_TLoss2kAvg')
reporte.show_notebook()


# In[9]:


reporte = sv.analyze(Datos, target_feat='TLoss2kAvg')
reporte.show_notebook()


# In[10]:


# Lista de nombres de columnas a eliminar
columnas_a_eliminar = ['pers_edad_int',
 'Smp2kxvect',
 'Smp2kyvect',
 'Weigh2kAvg',
 'idx_count_0-9a',
 '2018_grd_Id',
 '2018_grd_n_edif',
 '2018_grd_n_hogr',
 '2018_grd_totPer',
 '2018_num_vivnda',
 '2018_num_vIndig',
 '2018_num_vivEtn',
 '2018_num_vivOtr',
 '2018_Id2_zona',
 '2018_idx_unit',
 '2018_idx_pers',
 '2018_grd_Y_elec',
 '2018_grd_N_elec',
 '2018_grd_person',
 '2018_grd_hombre',
 '2018_grd_mujere',
 '2018_grd_0-9añ',
 '2018_grd_10-19a',
 '2018_grd_20-29a',
 '2018_grd_30-39a',
 '2018_grd_40-49a',
 '2018_grd_50-59a',
 '2018_grd_60-69a',
 '2018_grd_70-79a',
 '2018_grd_80plus',
 '2018_grd_primar',
 '2018_grd_secund',
 '2018_grd_tecpro',
 '2018_grd_postgr',
 '2018_grd_sinedu',
 '2018_TLoss2kAvg',
 '2018_TCovr2kAvg',
 '2018_ALOS2kAvg',
 '2018_Smp2kxvect',
 '2018_Smp2kyvect',
 '2018_Weigh2kAvg',
 ]

# Eliminar las columnas del DataFrame
df = Datos.drop(columnas_a_eliminar, axis=1)

archivo_resultante = "C:\\Users\\lepid\\Documents\\Los libertadores\\Proyecto_Articulo_EstadisticaAplicada\\Integrado_sin._mod.csv"

# Guardar el nuevo DataFrame en un archivo CSV en el disco duro
df.to_csv(archivo_resultante, index=False)

print(f"Archivo guardado en {archivo_resultante}")

# Muestra los primeros registros del DataFrame después de eliminar las columnas
print(df.head())


# In[11]:


reporte2 = sv.analyze(df, target_feat='TLoss2kAvg')
reporte2.show_notebook()


# In[12]:


from sklearn.model_selection import train_test_split
from pycaret import regression

seed = 123

# Paso 1: Preparar tabla 
# (ya se hizo se llama df)

# Paso 2: Separar en X,y
y_obj = 'TLoss2kAvg' 
X = df.drop(columns=[y_obj])
y = df[y_obj]

# Paso 3: Separar en entrenamiento y validación
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
Xy_train = pd.concat([X_train, y_train], axis=1)
Xy_test = pd.concat([X_test, y_test], axis=1)

# Corrige la configuración de la regresión
regression.setup(data=Xy_train, target=y_obj)


# In[13]:


# Probar distintos modelos
best_model = regression.compare_models()


# Model: El nombre del modelo de regresión evaluado.
# 
# MAE (Mean Absolute Error): Es el promedio de las diferencias absolutas entre las predicciones del modelo y los valores reales.
# 
# Mide el error promedio entre las predicciones y los valores reales.
# 
# MSE (Mean Squared Error): Es el promedio de los cuadrados de las diferencias entre las predicciones y los valores reales. 
# Penaliza más los errores grandes.
# 
# RMSE (Root Mean Squared Error): Es la raíz cuadrada del MSE. Proporciona una medida de la magnitud promedio de los errores.
# 
# R2 (R-squared): También conocido como coeficiente de determinación, indica cuánta variabilidad en los datos es capturada por el modelo. Un valor más cercano a 1 indica un mejor ajuste.
# 
# RMSLE (Root Mean Squared Logarithmic Error): Es similar al RMSE pero se aplica al logaritmo de los valores. Puede ser útil cuando los datos tienen una distribución logarítmica.
# 
# MAPE (Mean Absolute Percentage Error): Es el promedio de los porcentajes absolutos de error entre las predicciones y los valores reales. Mide el error porcentual promedio.
# 
# TT (Sec): Tiempo total en segundos que llevó entrenar el modelo.
# 
# En general, cuando se evalúan modelos de regresión, se busca minimizar métricas como MAE, MSE, RMSE y RMSLE, y maximizar R2. El MAPE también puede ser útil para comprender el error porcentual promedio.
# 
# Observando estos resultados, se puede  determinar qué el  modelo que tiene el mejor rendimiento en función de las métricas es el modelo K Neighbors Regressor tiene el MAE más bajo, lo que indica que tiene el menor error promedio absoluto en comparación con los otros modelos evaluados. Sin embargo, es importante considerar todas las métricas y no solo una para tomar una decisión informada sobre qué modelo usar en tu caso particular.

# In[14]:


regression.evaluate_model(best_model)


# In[15]:


modelo_rf = regression.create_model('rf')
print(modelo_rf)


# In[16]:


# Evaluar el modelo afinado
from pycaret.regression import setup, create_model, tune_model, evaluate_model, plot_model
evaluate_model(modelo_rf)


# In[17]:


##from pycaret.regression import setup, create_model, tune_model, evaluate_model, plot_model

# Configurar el entorno de PyCaret para regresión
setup(data=df, target='TLoss2kAvg')

# Crear un modelo de regresión omp
##modelo_omp = create_model('omp')

# Ajustar manualmente los hiperparámetros
#modelo_rf_tuneado = tune_model(modelo_rf)

# Evaluar el modelo afinado
#evaluate_model(modelo_rf_tuneado)


# In[18]:


from pycaret.regression import *
import shap

# Cargar el conjunto de datos y configurar el entorno de experimento
data = df # Cargar tu conjunto de datos aquí
reg_experiment = setup(data=df, target='TLoss2kAvg')

# Crear un modelo 
reg_model = (modelo_rf)

prediction_holdout = predict_model(reg_model)


# In[19]:


new_data = data.copy().drop('TLoss2kAvg', axis=1)
predictions = predict_model(modelo_rf, data=new_data)


# In[20]:


from pycaret.regression import save_model
save_model(modelo_rf, 'best_pipeline')


# In[21]:


from pycaret.regression import plot_model

# Cargar el modelo desde el archivo
loaded_model = load_model('best_pipeline')

# Visualizar la importancia de las características
plot_model(loaded_model, plot='feature')


# In[22]:


import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

# Obtener curvas de aprendizaje
train_sizes, train_scores, test_scores = learning_curve(loaded_model, X_train, y_train, cv=5)

# Calcular las medias y desviaciones estándar de las puntuaciones de entrenamiento y prueba
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Graficar las curvas de aprendizaje
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Entrenamiento')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Prueba')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.xlabel('Tamaño del Conjunto de Entrenamiento')
plt.ylabel('Puntuación')
plt.title('Curva de Aprendizaje')
plt.legend()
plt.show()


# In[23]:


import matplotlib.pyplot as plt

# Definir el modelo (por ejemplo, final_model)
model = loaded_model

# Obtener predicciones del modelo
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)


# In[24]:


from sklearn.metrics import mean_squared_error, r2_score

# Definir el modelo (por ejemplo, final_model)
model = loaded_model

# Obtener predicciones del modelo en conjuntos de entrenamiento y prueba
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)


# In[25]:


from sklearn.metrics import mean_squared_error, r2_score

# Obtener predicciones del modelo en conjuntos de entrenamiento y prueba
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Calcular MSE y R2 en conjuntos de entrenamiento y prueba
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print(f'MSE en entrenamiento: {mse_train}')
print(f'MSE en prueba: {mse_test}')
print(f'R2 en entrenamiento: {r2_train}')
print(f'R2 en prueba: {r2_test}')


# In[26]:


# Realizar predicciones con el modelo
y_pred_test = model.predict(X_test)

# Imprimir los resultados predictivos
for i, (real, predicho) in enumerate(zip(y_test, y_pred_test)):
    print(f'Observación {i + 1}: Valor Real = {real}, Valor Predicho = {predicho}')


# In[27]:


# Realizar predicciones con el modelo
y_pred_test = model.predict(X_test)

# Calcular los residuos
residuos = y_test - y_pred_test

# Imprimir los resultados predictivos y los residuos
for i, (real, predicho, residuo) in enumerate(zip(y_test, y_pred_test, residuos)):
    print(f'Observación {i + 1}: Valor Real = {real}, Valor Predicho = {predicho}, Residuo = {residuo}')


# In[28]:


import matplotlib.pyplot as plt

# Supongamos que ya tienes 'y_true' (valores reales) y 'y_pred' (valores predichos)
# Reemplaza estos valores con tus propios datos reales y predicciones

# Crear un gráfico de dispersión con una línea de regresión
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_test, color='blue', label='Datos reales vs. Predicciones', alpha=0.4)

# Agregar una línea de regresión
plt.plot( y_test,  y_test, color='red', linestyle='--', linewidth=1, label='Línea de Regresión')

# Etiquetas y título
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones del Modelo')
plt.title('Gráfico de Regresión')

# Mostrar leyenda
plt.legend()

# Mostrar el gráfico
plt.show()


# In[29]:


import matplotlib.pyplot as plt

# Realizar predicciones con el modelo
y_pred_test = model.predict(X_test)

# Calcular los residuos
residuos = y_test - y_pred_test

# Crear un gráfico de residuos
plt.figure(figsize=(8, 6))
plt.scatter(y_pred_test, residuos, color='green', alpha=0.7)

# Etiquetas y título
plt.xlabel('Valores Predichos')
plt.ylabel('Residuos')
plt.title('Gráfico de Residuos vs. Valores Predichos')

# Agregar una línea horizontal en y=0 para referencia
plt.axhline(y=0, color='red', linestyle='--', linewidth=1)

# Mostrar el gráfico
plt.show()


# In[30]:


import matplotlib.pyplot as plt

# Realizar predicciones con el modelo
y_pred_test = model.predict(X_test)

# Calcular los residuos
residuos = y_test - y_pred_test

# Crear un histograma de residuos
plt.figure(figsize=(8, 6))
plt.hist(residuos, bins=20, color='green', alpha=0.7)

# Etiquetas y título
plt.xlabel('Residuos (% de deforetación)')
plt.ylabel('Frecuencia')
plt.title('Histograma de Residuos')

# Mostrar el histograma
plt.show()


# In[31]:


import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# Supongamos que 'datos' son tus datos observados (reemplaza con tus propios datos)
residuos = np.random.normal(y_test - y_pred_test)  # Ejemplo de datos distribuidos normalmente

# Calcular los cuantiles esperados de una distribución normal
cuantiles_esperados = stats.probplot(residuos, dist="norm", plot=plt)

# Etiquetas y título
plt.xlabel('Cuantiles teóricos (Distribución Normal)')
plt.ylabel('Cuantiles de los Datos Observados')
plt.title('Gráfico de Probabilidad Normal-QQ')

# Mostrar el gráfico
plt.show()





# In[50]:


import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# Supongamos que 'y_test' son los valores observados y 'y_pred_test' son las predicciones (reemplaza con tus propios datos)
# Asegúrate de que 'y_test' y 'y_pred_test' tengan las mismas unidades que la variable objetivo

# Calcular los residuos
residuos = y_test - y_pred_test

# Calcular los cuantiles esperados de una distribución normal
cuantiles_esperados = stats.probplot(residuos, dist="norm", plot=plt)

# Etiquetas y título
plt.xlabel('Cuantiles teóricos (Distribución Normal)')
plt.ylabel('Cuantiles de los Residuos (% de deforestación)')
plt.title('Gráfico de Probabilidad Normal-QQ de los Residuos')

# Mostrar el gráfico
plt.show()


# In[32]:


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

# Supongamos que tienes datos 'X' con tres variables predictoras (reemplaza con tus propios datos)
X = df[['idx_count_70-79a', 'idx_count_secun18', 'idx_count_desempl']].values

# Crear un objeto PCA y ajustarlo a los datos
pca = PCA(n_components=3)  # Reducir a 3 componentes principales
X_pca = pca.fit_transform(X)

# Crear un gráfico de PCA Plot con tres componentes principales
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], color='blue', alpha=0.7)

# Etiquetas y título
ax.set_xlabel('Componente Principal 1')
ax.set_ylabel('Componente Principal 2')
ax.set_zlabel('Componente Principal 3')
plt.title('Gráfico de PCA Plot (3 Componentes Principales)')

# Mostrar el gráfico
plt.show()


# In[33]:


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

# Supongamos que tienes datos 'X' con tres variables predictoras (reemplaza con tus propios datos)
X = df[['idx_count_70-79a', 'idx_count_desempl']].values

# Crear un objeto PCA y ajustarlo a los datos con 2 componentes principales
pca = PCA(n_components=2)  # Reducir a 2 componentes principales
X_pca = pca.fit_transform(X)

# Crear un gráfico de PCA Plot con dos componentes principales
plt.figure(figsize=(10, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], color='blue', alpha=0.7)

# Etiquetas y título
plt.xlabel('Población entre 70 y 79 años')
plt.ylabel('Perosnas desempleadas')
plt.title('Gráfico de PCA Plot (2 Componentes Principales)')

# Mostrar el gráfico
plt.show()


# In[34]:


import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Importa la proyección 3D de Matplotlib

# Gráfico de Dispersión 3D
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data['idx_count_70-79a'], data['idx_count_secun18'], data['idx_count_desempl'], c=data['TLoss2kAvg'], cmap='viridis')
ax.set_xlabel('idx_count_70-79a')
ax.set_ylabel('idx_count_secun18')
ax.set_zlabel('idx_count_desempl')
ax.set_title('Gráfico de Dispersión 3D')
plt.show()


# In[35]:


import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Gráfico de Dispersión 3D con colores basados en una categoría
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Especifica la columna de categoría que determinará los colores
category_column = 'TLoss2kAvg'  # Reemplaza 'CATEGORIA' con el nombre de tu columna de categoría

# Define una paleta de colores (cmap) adecuada para tus categorías
cmap = plt.get_cmap('turbo')  # Puedes cambiar 'viridis' a otra paleta de colores

# Dibuja los puntos de dispersión utilizando la columna de categoría para los colores
scatter = ax.scatter(data['idx_count_70-79a'], data['idx_count_secun18'], data['idx_count_desempl'], c=data[category_column], cmap=cmap)

ax.set_xlabel('Población de 70 a 79 años')
ax.set_ylabel('Población con último grado de formación Secundaria')
ax.set_zlabel('Población desempleada')
ax.set_title('Gráfico de Dispersión 3D con Categorías de Colores')

# Agrega una barra de colores para mostrar la correspondencia entre colores y categorías
legend = plt.colorbar(scatter, ax=ax)
legend.set_label('Pérdida de Vegetación')  # Etiqueta de la barra de colores

plt.show()


# In[36]:


import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Gráfico de Dispersión 3D con colores basados en una categoría utilizando solo dos componentes
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Especifica la columna de categoría que determinará los colores
category_column = 'TLoss2kAvg'  # Reemplaza 'CATEGORIA' con el nombre de tu columna de categoría

# Define una paleta de colores (cmap) adecuada para tus categorías
cmap = plt.get_cmap('turbo')  # Puedes cambiar 'viridis' a otra paleta de colores

# Utiliza solo dos componentes principales en lugar de tres
X = data[['idx_count_70-79a', 'idx_count_secun18']].values

# Dibuja los puntos de dispersión utilizando la columna de categoría para los colores
scatter = ax.scatter(X[:, 0], X[:, 1], c=data[category_column], cmap=cmap)

ax.set_xlabel('Población de 70 a 79 años')
ax.set_ylabel('Población con último grado de formación Secundaria')
ax.set_title('Gráfico de Dispersión 3D con Categorías de Colores (2 Componentes)')

# Agrega una barra de colores para mostrar la correspondencia entre colores y categorías
legend = plt.colorbar(scatter, ax=ax)
legend.set_label('Pérdida de Vegetación')  # Etiqueta de la barra de colores

plt.show()


# In[37]:


import seaborn as sns
import matplotlib.pyplot as plt

# Especifica la columna de categoría que determinará los colores
category_column = 'TLoss2kAvg'  # Reemplaza 'CATEGORIA' con el nombre de tu columna de categoría

# Define una paleta de colores (cmap) adecuada para tus categorías
cmap = plt.get_cmap('turbo')  # Puedes cambiar 'viridis' a otra paleta de colores

# Utiliza solo dos componentes principales en lugar de tres
X = data[['idx_count_70-79a', 'idx_count_secun18']].values

# Crea el gráfico de dispersión en 2D
plt.figure(figsize=(10, 8))
scatter = plt.scatter(np.log10 (X[:, 0]+1), np.log10 (X[:, 1]+1), c=data[category_column], cmap=cmap)

# Etiquetas y título
plt.xlabel('Población de 70 a 79 años')
plt.ylabel('Población con último grado de formación Secundaria')
plt.title('Gráfico de Dispersión en 2D con Categorías de Colores')

# Agrega una barra de colores para mostrar la correspondencia entre colores y categorías
legend = plt.colorbar(scatter)
legend.set_label('Pérdida de Vegetación')  # Etiqueta de la barra de colores

plt.show()


# El gráfico de dispersión en 2D con colores basados en la categoría representa una visualización de datos que permite analizar la relación entre dos variables (por ejemplo, 'Población de 70 a 79 años' y 'Población con último grado de formación Secundaria') y, al mismo tiempo, observar cómo se relacionan con una tercera variable ('Pérdida de Vegetación') mediante el uso de colores.
# 
# Los ejes X e Y representan las dos variables que se están comparando, donde 'Población de 70 a 79 años' se encuentra en el eje X y 'Población con último grado de formación Secundaria' en el eje Y. Cada punto en el gráfico corresponde a una combinación de valores de estas dos variables para un punto de datos específico.
# 
# Los colores de los puntos de dispersión se basan en la tercera variable, que en este caso es 'Pérdida de Vegetación'. Cada color representa un rango o nivel diferente de pérdida de vegetación. Por ejemplo, los puntos verdes podrían indicar una baja pérdida de vegetación, mientras que los puntos rojos podrían indicar una alta pérdida.
# 
# La distribución de puntos en el gráfico revela cómo se relacionan las dos variables en los ejes X e Y en función de los colores. Es posible identificar patrones, como si los puntos de un color particular tienden a agruparse en una región específica del gráfico. Esto podría indicar una relación o tendencia entre las variables.
# 
# La capacidad de correlación entre las dos variables principales se evalúa observando cómo se distribuyen los puntos en el gráfico. Por ejemplo, si se nota que los puntos tienden a agruparse en una dirección específica, esto podría sugerir una correlación positiva o negativa entre las dos variables.
# 
# Además, la información de color permite considerar cómo la pérdida de vegetación está relacionada con las dos variables en los ejes X e Y. Los puntos de un color particular pueden asociarse con ciertos rangos de valores en los ejes X e Y.
# 

# In[38]:


import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Supongamos que tienes tus datos en 'data' y 'y', y deseas ajustar un modelo de regresión lineal
modelo = LinearRegression()

# Utiliza solo las variables 'idx_count_secun18' y 'idx_count_desempl'
X = data[['idx_count_secun18', 'idx_count_desempl']].values
y = data['TLoss2kAvg'].values

modelo.fit(X, y)  # Ajusta el modelo de regresión lineal

# Gráfico de Dispersión 2D con línea de regresión
plt.figure(figsize=(10, 8))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='turbo')  # Colores basados en 'TLoss2kAvg'
plt.xlabel('Población con último grado de formación Secundaria')
plt.ylabel('Población desempleada')
plt.title('Gráfico de Dispersión 2D con Línea de Regresión')

# Agrega la línea de regresión
xmin, xmax = X[:, 0].min(), X[:, 0].max()
ymin, ymax = X[:, 1].min(), X[:, 1].max()
xx, yy = np.meshgrid(np.linspace(xmin, xmax, 100), np.linspace(ymin, ymax, 100))
Z = modelo.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.contour(xx, yy, Z, colors='green', linewidths=3)

# Agrega una barra de colores para mostrar la correspondencia entre colores y 'TLoss2kAvg'
legend = plt.colorbar()
legend.set_label('Pérdida de Vegetación')  # Etiqueta de la barra de colores

plt.show()


# In[39]:


import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

# Obtener curvas de aprendizaje
train_sizes, train_scores, test_scores = learning_curve(modelo_rf, X_train, y_train, cv=5)

# Calcular las medias y desviaciones estándar de las puntuaciones de entrenamiento y prueba
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Graficar las curvas de aprendizaje
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Entrenamiento')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Prueba')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.xlabel('Tamaño del Conjunto de Entrenamiento')
plt.ylabel('Puntuación')
plt.title('Curva de Aprendizaje')
plt.legend()
plt.show()


# In[40]:


import seaborn as sns
import matplotlib.pyplot as plt

# Especifica la columna de categoría que determinará los colores
category_column = 'TLoss2kAvg'  # Reemplaza 'CATEGORIA' con el nombre de tu columna de categoría

# Define una paleta de colores (cmap) adecuada para tus categorías
cmap = plt.get_cmap('RdYlGn')  # Puedes cambiar 'viridis' a otra paleta de colores

# Utiliza solo dos componentes principales en lugar de tres
X = data[['idx_count_desempl', 'idx_count_secun18']].values

# Crea el gráfico de dispersión en 2D
plt.figure(figsize=(10, 8))
scatter = plt.scatter(np.log10 (X[:, 0]+1), np.log10 (X[:, 1]+1), c=data[category_column], cmap=cmap)

# Etiquetas y título
plt.xlabel('Población desempleada')
plt.ylabel('Población con último grado de formación Secundaria')
plt.title('Gráfico de Dispersión en 2D con Categorías de Colores')

# Agrega una barra de colores para mostrar la correspondencia entre colores y categorías
legend = plt.colorbar(scatter)
legend.set_label('Pérdida de Vegetación')  # Etiqueta de la barra de colores

plt.show()


# In[41]:


from pycaret.regression import *

# Configura tu entorno de experimento
exp = setup(data, target='TLoss2kAvg', session_id=123)

# Crea un modelo de Random Forest para regresión y ajusta sus hiperparámetros
modelo_rf_50 = create_model('rf', n_estimators=50, max_depth=6)  # Ejemplo de ajuste de hiperparámetros

# Evalúa el modelo ajustado
evaluate_model(modelo_rf_50)




# In[42]:


import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

# Obtener curvas de aprendizaje
train_sizes, train_scores, test_scores = learning_curve(modelo_rf_50, X_train, y_train, cv=5)

# Calcular las medias y desviaciones estándar de las puntuaciones de entrenamiento y prueba
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Graficar las curvas de aprendizaje
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Entrenamiento')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Prueba')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.xlabel('Tamaño del Conjunto de Entrenamiento')
plt.ylabel('Puntuación')
plt.title('Curva de Aprendizaje')
plt.legend()
plt.show()


# In[43]:


from pycaret.regression import *

# Configura tu entorno de experimento
exp = setup(data, target='TLoss2kAvg', session_id=123)

# Crea un modelo de Random Forest para regresión y ajusta sus hiperparámetros
modelo_rf_80 = create_model('rf', n_estimators=80, max_depth=6)  # Ejemplo de ajuste de hiperparámetros

# Evalúa el modelo ajustado
evaluate_model(modelo_rf_80)


# In[44]:


import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

# Obtener curvas de aprendizaje
train_sizes, train_scores, test_scores = learning_curve(modelo_rf_80, X_train, y_train, cv=10)

# Calcular las medias y desviaciones estándar de las puntuaciones de entrenamiento y prueba
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Graficar las curvas de aprendizaje
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Entrenamiento')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Prueba')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.xlabel('Tamaño del Conjunto de Entrenamiento')
plt.ylabel('Puntuación')
plt.title('Curva de Aprendizaje')
plt.legend()
plt.show()


# In[45]:


from pycaret.regression import *

# Configura tu entorno de experimento
exp = setup(data, target='TLoss2kAvg', session_id=123)

# Crea un modelo de Random Forest para regresión y ajusta sus hiperparámetros
modelo_rf_60 = create_model('rf', n_estimators=60, max_depth=3)  # Ejemplo de ajuste de hiperparámetros

# Evalúa el modelo ajustado
evaluate_model(modelo_rf_60)


# In[46]:


import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

# Obtener curvas de aprendizaje
train_sizes, train_scores, test_scores = learning_curve(modelo_rf_60, X_train, y_train, cv=5)

# Calcular las medias y desviaciones estándar de las puntuaciones de entrenamiento y prueba
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Graficar las curvas de aprendizaje
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Entrenamiento')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Prueba')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.xlabel('Tamaño del Conjunto de Entrenamiento')
plt.ylabel('Puntuación')
plt.title('Curva de Aprendizaje')
plt.legend()
plt.show()


# In[47]:


from pycaret.regression import *

# Configura tu entorno de experimento
exp = setup(data, target='TLoss2kAvg', session_id=123)

# Crea un modelo de Random Forest para regresión y ajusta sus hiperparámetros
modelo_rf_801 = create_model('rf', n_estimators=80, max_depth=15)  # Ejemplo de ajuste de hiperparámetros

# Evalúa el modelo ajustado
evaluate_model(modelo_rf_801)


# In[48]:


import matplotlib.pyplot as plt

# Valores de las métricas
metricas = ['MSE en entrenamiento', 'MSE en prueba', 'R2 en entrenamiento', 'R2 en prueba']
valores = [mse_train, mse_test, r2_train, r2_test]

# Crear el gráfico de barras
plt.figure(figsize=(8, 6))
plt.bar(metricas, valores, color=['blue', 'red', 'blue', 'red'])
plt.xlabel('Métrica')
plt.ylabel('Valor')
plt.title('Métricas de Rendimiento del Modelo')
plt.ylim(min(valores) - 0.1, max(valores) + 0.1)  # Ajustar el rango del eje y
plt.xticks(rotation=45)  # Rotar etiquetas en el eje x para mayor legibilidad

# Mostrar los valores en las barras
for i, v in enumerate(valores):
    plt.text(i, v + 0.01, str(round(v, 2)), ha='center', va='bottom')

plt.tight_layout()
plt.show()


# In[49]:


## Determinación de estimadores usando el método R2 Vs Estimadores

from pycaret.regression import *

# Configura tu entorno de experimento en PyCaret
exp = setup(data, target='TLoss2kAvg', session_id=123)

# Crea un modelo RandomForestRegressor con un valor inicial de n_estimators
modelo_rf = create_model('rf', n_estimators=10)  # Reemplaza 10 con tu valor inicial deseado

# Evalúa la evolución del R² en función del número de estimadores
train_scores = []
test_scores = []
estimator_range = range(1, 150, 5)

for n_estimators in estimator_range:
    modelo_rf.n_estimators = n_estimators
    modelo_rf.fit(X_train, y_train)
    train_scores.append(modelo_rf.score(X_train, y_train))
    test_scores.append(modelo_rf.score(X_test, y_test))

# Gráfico con la evolución del R²
plt.figure(figsize=(10, 6))
plt.plot(estimator_range, train_scores, label='R² en Entrenamiento')
plt.plot(estimator_range, test_scores, label='R² en Prueba')
plt.xlabel('Número de Estimadores')
plt.ylabel('R²')
plt.title('Evolución de R² vs. Número de Estimadores')
plt.legend()
plt.grid(True)
plt.show()

print(f"Valor óptimo de n_estimators: {estimator_range[np.argmax(test_scores)]}")



# In[ ]:


from pycaret.regression import *

# Configura tu entorno de experimento en PyCaret
exp = setup(data, target='TLoss2kAvg', session_id=123)

# Crea un modelo RandomForestRegressor con un valor inicial de max_depth
modelo_rf = create_model('rf', max_depth=100)  # Reemplaza 10 con tu valor inicial deseado

# Evalúa la evolución del R² en función de la profundidad máxima del árbol
train_scores = []
test_scores = []
depth_range = range(1, 21)  # Por ejemplo, evalúa de 1 a 20 niveles de profundidad

for max_depth in depth_range:
    modelo_rf.max_depth = max_depth
    modelo_rf.fit(X_train, y_train)
    train_scores.append(modelo_rf.score(X_train, y_train))
    test_scores.append(modelo_rf.score(X_test, y_test))

# Gráfico con la evolución del R²
plt.figure(figsize=(10, 6))
plt.plot(depth_range, train_scores, label='R² en Entrenamiento')
plt.plot(depth_range, test_scores, label='R² en Prueba')
plt.xlabel('Profundidad Máxima del Árbol')
plt.ylabel('R²')
plt.title('Evolución de R² vs. Profundidad Máxima del Árbol')
plt.legend()
plt.grid(True)
plt.show()

print(f"Valor óptimo de max_depth: {depth_range[np.argmax(test_scores)]}")


# In[ ]:


from pycaret.regression import *

# Configura tu entorno de experimento
exp = setup(data, target='TLoss2kAvg', session_id=123)

# Crea un modelo de Random Forest para regresión y ajusta sus hiperparámetros
modelo_rf_21 = create_model('rf', n_estimators=21, max_depth=8)  # Ejemplo de ajuste de hiperparámetros

# Evalúa el modelo ajustado
evaluate_model(modelo_rf_21)


# In[ ]:


import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

# Obtener curvas de aprendizaje
train_sizes, train_scores, test_scores = learning_curve(modelo_rf, X_train, y_train, cv=2)

# Calcular las medias y desviaciones estándar de las puntuaciones de entrenamiento y prueba
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Graficar las curvas de aprendizaje
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Entrenamiento')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Prueba')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.xlabel('Tamaño del Conjunto de Entrenamiento')
plt.ylabel('Puntuación')
plt.title('Curva de Aprendizaje')
plt.legend()
plt.show()


# ## Pruebas con modelo lightgbm

# In[ ]:


from pycaret.regression import *

# Configura tu entorno de experimento
exp = setup(data, target='TLoss2kAvg', session_id=123)

# Crea un modelo de Random Forest para regresión y ajusta sus hiperparámetros
modelo_lightgbm = create_model('lightgbm')  # Ejemplo de ajuste de hiperparámetros

# Evalúa el modelo ajustado
evaluate_model(modelo_lightgbm)


# In[ ]:


import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

# Obtener curvas de aprendizaje
train_sizes, train_scores, test_scores = learning_curve(modelo_lightgbm, X_train, y_train, cv=5)

# Calcular las medias y desviaciones estándar de las puntuaciones de entrenamiento y prueba
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Graficar las curvas de aprendizaje
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Entrenamiento')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Prueba')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.xlabel('Tamaño del Conjunto de Entrenamiento')
plt.ylabel('Puntuación')
plt.title('Curva de Aprendizaje')
plt.legend()
plt.show()


# In[ ]:


## Determinación de estimadores usando el método R2 Vs Estimadores

from pycaret.regression import *

# Configura tu entorno de experimento en PyCaret
exp = setup(data, target='TLoss2kAvg', session_id=123)

# Crea un modelo RandomForestRegressor con un valor inicial de n_estimators
modelo_rf = create_model('lightgbm', n_estimators=10)  # Reemplaza 10 con tu valor inicial deseado

# Evalúa la evolución del R² en función del número de estimadores
train_scores = []
test_scores = []
estimator_range = range(1, 150, 5)

for n_estimators in estimator_range:
    modelo_rf.n_estimators = n_estimators
    modelo_rf.fit(X_train, y_train)
    train_scores.append(modelo_rf.score(X_train, y_train))
    test_scores.append(modelo_rf.score(X_test, y_test))

# Gráfico con la evolución del R²
plt.figure(figsize=(10, 6))
plt.plot(estimator_range, train_scores, label='R² en Entrenamiento')
plt.plot(estimator_range, test_scores, label='R² en Prueba')
plt.xlabel('Número de Estimadores')
plt.ylabel('R²')
plt.title('Evolución de R² vs. Número de Estimadores')
plt.legend()
plt.grid(True)
plt.show()

print(f"Valor óptimo de n_estimators: {estimator_range[np.argmax(test_scores)]}")


# In[ ]:


from pycaret.regression import *

# Configura tu entorno de experimento en PyCaret
exp = setup(data, target='TLoss2kAvg', session_id=123)

# Crea un modelo RandomForestRegressor con un valor inicial de max_depth
modelo_rf = create_model('lightgbm', max_depth=10)  # Reemplaza 10 con tu valor inicial deseado

# Evalúa la evolución del R² en función de la profundidad máxima del árbol
train_scores = []
test_scores = []
depth_range = range(1, 21)  # Por ejemplo, evalúa de 1 a 20 niveles de profundidad

for max_depth in depth_range:
    modelo_rf.max_depth = max_depth
    modelo_rf.fit(X_train, y_train)
    train_scores.append(modelo_rf.score(X_train, y_train))
    test_scores.append(modelo_rf.score(X_test, y_test))

# Gráfico con la evolución del R²
plt.figure(figsize=(10, 6))
plt.plot(depth_range, train_scores, label='R² en Entrenamiento')
plt.plot(depth_range, test_scores, label='R² en Prueba')
plt.xlabel('Profundidad Máxima del Árbol')
plt.ylabel('R²')
plt.title('Evolución de R² vs. Profundidad Máxima del Árbol')
plt.legend()
plt.grid(True)
plt.show()

print(f"Valor óptimo de max_depth: {depth_range[np.argmax(test_scores)]}")


# In[ ]:


from pycaret.regression import *

# Configura tu entorno de experimento
exp = setup(data, target='TLoss2kAvg', session_id=123)

# Crea un modelo de Random Forest para regresión y ajusta sus hiperparámetros
modelo_lightgbm_80 = create_model('lightgbm', n_estimators=146, max_depth=18)  # Ejemplo de ajuste de hiperparámetros

# Evalúa el modelo ajustado
evaluate_model(modelo_lightgbm_80)


# In[ ]:


import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

# Obtener curvas de aprendizaje
train_sizes, train_scores, test_scores = learning_curve(modelo_lightgbm_80, X_train, y_train, cv=5)

# Calcular las medias y desviaciones estándar de las puntuaciones de entrenamiento y prueba
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Graficar las curvas de aprendizaje
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Entrenamiento')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Prueba')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.xlabel('Tamaño del Conjunto de Entrenamiento')
plt.ylabel('Puntuación')
plt.title('Curva de Aprendizaje')
plt.legend()
plt.show()


# In[ ]:


from pycaret.regression import *

# Configura tu entorno de experimento
exp = setup(data, target='TLoss2kAvg', session_id=123)

# Crea un modelo de Random Forest para regresión y ajusta sus hiperparámetros
modelo_lightgbm_50 = create_model('lightgbm', n_estimators=50, max_depth=10)  # Ejemplo de ajuste de hiperparámetros

# Evalúa el modelo ajustado
evaluate_model(modelo_lightgbm_50)


# In[ ]:


from pycaret.regression import *

# Configura tu entorno de experimento
exp = setup(data, target='TLoss2kAvg', session_id=123)

# Crea un modelo de Random Forest para regresión y ajusta sus hiperparámetros
modelo_lightgbm_505 = create_model('lightgbm', n_estimators=50, max_depth=3)  # Ejemplo de ajuste de hiperparámetros

# Evalúa el modelo ajustado
evaluate_model(modelo_lightgbm_505)


# In[ ]:


import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

# Obtener curvas de aprendizaje
train_sizes, train_scores, test_scores = learning_curve(modelo_lightgbm_505, X_train, y_train, cv=5)

# Calcular las medias y desviaciones estándar de las puntuaciones de entrenamiento y prueba
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Graficar las curvas de aprendizaje
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Entrenamiento')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Prueba')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.xlabel('Tamaño del Conjunto de Entrenamiento')
plt.ylabel('Puntuación')
plt.title('Curva de Aprendizaje')
plt.legend()
plt.show()


# In[ ]:


predicted_values = modelo_rf.predict(X_test)

print(predicted_values)


# In[ ]:


#import seaborn as sns
#sns.pairplot(data, hue='TLoss2kAvg')


# In[ ]:


from sklearn.linear_model import LinearRegression

# Supongamos que tienes tus datos en 'X' e 'y', y deseas ajustar un modelo de regresión lineal
modelo = LinearRegression()
modelo.fit(X, y)  # Donde X son tus variables predictoras y 'y' es tu variable objetivo

# Define la lista de variables predictoras que deseas incluir en el gráfico
features = ['idx_count_desempl', 'idx_count_secun18', 'otra_variable_1', 'otra_variable_2']  # Eliminamos 'otra_variable_3'

# Obtener los coeficientes del modelo para las variables seleccionadas
coefficients = [modelo.coef_[features.index(feature)] for feature in features]

# Crear un gráfico de barras para mostrar los coeficientes de las variables seleccionadas
plt.figure(figsize=(12, 6))  # Puedes ajustar el tamaño del gráfico según sea necesario
plt.bar(features, coefficients)
plt.xlabel('Variables Predictoras')
plt.ylabel('Coeficiente de Regresión')
plt.title('Coeficientes de Regresión para Variables en el Modelo Lineal')
plt.xticks(rotation=45)  # Rotar etiquetas en el eje x si son largas
plt.show()


# Supongamos que tienes un modelo llamado 'modelo' entrenado previamente
# y 'features' es una lista de nombres de variables predictoras

#import matplotlib.pyplot as plt

# Obtener la importancia de las variables desde el modelo
#importances = modelo_rf_801.feature_importances_

# Crear un gráfico de barras para mostrar la importancia de las variables
#plt.figure(figsize=(10, 6))
#plt.bar(range(len(features)), importances, tick_label=features)
#plt.xlabel('idx_count_desempl', 'idx_count_secun18')
#plt.ylabel('Importancia')
#plt.title('Importancia de Variables en la Predicción de TLoss2kAvg')
#plt.xticks(rotation=90)
#plt.show()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

# Especifica la columna de categoría que determinará los colores
category_column = 'TLoss2kAvg'  # Reemplaza 'CATEGORIA' con el nombre de tu columna de categoría

# Define una paleta de colores (cmap) adecuada para tus categorías
cmap = plt.get_cmap('turbo')  # Puedes cambiar 'viridis' a otra paleta de colores

# Utiliza solo dos componentes principales en lugar de tres
X = data[['idx_count_70-79a', 'idx_count_secun18']].values

# Crea el gráfico de dispersión en 2D
plt.figure(figsize=(10, 8))
scatter = plt.scatter(np.log10(X[:, 0] + 1), np.log10(X[:, 1] + 1), c=data[category_column], cmap=cmap)

# Etiquetas y título
plt.xlabel('Población de 70 a 79 años')
plt.ylabel('Población con último grado de formación Secundaria')
plt.title('Gráfico de Dispersión en 2D con Categorías de Colores')

# Agrega una barra de colores para mostrar la correspondencia entre colores y categorías
legend = plt.colorbar(scatter)
legend.set_label('Pérdida de Vegetación')  # Etiqueta de la barra de colores

plt.show()


# In[ ]:


from sklearn.model_selection import train_test_split
from pycaret import regression
from sklearn.linear_model import Ridge  # Asegúrate de importar Ridge

seed = 123

# Paso 1: Preparar tabla 
# (ya se hizo se llama df)

# Paso 2: Separar en X, y
y_obj = 'TLoss2kAvg' 
X = df.drop(columns=[y_obj])
y = df[y_obj]

# Paso 3: Separar en entrenamiento y validación
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
Xy_train = pd.concat([X_train, y_train], axis=1)
Xy_test = pd.concat([X_test, y_test], axis=1)

# Corrige la configuración de la regresión
regression.setup(data=Xy_train, target=y_obj)

# Crea y ajusta un modelo de Ridge
modelo_ridge = Ridge(alpha=1.0)  # Puedes ajustar el valor de alpha según sea necesario

# Ajusta el modelo Ridge a tus datos de entrenamiento
modelo_ridge.fit(X_train, y_train)

# Haz predicciones con el modelo Ridge
y_pred_ridge = modelo_ridge.predict(X_test)



# In[ ]:


from pycaret.regression import *

# Configura tu entorno de experimento
exp = setup(data, target='TLoss2kAvg', session_id=123)

# Crea un modelo de Random Forest para regresión y ajusta sus hiperparámetros
modelo_rf_R2 = create_model('rf', n_estimators=146, max_depth=14)  # Ejemplo de ajuste de hiperparámetros

# Evalúa el modelo ajustado
evaluate_model(modelo_rf_R2)


# In[ ]:


from pycaret.regression import *

# Configura tu entorno de experimento
exp = setup(data, target='TLoss2kAvg', session_id=123)

# Crea un modelo de Random Forest para regresión y ajusta sus hiperparámetros
modelo_rf = create_model('rf', n_estimators=114, max_depth=14)  # Ejemplo de ajuste de hiperparámetros

# Evalúa el modelo ajustado
evaluate_model(modelo_rf)

summary = pull()
print(summary)



# In[ ]:


import numpy as np

# Obtén las predicciones del modelo en los conjuntos de entrenamiento y prueba
y_pred_train = modelo_rf.predict(X_train)
y_pred_test = modelo_rf.predict(X_test)

# Calcula los residuos restando las predicciones de los valores reales
residuos_train = np.array(y_train) - y_pred_train
residuos_test = np.array(y_test) - y_pred_test

from scipy import stats

# residuos es la lista o arreglo que contiene los residuos del modelo RF
stat, p_value = stats.shapiro(residuos)

# Comprobar el valor p
if p_value > 0.5:
    print("Los residuos siguen una distribución normal (no se puede rechazar H0)")
else:
    print("Los residuos no siguen una distribución normal (se rechaza H0)")


# In[51]:


from pycaret.regression import *
from sklearn.linear_model import LinearRegression, RidgeCV  # Importa RidgeCV aquí

# Configura tu entorno de experimento
exp = setup(data, target='TLoss2kAvg', session_id=123)

# Crea modelos individuales (Random Forest y Ridge)
modelo_rf = create_model('rf', n_estimators=146, max_depth=14)
modelo_ridge = create_model('ridge')  # Ajusta los hiperparámetros según sea necesario

# Combina los modelos en un ensamblaje personalizado
from sklearn.ensemble import StackingRegressor

# Define el ensamblaje personalizado con Random Forest y Ridge como modelos base
ensamblaje_personalizado = StackingRegressor(
    estimators=[('rf', modelo_rf), ('ridge', modelo_ridge)],
    final_estimator=RidgeCV()  # Puedes ajustar los hiperparámetros del final_estimator
)

# Ajusta el ensamblaje personalizado a tus datos de entrenamiento
ensamblaje_personalizado.fit(X_train, y_train)

# Realiza predicciones con el ensamblaje personalizado
y_pred_ensamblaje = ensamblaje_personalizado.predict(X_test)

# Define el ensamblaje personalizado con Random Forest y Ridge como modelos base
ensamblaje_personalizado = StackingRegressor(
    estimators=[('rf', modelo_rf), ('ridge', modelo_ridge)],
    final_estimator=RidgeCV()  # Utiliza RidgeCV desde sklearn.linear_model
)


# In[ ]:


from sklearn.metrics import mean_squared_error, r2_score

# Calcula el MSE y R^2 en el conjunto de prueba
mse = mean_squared_error(y_test, y_pred_ensamblaje)
r2 = r2_score(y_test, y_pred_ensamblaje)

print(f'MSE: {mse}')
print(f'R^2: {r2}')


# In[ ]:


# Variables predictoras más importantes para el modelo Random Forest
importances_rf = modelo_rf.feature_importances_

# Variables predictoras más importantes para el modelo Ridge (coeficientes)
coef_ridge = modelo_ridge.coef_

# Calcula el promedio de importancias de características de los modelos base
average_importance = (importances_rf + coef_ridge) / 2

# Crear un gráfico de barras para mostrar las importancias promedio
plt.figure(figsize=(10, 6))
plt.bar(X_train.columns, average_importance)
plt.xlabel('Variables Predictoras')
plt.ylabel('Importancia Promedio')
plt.title('Importancia Promedio de Variables Predictoras en el Ensamble')
plt.xticks(rotation=90)
plt.show()


# In[54]:


importances_rf = modelo_rf.feature_importances_

coef_ridge = modelo_ridge.coef_

average_importance = (importances_rf + coef_ridge) / 2

# Crear una lista de tuplas (variable, importancia) para ordenarlas
importance_tuples = [(variable, importance) for variable, importance in zip(X_train.columns, average_importance)]

# Ordenar la lista de tuplas por importancia (de mayor a menor)
importance_tuples.sort(key=lambda x: x[1], reverse=True)

# Extraer las variables ordenadas y sus importancias
sorted_variables, sorted_importances = zip(*importance_tuples)

# Crear un gráfico de barras para mostrar las importancias promedio ordenadas
plt.figure(figsize=(10, 6))
plt.bar(sorted_variables, sorted_importances)
plt.xlabel('Variables Predictoras')
plt.ylabel('Importancia Promedio')
plt.title('Importancia Promedio de Variables Predictoras en el Ensamble')
plt.xticks(rotation=90)
plt.show()



# In[55]:


import matplotlib.pyplot as plt

# Define colores para las barras positivas y negativas
colors = ['green' if imp >= 0 else 'red' for imp in sorted_importances]

# Crear un gráfico de barras personalizado
plt.figure(figsize=(10, 6))
bars = plt.bar(sorted_variables, sorted_importances, color=colors)
plt.xlabel('Variables Predictoras')
plt.ylabel('Importancia Promedio')
plt.title('Importancia Promedio de Variables Predictoras en el Ensamble')
plt.xticks(rotation=90)

# Agregar una barra general para indicar "Positivo" o "Negativo"
general_bar_color = 'red' if all(imp >= 0 for imp in sorted_importances) else 'green'
general_bar_height = max(sorted_importances) * 1.1  # Ajusta la altura de la barra general
general_bar_label = 'Aumento de Deforestación' if general_bar_color == 'green' else 'Disminución de Deforestación'

# Agregar la leyenda personalizada
plt.legend(handles=[
    plt.Rectangle((0, 0), 1, 1, fc='green'),
    plt.Rectangle((0, 0), 1, 1, fc='red')
], labels=['Aumento de Deforestación', 'Disminución de Deforestación'], loc='upper right')

plt.show()


# In[ ]:




