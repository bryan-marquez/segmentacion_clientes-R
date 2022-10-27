#### Segmentación de Clientes - Tarjetas de Crédito

### 1. Análisis Exploratorio de los Datos (EDA)


## 1.1 Carga de Librerías

## Cargamos las librerías a utilizar
library(dplyr)
library(caret)
library(RANN)


## 1.2 Carga de Datos

# Cargamos los datos
archivo <- "/cloud/project/data/CreditCard.csv"
credit_card <- read.csv(archivo)
head(credit_card)

# Mostramos el número de filas y columnas
dim(credit_card)

# Mostramos los tipos de datos
str(credit_card)
sapply(credit_card, class)


## 1.3 Estadística Descriptiva

# Resumimos las variables numéricas
summary(credit_card)

# Obtenemos la desviación estándar
sapply(credit_card, sd)

# Elaboramos la matriz de correlación
correlacion <- cor(select(credit_card, -CUST_ID))
correlacion


## 1.4 Visualización de las Variables Numéricas

# Trazamos los histogramas
hist(select(credit_card, -CUST_ID))
hist


### 2. Preparación de los Datos


## 2.1 Limpieza de Datos

# Eliminamos la variable ID
dataframe <- select(credit_card, -CUST_ID) # quitamos la variable identidad
dim(dataframe)

# Mostramos la suma de valores nulos
sapply(credit_card, function(x) sum(is.na(x)))

# Mostramos la suma de valores cero
sapply(credit_card, function(x) sum(x == 0, na.rm = TRUE))

# Imputamos los valores nulos con "knnImpute" y estandarizamos los datos númericos con "center" y "scale"
imputer <- preProcess(credit_card, method = c('knnImpute'))
imputer

# Transformamos los valores nulos y transformamos los datos númericos a media = 0 y sd = 1
transformed <- predict(imputer, dataframe)
head(transformed)
sapply(transformed, function(x) sum(is.na(x)))


### 3. Modelado


## 3.1 Determinación del número de clusters