#### Segmentación de Clientes - Tarjetas de Crédito

### 1. Análisis Exploratorio de los Datos (EDA)


## 1.1 Carga de Librerías

## Cargamos las librerías a utilizar
library(dplyr)
library(caret)
library(RANN)
library(fpc)
library(vegan)
library(factoextra)

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
set.seed(13)


## 3.1 Determinación del número de clusters

# Dibujamos el dendrograma
transformed_hclust <- dist(as.matrix(transformed))
plot(hclust(transformed_hclust))

hierarchical <- dist(transformed, method = "euclidean")
hfit <- hclust(hierarchical, method = "ward.D" )
plot(hfit)

grps <- cutree(hfit, k = 3)
rect.hclust(hfit, k = 3, border = "green")

# Aplicamos el método del codo
wss <- (nrow(transformed)-1) * sum(apply(transformed, 2, var))
for (i in 2:10)
  wss[i] <- sum(kmeans(transformed, centers = i)$withinss)
plot(1:10, wss, type = "b", main = "Método del Codo", xlab = "Clusters", ylab = "Coste")

# Aplicamos el método de la silueta
n_cluster <- pamk(transformed)
n_cluster
cat("Número de clusters con mejor score silhoutte: ", n_cluster$nc)
plot(pam(transformed, n_cluster$nc))

# Aplicamos el método GAP
clusGap(transformed, kmeans, 10, B = 100, verbose = interactive())

# Aplicamos el índice calinski harabasz
cal_har <- cascadeKM(transformed, 1, 10, iter = 100)
plot(cal_har, sortg = TRUE, grpmts.plot = TRUE)
n_clust <- as.numeric(which.max(cal_har$results[2,]))
cat("Número de clusters según el índice calinski harabasz: ", n_clust)


## 3.2 Algoritmo K Means

# Ajustamos el algoritmo KMeans con n_clusters
km <- kmeans(transformed, 3)
km
attributes(km)

# Graficamos en 2D
clusplot(transformed, km$cluster, main = "Representación 2D", shade = TRUE, labels = 2, lines = 0)
fviz_cluster(km, data = transformed)

# Agregamos el segmento al dataframe original 
segmento <- cbind(km$cluster)
credit_card_segmented <- merge(x = credit_card, y = segmento)
