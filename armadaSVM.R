#install.packages("ggplot2")
library(caret)
library(kernlab)
library(ggplot2)
library(e1071)
library(readr)
library(dplyr) 
library(gmodels)

#--Lectura del dataset
armada <- read_table("RFiles/armada.dat",col_names = FALSE)

#--Preparaciónn y limpieza de los datos
names(armada) <- c("Battle","Year","Portuguese_Ships", "Dutch_Ships","English_Ships",
                   "Proportion", "Spain_Participation", "Portuguese_Result")
View(armada)

armada <- select(armada,Portuguese_Ships,Dutch_Ships,English_Ships,Spain_Participation,Portuguese_Result)
#Todos las columnas a tratar deben ser factores
armada$Portuguese_Result <- factor(armada$Portuguese_Result, levels = c(-1,0,1), labels = c("Derrota","Empate","Victoria"))

#--Creación conjuntos de trainning y test
set.seed(122)
indices <- createDataPartition(armada$Portuguese_Result, p = .75, list = FALSE)
armada_train = armada[indices, ]
armada_test = armada[-indices, ] 

View(armada_train)
View(armada_test)

#Explotramos como se distribuyen las muestras a partir de  nuestras variables predictoras
featurePlot(armada_train[,-5], y = armada_train$Portuguese_Result, plot = 'ellipse')

#--Entrenamos nuestro modelo, para este caso no declaramos el tipo de kernel y dejamos que lo escoja la función; 
modelo = svm(formula = Portuguese_Result ~ .,
                data = armada_train,
                type = 'C-classification',
                cost = 10)
modelo

#Función tune con un kernel radial 
tuned <- tune(e1071::svm, Portuguese_Result ~., data = armada_train, kernel = "radial", ranges = list(cost=c(0.001,0.01,.1,1,10,100)))
#Analisis de costo
summary(tuned)

#--Predecir los datos de test
predictionResult = predict(modelo, newdata = armada_test[-5]) #sin considerar la variable dependiente 
predictionResult

#--Creación de matriz de confusión
confusionMatrix(predictionResult, armada_test$Portuguese_Result) 

#--Se realiza un ajuste 
train10CV <- trainControl(method = "cv",      # método de validación cruzada
                            number = 10,      # diez submuestras
                            classProbs = TRUE)  

#--Entrenamos una SVM con un metodo SVM radial
svmRBF <- train(Portuguese_Result ~ ., 
                data = armada_train, 
                method = "svmRadial", 
                trControl = train10CV,   # valor óptimo 
                preProc = c("center", "scale"))
svmRBF

#--Predecir los datos de test
predictionResult = predict(svmRBF, newdata = armada_test[-5]) #sin considerar la variable dependiente 
predictionResult

#--Creación de matriz de confusión
confusionMatrix(predictionResult, armada_test$Portuguese_Result) 

# Comparación de la SVM con otros modelos
knnModel <- train(Portuguese_Result ~ ., data = armada_train, method = "knn", trControl = train10CV, preProc = c("center", "scale"))
ldaModel <- train(Portuguese_Result ~ ., data = armada_train, method = "lda", trControl = train10CV, preProc = c("center", "scale"))
svmLinear <- train(Portuguese_Result ~ ., data = armada_train, method = "svmLinear", trControl = train10CV, preProc = c("center", "scale"))

results <- resamples(list(svmRBF = svmRBF, svmLinear = svmLinear, kNN = knnModel, LDA = ldaModel))
summary(results)

#--Ploteo de comparación entre modelos
dotplot(results)

#--Plotear el modelo 
plot(modelo, armada_test, Dutch_Ships ~ Portuguese_Ships,slice = list(English_Ships = 1, Spain_Participation = 0)) 
