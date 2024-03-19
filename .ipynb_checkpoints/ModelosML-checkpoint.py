import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

class modelos_lineales: 
       
    
    def regresion_lineal(self, x, y, epochs, imprimir_error_cada, learning_rate):

        models = {}

        ones_like_x= np.ones_like(x.reshape(1,len(x)).T) # creando el arreglo de 1 utilizando la forma del vector x redefinido en 2D y colocandolo vertical utilizando la Transpuesta

        x_matrix = np.concatenate((x.reshape(1,len(x)).T,ones_like_x), axis=1) 
        b0=0
        b1=0
        betas = [[b1],
                 [b0]]


        for i in range(epochs):

            y_estim, cost_error, betas  = self.epoch(x_matrix, betas,y,learning_rate)

            if i%imprimir_error_cada == 0 :
                print(f'Epoch No.{i+1} error de costo = {cost_error}')

            if i == 0:
                error_history = np.array([[cost_error]])
            else:
                error_history = np.concatenate((error_history, np.array([[cost_error]])),axis=0)

            models[i]= {'betas':betas, 'cost':float(error_history[i])}

        return models, error_history


    def epoch(self, x_matrix, betas, y, learning_rate):

        y_estim = np.matmul(x_matrix,betas)

        dif_y = y.reshape(-1,1) - y_estim

        cost_error = (1/(2*(len(y_estim))))*np.sum((dif_y)**2)

        updated_betas = self.gradient_descent(x_matrix, betas, y, y_estim, dif_y,learning_rate)


        return y_estim, cost_error, updated_betas
    
    
    def gradient_descent(self, x_matrix,betas, y, y_estim, dif_y,learning_rate):


        gradiente_array = (1/(len(y_estim))) * np.matmul(dif_y.reshape(1,-1),x_matrix)
        betas = betas - learning_rate * gradiente_array.reshape(-1,1)

        return betas

    
    def error_graph(self,error_history):
        iteraciones = np.arange(len(error_history)).reshape(-1,1)
        relationship = np.concatenate((iteraciones,error_history), axis=1)
        plt.figure(figsize=(10,10))
        plt.plot(relationship[:,0],relationship[:,1])
        plt.title('Curva de aprendizaje')
        plt.xlabel('Numero de iteración')
        plt.ylabel('Error')


    def model_graph(self, models, n, x_train, y_train):

        search_aux =  search_aux = np.arange(0,len(models.keys()),n)
        print(search_aux)
        x= x_train
        plt.figure(figsize=(10,10))
        for key in search_aux:
            betas = models[key]['betas']
            y = x*betas[0]+betas[1] 
            plt.plot(x,y ,label=f'iteración n = {key}')
            plt.scatter(x_train, y_train)
            plt.legend()
            plt.title(f'Evolución del modelo en el tiempo cada: {n} iteraciones')
            plt.xlabel('x = variable de interes')
            plt.ylabel('y = precio de las casas')


    def comparacion_modelos(self, modelo_manual, modelo_sklearn, x):

        y_sklearn = modelo_sklearn.predict(x)

        y_manual = self.predict(modelo_manual, x)

        y_prom =   (y_sklearn + y_manual )* 1/2

        return y_sklearn, y_manual, y_prom

    
    
    def predict(self, modelo_manual, x):

        x_matrix = np.concatenate((x, np.ones_like(x)), axis=1)
        betas = modelo_manual['betas']
        y_est = np.matmul(x_matrix,betas)



        return y_est

    def model_performace_comparison(self, y_sklearn, y_modelo, y_real):



        error_sklearn = (1/(2*(len(y_real))))*np.sum((y_sklearn-y_real)**2)
        error_manual =  (1/(2*(len(y_real))))*np.sum((y_modelo-y_real)**2)

        return error_sklearn, error_manual

