import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import linalg as la
import math

iris=pd.read_csv("Input/iris.csv",delimiter=";")

class PCA:
    def __init__(self,Data,NumericalCols=[]):
        self.Data=Data
        self.NumericalCols=NumericalCols

    def Standardization(self):
        X_std= StandardScaler().fit_transform(self.Data[self.NumericalCols])
        return X_std

    def CovMatrix(self):
        Matrix = self.Standardization()
        Transpose= np.transpose(Matrix)
        Size=len(Matrix)
        CovMatrix=(1/Size)*np.dot(Transpose,Matrix)
        return CovMatrix


    def EigenValue_Vector(self):
        CovMatrix= self.CovMatrix()
        valores_propios, vectores_propios = la.eig(CovMatrix)
        return valores_propios.real,vectores_propios

    def PrincipalComponents(self):
        Matrix = self.Standardization()
        Vectores_propios= self.EigenValue_Vector()[1]
        PCMA=np.dot(Matrix,Vectores_propios)
        return PCMA

    def VarComponents(self):
        Matrix = self.Standardization()
        Transpose = np.transpose(Matrix)
        Size = len(Matrix)
        VarComponents = (1 / Size) * np.dot(Matrix,Transpose)
        return VarComponents

    def Inertia(self,Eigenvalues):
        Inertia=self.EigenValue_Vector()[0]/len(self.EigenValue_Vector()[0])
        return Inertia[Eigenvalues]


    def cos(self):
        PCAMatrix=self.PrincipalComponents()
        rows, cols = PCAMatrix.shape
        Cosdf = pd.DataFrame(np.zeros((rows, cols)))
        StandardMatrixSum = self.Standardization().sum(axis=1)
        for i in range(rows):
            for j in range(cols):
                Cosdf.iloc[i,j]= PCAMatrix[i,j]**2/ StandardMatrixSum[i]**2
        return Cosdf








x= PCA(iris,['s.largo', 's.ancho', 'p.largo', 'p.ancho']).cos()


print(x)
