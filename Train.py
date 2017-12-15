import seaborn as sns
import matplotlib.pyplot as plt
import Constant as c
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def Projection():
    X = np.array([np.asarray(c.Images["RAWRED-MEAN"]),np.asarray(c.Images["RAWBLUE-MEAN"]),np.asarray(c.Images["RAWGREEN-MEAN"])])
    Images_array = []
    for i in c.Images_Features_Dictionery:
        Images_array.append(np.asarray(c.Images[c.Images_Features_Dictionery[i]]))
    Images_array = np.array(Images_array).T
    
    Invers = np.linalg.linalg.inv(np.linalg.linalg.dot(Images_array.T,Images_array))
    Image_Inverse =  np.linalg.linalg.dot(Images_array,Invers)
    
    Projection_Matrix = np.linalg.linalg.dot(Image_Inverse,Images_array.T)

    x_axis = np.linalg.linalg.dot(Projection_Matrix,X[0].T)
    y_axis = np.linalg.linalg.dot(Projection_Matrix,X[1].T)
    z_axis = np.linalg.linalg.dot(Projection_Matrix,X[2].T)

    
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter3D(x_axis, y_axis, z_axis)
    plt.show()

def Rug_Vis ():
    for i in c.Images_Features_Dictionery:
        sns.rugplot(c.Images[c.Images_Features_Dictionery[i]],0.05)
        plt.title(c.Images_Features_Dictionery[i] + " Rug Plot")
        plt.show()  
        

def Scatter_Vis():
    sns.pairplot(c.Images, size=3,kind='scatter',hue="Label").add_legend()
    plt.title("Data Set Scattering")
    plt.show()

def Hist_Vis():
    for i in c.Images_Features_Dictionery:
        c.Images.plot(kind="hist", x=c.Images_Features_Dictionery[i])
        plt.title(c.Images_Features_Dictionery[i] + " Histogram")
        plt.show()
        
def Box_Vis():
    for i in c.Images_Features_Dictionery:
        sns.boxplot(data=c.Images[c.Images_Features_Dictionery[i]], orient="h", width=1, linewidth=0.5)
        plt.title(c.Images_Features_Dictionery[i] + " Box Plot")
        plt.show()

def Get_Mean_Var():
    print("The mean : ")
    print(c.Images.mean())
    print("\nThe variance : ")
    print(c.Images.var())
    
def Generate_Samples(samples=50, observations=10):
    samples_mean = []
    samples_variance = []
    Features_samples_mean = {}
    
    for i in range(1,samples+1):
        s = c.Images.sample(observations)
        samples_mean.append(s.mean())
        samples_variance.append(s.var())
    
    for j in c.Images_Features_Dictionery:
        for k in samples_mean:
            try :
                Features_samples_mean[c.Images_Features_Dictionery[j]] = Features_samples_mean[c.Images_Features_Dictionery[j]]+(k[c.Images_Features_Dictionery[j]])
            except:
                Features_samples_mean[c.Images_Features_Dictionery[j]] = (k[c.Images_Features_Dictionery[j]])
                
    for l in Features_samples_mean:
        Features_samples_mean[l] = Features_samples_mean[l]/samples

    return samples_mean, samples_variance, Features_samples_mean

def Cov_Corr():
    
    print(c.Images.cov())
    sns.pairplot(c.Images.cov(), size=3,kind='scatter')
    plt.title("Covariance Scattring")
    plt.show()
    
    sns.heatmap(c.Images.cov())
    plt.title("Covariance Image")
    plt.show()
    
    sns.heatmap(c.Images.corr())
    plt.title("Corrleation Image")
    plt.show()
