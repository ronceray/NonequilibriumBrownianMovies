import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
from statsmodels.tsa.stattools import acf
def PCA(input_directory,output_directory,truncation_criteria=True):
  """  
  This function loads the brownian movie in the form of an NxS (N=number of time steps, S=LxW=size of the flattened movie) numpy array of images (X),
  Performs Principal Component Analysis by computing and diagonalizing the covariance matrix of the first 1/10 of the included data length in time steps.
  If truncation_criteria=True, the threshold according to the two criteria mentioned in the manuscript are computed.

  """
  print("Performing PCA. Load images....")
  X=np.load(input_directory+"Images/images_zoom.npy") #Loads brownian movie (rows=time, columns=pixel values)
  print("Done. Compute Covariance...")
  X=X-np.mean(X,axis=0) #Subtract mean image from all images
  print("Splitting dataset into train/test set...")
  X_train=X[:X.shape[0]//10] #First 1/10 of the trajectory is training set
  X_test=X[X.shape[0]//10:] #9/10 of the trajectory is test set
  print(f"Length of train set is {X_train.shape[0]}, length of test set is {X_test.shape[0]}")
  Covariance=np.dot(X_train.T,X_train)/(X_train.shape[0]-1) #Covariance Matrix
  print("Diagonalization covariance...")
  eigenvals,eigenvecs=np.linalg.eigh(Covariance) #Diagonalization: CAREFUL! Evals and Evecs are stored as columns and in ascending order of magnitude
  print('Done')
  PCA_eigenvalues=eigenvals[::-1] #Reversing order of eigenvalues for convenience
  PCA_components=(eigenvecs.T)[::-1] #It is convenient to have Principal Components in rows. Again order is reversed to keep ordering same as eigenvalues
  max_modes=200 #Sets maximum number of modes to be retained. If set to the size of an image (X.shape[1]), all modes are retained.
  print('Projecting trainset onto PCs')
  PCA_projections_all=(X).dot(PCA_components[:max_modes].T) #Projects the full data set onto the PCs
  PCA_projections_test=PCA_projections[:X_test.shape[0],:] #Projected test set

  print('Done, saving...')
  np.save(output_directory+'PCA_components.npy', PCA_components[:max_modes])
  np.save(output_directory+'PCA_projections_all.npy', PCA_projections)
  np.save(output_directory+'PCA_projections_test.npy', PCA_projections_test)
  np.save(output_directory+'PCA_eigenvalues.npy', PCA_eigenvalues[:max_modes])
  if(truncation_criteria==True):
    """
    Criterion 1): Data is shuffled, covariance computed for the shuffled data and the largest eigenvalue of the covariance used as threshold.
    """
    print('First truncation criterion:shuffling...')
    X_shuffled=X_train #Copies data to be shuffled
    np.random.seed(42)    
    for i in range(X_shuffled.shape[1]): #Shuffle along columns separately
      np.random.shuffle(X_shuffled[:,i])
    print('Diagonalisation..')
    X_shuffled=X_shuffled-np.mean(X_shuffled,axis=0)
    Covariance_shuffled=np.dot(X_shuffled.T,X_shuffled)/(X_shuffled.shape[0]-1)
    eigenvals_shuffled,eigenvecs_shuffled=np.linalg.eig(Covariance_shuffled)
    print('Outputting figure..')
    """Output Figure for 1st truncation criterion (cf. Supplementary Fig. 3b)"""
    fig=plt.figure()
    plt.plot(PCA_eigenvalues[:max_modes],'o',markersize=3,label='signal') #Plots eigenvalues of the covariance matrix for the unshuffled data.
    plt.plot(np.linspace(0,max_modes,len(eigenvals)),[eigenvals_shuffled[0] for i in range(len(eigenvals))],label='threshold') #Plots threshold as a horizontal line.
    plt.xlabel(r"mode $i$")
    plt.ylabel(r"$\lambda_i$")
    plt.yscale('log')
    plt.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(output_directory+"1st_truncation_criterion.png")
    """
    Criterion 2): The (normalized) autocorrelation function of the first max_modes trajectories is computed using the built-in acf module of statsmodels.tsa.stattools. 
    The jump of the autocorrelation function in one time step is plotted as a function of the mode number. We set the maximal threshold at 0.25.
    """ 
    print('Second truncation criterion:computing autocorrelation...')
    decorrelation_one_timestep=[]
    for i in range(max_modes):
        autocorrelation_function=acf(PCA_projections[:,i],nlags=10) #Computes first 10 values of autocorrelation function.
        decorrelation_one_timestep.append(np.abs(autocorrelation_function[1]-autocorrelation_function[0])) #Appends one-timestep jump of autocorrelation
    """Output Figure for 2nd truncation criterion (cf. Supplementary Fig. 3d)"""
    print('Done. Plotting..')
    fig=plt.figure()
    plt.plot(decorrelation_one_timestep)
    plt.plot(np.linspace(0,max_modes,len(eigenvals)),[0.25 for i in range(len(eigenvals))])
    plt.xlabel(r"mode $i$")
    plt.ylabel(r"$|C_i(\Delta t)-C_i(0)|$")
    plt.tight_layout()
    fig.savefig(output_directory+"2nd_truncation_criterion.png")
  return PCA_projections_test

