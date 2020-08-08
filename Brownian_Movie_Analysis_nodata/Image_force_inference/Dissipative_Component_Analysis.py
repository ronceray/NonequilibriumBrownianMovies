import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as scilg
from numpy import linalg as nlg


def DCA(X,output_directory):
    """
    Performs Dissipative Component Analysis (DCA) starting from Projected data onto PCs. For details please refer to the Supplementary Material (SM) (Sec. IV ).
    Returns data projected onto DCs.
    """
    print("Perfomind DCA..")
    dt=0.005 #Time step of simulation
    max_modes=70 #maximum number of retained modes
    X = X - np.mean(X,axis=0) #center data around average
    print("Splitting dataset into train/test sets")
    X_train=X[:X.shape[0]//10,:max_modes] #Split trajectories in train/test sets
    X_test=X[X.shape[0]//10:,:max_modes] 
    print('Computing Adot..')
    dX=X_train[1:]-X_train[:-1] #Compute one time-step difference 
    X_times_dX=np.tensordot(X_train[:-1],dX,axes=([0],[0])) 
    Adot=(X_times_dX-X_times_dX.T)/(2.*X_train.shape[0]*dt) #Compute area enclosing rate
    print('Computing change-of-basis matrix B..')
    Covariance=np.dot(X_train.T,X_train)/(X_train.shape[0]-1) #Compute Covariance: note that we are already in PC-coordinates, so Covariance is diagonal.
    Covariance_evals,Covariance_evecs=np.linalg.eigh(Covariance) #Diagonalize
    B=(Covariance_evecs[:,::-1])/np.sqrt(Covariance_evals[::-1]) #Compute change-of-basis matrix to covariance-identity-coordinates (cic). In the manuscript it is indicated as C_pca
    B_inv=Covariance_evecs[:,::-1]*np.sqrt(Covariance_evals[::-1]) #Inverse of B
    print('Computing Diffusion matrix D..')
    D=np.dot(dX.T,dX)/(2*dX.shape[0]*dt) #Compute diffusion matrix via mean-square-displacement
    D_inv=nlg.inv(D) #Inverse of D
    print('Transforming into cic..')
    Adot_cic=(B.T).dot(Adot).dot(B) #Area enclosing rate in cic
    D_inv_cic=B_inv.dot(D_inv.dot(B_inv.T)) #Diffusion matrix in cic
    Adot_cic_vals,Adot_cic_vecs=nlg.eigh(Adot_cic.dot(Adot_cic.T)) #Diagonalize product A_dot_cic.Adot_cic^T (Eigenvalues are denoted as 'lambda' in Sec. IV of SM)
    D_inv_scic=Adot_cic_vecs.T.dot(D_inv_cic).dot(Adot_cic_vecs) #D_inv in special-cic (scic)
    print('Ordering terms..')
    Sdot_terms=Adot_cic_vals[::2]*(np.diag(D_inv_scic)[::2]+np.diag(D_inv_scic)[1::2]) #Form Sdot pairs that will be used for ordering the dissipative components (see Eq. S4 of Sec. IV of SM)
    vec_pairs=np.array([[Adot_cic_vecs[:,i],Adot_cic_vecs[:,i+1]] for i in range(0,Adot_cic_vecs.shape[1],2)]) #Form pairs of eigenvectors of Adot_cic
    Sdot_ordering_args=np.argsort(Sdot_terms)[::-1] #Find indices that sort Sdot_terms from largest to smallest contributions.
    DCA_components=vec_pairs[Sdot_ordering_args,:,:].reshape(Adot_cic_vecs.shape).T #Dissipative components: they are the reordered eigenvectors of A_dot_cic according to Sdot_terms
    DCA_eigenvalues=Adot_cic_vals[::2][Sdot_ordering_args] #Reordering eigenvalues (lambda in SM Sec. IV)
    DCA_components=DCA_components.T #Transpose and put vectors in rows for convenience
    X_test_cic=B.T.dot((X_test-np.mean(X_test,axis=0)).T) #Change basis of X_test to cic
    DCA_projections=X_test_cic.T.dot(DCA_components.T) #project the test set onto DCs
    np.save(output_directory+"DCA_components.npy", DCA_components[:max_modes])
    np.save(output_directory+"DCA_projections.npy", DCA_projections[:,:max_modes])
    np.save(output_directory+"DCA_eigenvalues.npy", DCA_eigenvalues)
    print('Done.')
    return DCA_projections
