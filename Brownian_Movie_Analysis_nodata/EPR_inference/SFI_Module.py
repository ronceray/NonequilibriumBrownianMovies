import sys
sys.path.append('../SFI_latest/') 
import numpy as np
import pandas as pd
import os
from SFI_data import StochasticTrajectoryData

def Call_SFI(projections_trajectory,params,output_directory):
	print('Starting SFI with: ')

	dt=0.005 #Simulation time_step
	analysis_type=params[0]
	number_retained_modes=params[1] 
	basis_type=params[2]
	diff_coeff_basis_order=params[3]
	force_basis_order=params[4]
	print(f'Type of analysis: {analysis_type}')
	print(f'Number of Modes: {number_retained_modes}')
	print(f'Basis Type: {basis_type}')
	print(f'Order of basis for diffusiion inference: {diff_coeff_basis_order}')
	print(f'Order of basis for force inference: {force_basis_order}')
	Xlist= np.einsum('imt->tim',np.array([[projections_trajectory[:,i] for i in range(number_retained_modes)]])) #Rearranges data for SFI
	tlist=np.linspace(0,projections_trajectory.shape[0]*dt,projections_trajectory.shape[0]) #List of time-steps
	data = StochasticTrajectoryData(Xlist,tlist) #Prepares data for SFI	
	center = data.X_ito.mean(axis=(0,1)) 
	width  = 1.1 * ( data.X_ito.max(axis=(0,1)) - data.X_ito.min(axis=(0,1)) )
	single_output_directory=output_directory+params[0]+'/'+str(basis_type)+'/Modes'+str(number_retained_modes)+'/F_basis'+str(force_basis_order)+'_D_basis'+str(diff_coeff_basis_order)+'/' #name of directory for single output of SFI
	
	if not os.path.exists(single_output_directory):
	  os.makedirs(single_output_directory) #create directory
	else:
		print('Single Output Directory exists: this will overwrite the present files') #warning

	from SFI_diffusion import DiffusionInference
	"""
	Diffusion Inference. Throughout the manuscript we always use a 1st order polynomial basis with 'Vestergaard' noise-corrected estimator.
	"""
	print('Performing Diffusion Inference..')
	DI = DiffusionInference( data, 
	                         #basis = { 'type' : 'polynomial', 'order' : 2, 'hierarchical' : False },
	                         basis = { 'type' : basis_type, 'order' : diff_coeff_basis_order , 'center' : center, 'width' : width },
	                         #diffusion_method = 'MSD',
	                         diffusion_method = 'Vestergaard', verbose=False
	)
	
	print('Done.')
	
	from SFI_forces import StochasticForceInference
	
	"""
	Inference of Force field, Velocity field, Entropy Production Rate, and related error bars. SFI computes additional quantities that we do not employ in our manuscript.
	For a complete list of the capabilities of SFI please refer to  https://link.aps.org/doi/10.1103/PhysRevX.10.021009.
	"""

	print('Performing Force Inference..')
	L = StochasticForceInference( data,
	                              diffusion_data = { 'type' : 'DiffusionInference', 'DI' : DI, 'cutoff' : np.sqrt(DI.projections_self_consistent_error)},
	                              #D,
	                              #basis = { 'type' : 'polynomial', 'order' : 2, 'hierarchical' : False },
	                              basis = { 'type' : basis_type, 'order' : force_basis_order, 'center' : center, 'width' : width },
	                              verbose=True
	)
	print('Done. Outputting...')
	Sdot_exact=50.5 #Exact entropy production rate for the movie
	SFI_output_list=[single_output_directory,analysis_type,number_retained_modes,basis_type,diff_coeff_basis_order,force_basis_order,np.sqrt(DI.projections_self_consistent_error),
	L.Sdot,L.Sdot_error,Sdot_exact,L.Sdot_bias,L.projections_self_consistent_error,L.trajectory_length_error,
	L.discretization_error_bias,L.discretization_error_flct,DI.projections_self_consistent_error,DI.trajectory_length_error,
	DI.discretization_error_bias,np.array(L.F_ansatz(data.X_ito[:,0,:]))] #List of SFI return values
	
	keys_list=['Output_Dir','Analysis_Type','Modes','Basis_Type', 'Diff_Basis', 'Force_Basis','Cutoff','Sdot','Sdot_error',
	'Sdot_exact','Sdot_bias','F_error_self_cons','F_error_traj_length','F_error_discr_bias','F_erro_discr_fluct',
	'D_error_self_cons','D_error_traj_length','D_error_discr','F_field_traj'] #Keys referring to SFI_output_list
	
	SFI_output_data_frame=pd.DataFrame([dict(zip(keys_list,SFI_output_list))]) #SFI output is stored as a pandas dataframe
	store = pd.HDFStore(single_output_directory+'Results.h5') #Outputs to hdf5 file
	store['data']=SFI_output_data_frame
	store.close()
	print('Done..')
	return SFI_output_data_frame