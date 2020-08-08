#Import all relevant modules
from Principal_Component_Analysis import *
from Dissipative_Component_Analysis import *
from SFI_Module import *
from Plotting_Module import *
import numpy as np
import multiprocessing
import pandas as pd
def Link_Stochastic_Force_Inference(projections_trajectory,output_directory,analysis_type="PCA"):
	"""
	This function simply calls Stochastic_Force_Inference via Call_SFI() for the number of modes specified here below in 'modes',
	for the specified basis type (see SFI for all available basis, here we use only a polynomial basis) coefficient,
	and for the selected order of the basis for the diffusion coefficient and for the force field. 
	Results are in the form of a pandas DataFrame and are saved in 'output_directory' as an h5 file.
	"""
	modes=[50] #Python list: each elements specifies how many modes are retained. Each element corresponds to a different call of SFI.
	basis_type=['polynomial'] #Python list: each elements specifies a different basis for the inference of diffusion coefficient and of force fields. Each element corresponds to a different call of SFI.
	diff_coeff_basis_order=[1] #Python list: each elements specifies the order of the basis used to infere the diffusion coefficient. Each element corresponds to a different call of SFI.
	force_basis_order=[1] #Python list: each elements specifies the order of the basis used to infere the force field. Each element corresponds to a different call of SFI.
	analysis_parameters=[[analysis_type,b,c,d,e] for b in modes for c in basis_type for d in diff_coeff_basis_order for e in force_basis_order] #List including all parameters specified above
	SFI_collected_output=pd.DataFrame([]) #Empty pandas dataframe to collect all results of SFI
	for params in analysis_parameters: 
		dataframe=Call_SFI(projections_trajectory,params,output_directory) #call Stochastic Force Inference (see SFI_module.py) for all parameters
		SFI_collected_output=SFI_collected_output.append(dataframe) #append all results to SFI_collected_output
	return SFI_collected_output


"""
Main file to perform PCA, DCA, and to analyze obtained trajectories via SFI in an automated way.
"""

current_directory=os.getcwd()+'/'
output_directory=current_directory+"Output/" #Path to output directory
input_directory = current_directory+"Input/" #Input directory. Needs to contain a Image/ subdirectory with 

#PCA_projections_test=PCA(input_directory,output_directory,truncation_criteria=True) #CAUTION: Needs images_zoom.npy in Input/Images/ folder. Performs PCA (See Principal_Component_Analaysis.py) with train/test separation and with truncation criteria activated. 
PCA_projections_test=np.load(output_directory+'PCA_projections_test.npy') #Load the PC-projected test set

SFI_collected_output=pd.DataFrame([])
SFI_output_PCA=Link_Stochastic_Force_Inference(PCA_projections_test,output_directory,analysis_type="PCA") #Performs SFI with PCs as input (See SFI_Module.py, the full SFI Software in folder SFI/, and https://link.aps.org/doi/10.1103/PhysRevX.10.021009). Returns a pandas dataframe collecting all results of SFI calls.

SFI_collected_output=SFI_collected_output.append(SFI_output_PCA)


"""
Plotting: in this part we call the plotting functions defined in Plotting_Module.py that are employed to obtain the manuscript results of Fig. 3	
and of related sections in the Supplementary Material.
"""
Plot_Components('PCA',output_directory) #Plots first 4 PCs, DCs, and related trajectories (Fig. 3c-d)

#Plot_Sdot_Perc_Modes(SFI_collected_output,output_directory) #Plots recovered percentage of entropy production rate as a function of retained modes (Fig. 3e)
Plot_Image_Force_Panels(SFI_collected_output,input_directory,output_directory)


