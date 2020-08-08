import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import numpy as np 
from matplotlib import rc
import os
import matplotlib.gridspec as gridspec
plt.rc('text', usetex=False)
plt.rc('font', family='serif')
plt.rcParams['xtick.labelsize']=20
plt.rcParams['ytick.labelsize']=20
plt.rcParams['font.size']=20
"""
This is a collection of matplotlib-based functions that conveniently plot the output of SFI in the form presented in the manuscript. 
All functions can be used by simply providing the pandas dataframe and without re-running the analysis.
"""
t_min=0 #Initial trajectory-timestep for force inference (not whole trajectory is used)
t_max=999 #Final trajectory-timestep for force inference (not whole trajectory is used)

def Plot_Sdot_Perc_Modes(results,output_dir):
	"""
	Plots the recovered percentage of entropy production rate as a function of the number
	of retained modes
	"""
	print('Plotting Recovered EPR as a function of retained modes..')
	cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
	Sdot_exact=results["Sdot_exact"].iloc[0]
	f, ax = plt.subplots(figsize=(8,6))
	for index,group in results.groupby(["Analysis_Type"]):
		analysis=group["Analysis_Type"].iloc[0]
		Modes=np.array(group["Modes"].tolist())
		S_dot=np.array(group["Sdot"].tolist())-np.array(group["Sdot_bias"].tolist())
		S_dot_err=np.array(group["Sdot_error"].tolist())
		if(analysis=="PCA"):
			ax.errorbar(Modes,S_dot/Sdot_exact*100,yerr=S_dot_err/Sdot_exact*100,marker='o',color=cycle[0],label='PCA')
		elif(analysis=="DCA"):
			ax.errorbar(Modes,S_dot/Sdot_exact*100,yerr=S_dot_err/Sdot_exact*100,marker='v',color=cycle[1],label='DCA')
	ax.set_xlabel("Retained modes")
	ax.set_ylabel(r"$\hat{\dot{S}}/\dot{S}_{\rm ex} \, (\%)$")
	ax.set_xticks([i for i in Modes])
	ax.axhline(y=0,c='k')
	ax.set_ylim(bottom=-.5)
	plt.legend(frameon=False)
	plt.tight_layout()
	f.savefig(output_dir+"Sdot_percentage_"+results['Basis_Type'].iloc[0]+f"_Force_order_{results['Force_Basis'].iloc[0]}_Diff_order{results['Diff_Basis'].iloc[0]}.png",dpi=400)
	plt.close()

def Plot_Components(Analysis_Type,output_dir):
	print('Plotting PCs/DCs and trajectories..')
	"""
	Loads and plots the first 4 PCs/DCs and the part of the related projection coefficient's trajectories
	"""
	cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
	components=np.load(output_dir+f"{Analysis_Type}_components.npy")
	projections=np.load(output_dir+f"{Analysis_Type}_projections.npy")
	if(Analysis_Type=='DCA'):
		components=components.dot(np.load(output_dir+f"PCA_components.npy")[:components.shape[0]])
	fig = plt.figure(figsize=(6,8))
	vmin=-0.05
	vmax=0.05
	gs = gridspec.GridSpec(8,1, height_ratios=[0.2,1,0.2,1,0.2,1,0.2,1]) 
	gs.update(left=0.05, right=0.3, bottom=0.2, top=0.93, wspace=0.04, hspace=0.5)

	ax1=plt.subplot(gs[0,0])
	plt.plot(np.linspace(0,5,500),projections[0:500,0],color=cycle[0])
	ax1.set_yticks([])
	ax1.set_xticks([0,5])
	ax1.tick_params(axis='both', which='major', labelsize=10)
	ax1.set_xlabel(r'time (a.u)',fontsize=10)
	ax1.xaxis.set_label_coords(0.5, -0.025)

	ax2=plt.subplot(gs[1,0])
	plt.imshow(components[0].reshape(80,80),cmap='RdGy_r',vmin=vmin,vmax=vmax)
	ax2.set_xticks([])
	ax2.set_yticks([])
	

	ax3=plt.subplot(gs[2,0])
	plt.plot(np.linspace(0,5,500),projections[0:500,1],color=cycle[1])
	ax3.set_yticks([])
	ax3.set_xticks([0,5])
	ax3.tick_params(axis='both', which='major', labelsize=10)
	ax3.set_xlabel(r'time (a.u)',fontsize=10)
	ax3.xaxis.set_label_coords(0.5, -0.025)

	ax4=plt.subplot(gs[3,0])
	plt.imshow(components[1].reshape(80,80),cmap='RdGy_r',vmin=vmin,vmax=vmax)
	ax4.set_xticks([])
	ax4.set_yticks([])
	

	ax5=plt.subplot(gs[4,0])
	plt.plot(np.linspace(0,5,500),projections[0:500,2],color=cycle[2])
	ax5.set_yticks([])
	ax5.set_xticks([0,5])
	ax5.tick_params(axis='both', which='major', labelsize=10)
	ax5.set_xlabel(r'time (a.u)',fontsize=10)
	ax5.xaxis.set_label_coords(0.5, -0.025)

	ax6=plt.subplot(gs[5,0])
	plt.imshow(components[2].reshape(80,80),cmap='RdGy_r',vmin=vmin,vmax=vmax)
	ax6.set_xticks([])
	ax6.set_yticks([])

	ax7=plt.subplot(gs[6,0])
	plt.plot(np.linspace(0,5,500),projections[0:500,3],color=cycle[3])
	ax7.set_yticks([])
	ax7.set_xticks([0,5])
	ax7.tick_params(axis='both', which='major', labelsize=10)
	ax7.set_xlabel(r'time (a.u)',fontsize=10)
	ax7.xaxis.set_label_coords(0.5, -0.025)

	ax8=plt.subplot(gs[7,0])
	plt.imshow(components[3].reshape(80,80),cmap='RdGy_r',vmin=vmin,vmax=vmax)
	ax8.set_xticks([])
	ax8.set_yticks([])

	plt.show()
	fig.savefig(output_dir+f'{Analysis_Type}_components_trajectory_plot.png',dpi=400,bbox_inches='tight')
	plt.close()

def Plot_Force_Images(image_forces_inferred,image_forces_exact,output_dir,height,width):
	"""
	Selects four instants of time from the input data and plots exact/inferred image-force maps (Fig 3f of the manuscript).
	"""
	print('Plotting Image Forces..')
	scale_min=-4.
	scale_max=4.
	for i in range(0,t_max-t_min,(t_max-t_min)//4):
		fig, ax = plt.subplots(2,1,figsize=(8,6),gridspec_kw={'hspace': 0.2})
		ax[0].axis('off')
		ax[0].imshow(image_forces_exact[i].reshape(height,width),cmap='RdBu_r',vmin=scale_min,vmax=scale_max)
		ax[0].set_title(r'$\mathcal{F}_{\rm ex}$')

		ax[1].axis('off')
		im=ax[1].imshow(image_forces_inferred[i].reshape(height,width),cmap='RdBu_r',vmin=scale_min,vmax=scale_max)
		ax[1].set_title(r'$\hat{\mathcal{F}}$')
		
		fig.colorbar(im,ax=ax.ravel().tolist(),ticks=[-4, 0, 4],location='right', shrink=0.6)# ax=ax.flatten(), ticks=[-4, 0, 4],location='right', shrink=0.6)    
		fig.savefig(output_dir+f'Image_Force_Comp{i}.png',dpi=300,bbox_inches='tight')
		plt.close()


def Plot_Hex_Scatter_Force(image_forces_inferred,image_forces_exact,output_dir):
	"""
	Constructs the scatter plot of Fig. 3g. Data are binned via the built-in function hexbin.
	"""
	print('Plotting Image Forces Scatter plot...')
	from matplotlib import cm
	fig, ax = plt.subplots()
	image_forces_exact_flattened=np.array(image_forces_exact[t_min:t_max]).flatten() #flattens the array
	image_forces_inferred_flattened=np.array(image_forces_inferred[t_min:t_max]).flatten()
	"""
	First 'hb' is used for finding the bin values that are used in log-scale as norm. norm is passed to the second 'hb' for plotting
	"""
	hb=ax.hexbin(image_forces_exact_flattened,image_forces_inferred_flattened,C=np.ones_like(image_forces_inferred_flattened)/len(image_forces_inferred),gridsize=50,extent=(-5.5,5.5,-5.5,5.5),reduce_C_function=np.sum,cmap='Blues')
	norm=cm.colors.LogNorm(hb.norm.vmin,hb.norm.vmax) #use this as norm function in hexbin (logscale)
	del hb
	hb=ax.hexbin(image_forces_exact_flattened,image_forces_inferred_flattened,C=np.ones_like(image_forces_inferred_flattened)/len(image_forces_inferred),gridsize=50,extent=(-5.5,5.5,-5.5,5.5),reduce_C_function=np.sum,norm=norm,cmap='Blues')
	cb = fig.colorbar(hb, ax=ax)
	cb.set_label(r'$N/N_{\rm tot}$')
	ax.plot(np.linspace(-1.5,1.5,100),np.linspace(-1.5,1.5,100),linestyle='dashed',c='w',alpha=0.5) #plot y=x line
	corrcoeff=np.corrcoef(image_forces_exact_flattened,image_forces_inferred_flattened)[0,1] #pearson corrcoeff
	"""
	Here below we compute the relative mean squared error (\sigma^2_{\hat{\mathcal{F}}} in the manuscript)
	"""
	delta_F_sq=np.sum((image_forces_exact-image_forces_inferred)**2,axis=1) #norm term in the numerator is sum of pixel values squared at each time step
	avg_delta_F_sq=np.mean(delta_F_sq) #average of numerator terms in time
	F_inf_sq=np.sum(image_forces_inferred**2,axis=1) #norm term in denominator is sum of pixel values of inferred image force at each time step
	avg_F_inf_sq=np.mean(F_inf_sq) #average of denominator terms in time
	self_cons_err=avg_delta_F_sq/avg_F_inf_sq #error

	ax.text(0.7,0.15, r'$\rho=%0.2f$'%corrcoeff,transform=ax.transAxes,color='k')
	ax.text(0.7,0.02, r'$\sigma^2_{\hat{\mathcal{F}}}=%0.2f$'%self_cons_err,transform=ax.transAxes,color='k')
	ax.set_ylabel(r'$\hat{\mathcal{F}}$')
	ax.set_xlabel(r'$\mathcal{F}_{\rm ex}$')
	ax.set_yticks([-4,2,0,2,4])
	ax.set_xticks([-4,2,0,2,4])
	fig.savefig(output_dir+f'Hex_Scatter_Force.png',dpi=300,bbox_inches='tight')
	plt.close()

def Plot_Image_Force_Panels(results,input_dir,output_dir):
	image_forces_exact=np.load(input_dir+'Image_Forces/image_forces.npy')[t_min:t_max]
	PCA_components=np.load(output_dir+'PCA_components.npy')
	for j,h in results.groupby(["Modes","Force_Basis","Diff_Basis","Basis_Type"]):				
		Modes=h['Modes'].item()
		force_ansatz=h['F_field_traj'].item()
		image_forces_inferred=force_ansatz[t_min:t_max,:Modes].dot(PCA_components[:Modes,:]) #Inferred image-forces are computed by multiplying the inferred force coefficients with the corresponding PC.
		Plot_Force_Images(image_forces_inferred,image_forces_exact,h['Output_Dir'].item(),30,30) #Plots exact and inferred image-forces (Fig. 3 f)
		Plot_Hex_Scatter_Force(image_forces_inferred,image_forces_exact,h['Output_Dir'].item()) #Produces Scatter plot (Fig. 3 g)
	









