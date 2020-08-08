Readme file for code associated to the manuscript 'Learning the nonequilibrium dynamics of Brownian movies' by Federico S. Gnesotto, Grzegorz Gradziuk, Pierre Ronceray, and Chase P. Broedersz


General remarks on this software:
This software is written in Python3. In the 'Sample_Code/' folder you can find three different subfolders:

	1) EPR_inference: this folder contains the sample code that we use to compute the recovered entropy production rate. The whole code can be run by opening a shell in this folder and typing 'python Main.py' or ('python3 Main.py' depending on the version installed).
	Note, performing PCA on the full Brownian movie (25.6 Gb, 10^6 time points, 6400-pixel images) requires at least 50Gb of RAM. For this reason, we have here performed PCA already and copied results into the Output/ folder. We do not include the full Brownian Movie, which is however available at the following link:

	(PUT LINK).

	Should the referees want to perform PCA, please download the full movie ('images.npy' 10^6x6400 array) from the previous link and place it into the Input/Images/ folder. Then, uncomment the 36th line of Main.py and comment out line 37th. By running 'python Main.py', Main.py simply loads the data already projected onto the first 200 PCs ('PCA_projections_all.npy'-line 37 of Main.py) and proceeds automatically with the following steps:
		
		a) Calls DCA() to perform Dissipative Component Analysis on the PC-projected data contained in 'PCA_Projections_all.npy', a 10^6x200 array  (File Dissipative_Component_Analysis.py)

		b) Runs Stochastic Force Inference (SFI) on PCA/DCA projections with the parameters specified in the function Link_Stochastic_Force_Inference() of Main.py (lines 40/41). 
		For each set of parameters specified, results are outputted as a pandas dataframe and saved in a single folder as a .h5 file. Results can be loaded using the built-in function 'read_hdf()' of pandas.

		c) Plots info about components and trajectory via Plot_Components() (Fig 3 c/d) and the recovered EPR  as a function of retained modes(Fig. 3 e) via Plot_Sdot_Perc_Modes(). These functions are defined in Plotting_Module.py

	Expected output (including the output of truncation criteria which is obtained from the PCA() function) for the specified parameters has been stored in the Output/ folder.

	2) Image_force_inference: this folder contains the sample code that we use to infer image forces on the region of the Brownian movie highlighted in white in Fig3b of the manuscript. To do so, we provide the exact image-forces ('image_forces.npy' a 1000x900 array) in the Input/Image_Forces/ subfolder. These 1000 time-step long exact image-forces are loaded by the Plot_Image_Force_Panels() function of Plotting_Module.py for comparison with the inferred image forces. As in EPR_inference, we have already performed PCA and stored in the the projected data on PCs and the corresponding Principal components into the Output/ folder. Our code begins by loading the 'PCA_projections_test.npy' file (a 9x10^5x50 numpy array containing the PC-projected test set). Should the referees want to perform PCA autonomously, the full Brownian movie is available at:

	https://www.dropbox.com/s/f4ze6vh63sz4l43/BM.zip?dl=0

	Please refer to the previous point for info on how to change the code to additionally perform PCA on the downloaded movie.

	Similarly to the previous point, the code in Main.py then proceeds automatically by running SFI with selected parameters (function Link_Stochastic_Force_Inference()) on the first 50 PCA projections of the test set, saving the output of SFI, plotting PCs with trajectories and finally calling Plot_Image_Force_Panels() to output the image force maps and the scatter plot (Fig.3 f-g) in the single directory created for each set of parameters. Note, running SFI on the first 50PCs is a computationally and memory-expensive task that lasts several hours. 

	Expected output for the specified parameters has been stored in the Output/ folder.

	3) SFI_latest: this is the Stochastic Force Inference software (https://journals.aps.org/prx/abstract/10.1103/PhysRevX.10.021009). 

