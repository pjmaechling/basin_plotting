These scripts are used to (1) use basin_query_mpi_complete (a UCVM executable) to extract basin depth information from UCVM models, and (2) to plot the resulting data files using matplotlib. 
The slurm scripts for extracting the basin depth values (configured for Stampede3) are in the slurm subdirectory.
The python script for plotting the extracted datasets are plot_z25.py (this works for other Z values, not just Z2.5).
Examples of the basin_depth datafiles and the resulting plots are in the z25 and z10 subdirectories.
