
## scGCO

**scGCO** is a method to identify genes which significantly depend on spatial coordinates.The intended applications are spatially resolved RNA-sequencing from e.g. Spatial Transcriptomics, or *in situ* gene expression measurements from e.g. SeqFISH or MERFISH.

The reposity contains both the implementations of our methods,as well as case studies in applying it.

The primary implementation is as a Python 3 package,and can be installed from the command line by

	pip install scgco

### License
MIT Licence, see LICENSE file.

###  Authors
See AUTHORS file.

### Contact
For bugs,feedback or help you can contact Peng Wang <wangpeng@picb.ac.cn>.

To see usage example of scGCO either keep reading, or look in the `Analysis` directory.The following examples are 
provided:

- **BreastCancer**  - Transcriptome wide study on breast cancer tissue from Spatial Transcriptomics
- **MouseOB** - Spatial Transcriptomics assay of a slice of Mouse Olfactory Bulb.
- **MERFISH** - Expression from single cells in a region of an osteoblast culture using the MERFISH technology with 140 probes.
- **SeqFISH** -  Expression counts of single cells from mouse hippocampus using the SeqFISH technology with 249 probes.


## scGCO significance test example use

As an example, let us As an example, let us look at spatially dependent gene expression in Mouse Olfactory Bulb using a data set published in [Stahl et al 2016](http://science.sciencemag.org/content/353/6294/78). With the authors method, hundrads of locations on a tissue slice can be sampled at once, and gene expression is measured by sequencing in an unbiased whole-transcriptome manner. 

### Input Format
The referred matrix format is the ST data format, a matrix of counts where spot coordinates are row names
and the genes are column names.This matrix format (.TSV) is split by tab.


	import scGCO
	
	# read spatial expression data tab split counts matrix
	ff = 'Analysis/data/MOB-breast-cancer/Rep11_MOB_count_matrix-1.tsv'
	locs, data = scGCO.read_spatial_expression(ff)
	
	# remove genes expressed in less than 10 cells
	data = data.loc[:,(data != 0).astype(int).sum(axis=0) >= 10]
	
	# normalize expression and use 1000 genes to test the algorithm
	data_norm = scGCO.normalize_count_cellranger(data)
	data_norm = data_norm.iloc[:,0:1000]
	
	# estimate number of segments
	factor_df, size_factor = scGCO.estimate_smooth_factor(locs, data_norm)
	
	# run the main algorithm to identify spatially expressed genes
	# this should take less than a minute 
	result_df = scGCO.identify_spatial_genes(locs, data_norm, size_factor)
