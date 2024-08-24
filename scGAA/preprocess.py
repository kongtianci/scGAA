import scanpy as sc
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse


data = sc.read('/data/your_raw_data.h5ad')

#QC
# mitochondrial gene
data.var["mt"] = data.var_names.str.startswith("MT-")
# ribosomal gene
data.var["ribo"] = data.var_names.str.startswith(("RPS", "RPL"))
# haemoglobin gene
data.var["hb"] = data.var_names.str.contains(("^HB[^(P)]"))

sc.pp.calculate_qc_metrics(
    data, qc_vars=["mt", "ribo", "hb"], inplace=True, percent_top=[20], log1p=True
)

from scipy.sparse import issparse
if issparse(data.X):
    data.obs['nUMIs'] = data.X.toarray().sum(axis=1)  
    data.obs['mito_perc'] = data[:, data.var["mt"]].X.toarray().sum(axis=1) / data.obs['nUMIs'].values
    data.obs['detected_genes'] = (data.X.toarray() > 0).sum(axis=1)  
else:
    data.obs['nUMIs'] = data.X.sum(axis=1)  
    data.obs['mito_perc'] = data[:, data.var["mt"]].X.sum(axis=1) / data.obs['nUMIs'].values
    data.obs['detected_genes'] = (data.X > 0).sum(axis=1)  

scales_counts = sc.pp.normalize_total(data, target_sum=None, inplace=False)
# log1p transform
data.layers["log1p_norm"] = sc.pp.log1p(scales_counts["X"], copy=True)

# #Conversion to sparse matrices
# from scipy.sparse import csr_matrix
# analytic_pearson = sc.experimental.pp.normalize_pearson_residuals(data, inplace=False)
# data.layers["analytic_pearson_residuals"] = csr_matrix(analytic_pearson["X"])

sc.pp.normalize_total(data)
sc.pp.log1p(data)

#Extraction of highly variable genes
data_dis_num=sc.pp.highly_variable_genes(
    data,
    flavor="seurat",
    n_top_genes=2000,
    subset=False,
    inplace=False,
)

data_dis_cutoff=sc.pp.highly_variable_genes(
    data,
    flavor="seurat",
    min_disp=0.5,
    min_mean=0.0125,
    max_mean=3,
    subset=False,
    inplace=False,
)
data_dis_cutoff['highly_variable'].value_counts()

sc.pp.highly_variable_genes(
    data,
    flavor="seurat",
    n_top_genes=3000,
)
#Extracting highly variable genes
data_top = data[:, data.var.highly_variable]
print('shape: ',data_top.shape)

data_top.write('./data/preprocessed_data.h5ad')