# NMRplatform
NMRplatform an integrated, intelligent platform for NMR metabolomics analysis that consolidates key processes: Fourier transformation, baseline correction, automated metabolite identification, relative quantification, multivariate statistical analysis, and pathway enrichment. 
### Environment
python 3.10
torch 1.10.0+cu113
you can use this comand to download torch :
```
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cpuonly -c pytorch
```

We use the pypls library for PLS-DA and OPLS-DA analyses, so we need to follow the pypls library with reference to https://github.com/Omicometrics/pypls/tree/master. Before that, a C/C++ environment needs to be set up.
