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
<br>
The platform uses the large model services provided by Alibaba Cloud. You can refer to https://bailian.console.aliyun.com/?spm=5176.29597918.J_SEsSjsNv72yRuRFS2VknO.2.78867b08NQ9OZn&tab=doc#/doc/?type=model&url=https%3A%2F%2Fhelp.aliyun.com%2Fdocument_detail%2F2840915.html&renderType=iframe to apply for the large model API key and replace the api_key in app.py.
