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

An online trial version of the platform has been launched (http://47.115.46.121:5000/). The online trial version cannot save data for a long time and has limited computing resources, so local deployment is recommended.
1.	Development environment preparation
First, you need to prepare the development environment. Mainly, you need to install the Python environment and the C/C++ environment.
It is recommended to install the Python environment via Anaconda. You need to go to the Anaconda official website(https://www.anaconda.com/download) to download and install Anaconda, and after installation, you will have the Python running environment.
As for the C/C++ running environment, it depends on the situation: if you are using a Windows system, you need to download the Visual Studio software(https://visualstudio.microsoft.com/zh-hans/downloads/); if you are using a Linux system, you can directly run sudo apt install gcc.

2.	Deploy the code
First, you need to download the code from GitHub; simply download the code zip package.
 
After downloading and extracting the code, you need to install the Python libraries required for the code to run by entering the following command in the terminal:
pip install -r requirements.txt
Additionally, run the following command to install the deep learning model runtime environment:
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cpuonly -c pytorch
We use the pypls library for PLS-DA and OPLS-DA analyses, so we need to follow the pypls library with reference to https://github.com/Omicometrics/pypls/tree/master.
In addition to the above steps, you need to apply for an Alibaba Cloud (Aliyun) Large Model API key to access the large model services. Specifically, follow these steps:
1.	Visit the Aliyun Large Model homepage(https://bailian.console.aliyun.com/#/home) to register and log in.
2.	After successful registration, you will receive 10 million free tokens.
3.	Open the app.py file and locate Line 104.
4.	Replace the placeholder with your newly obtained API key.
Finally, enter python app.py in the command line to run the platform code.
