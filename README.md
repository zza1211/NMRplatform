<!-- ABOUT THE PROJECT -->
## About The Project
### NMRplatform
NMRplatform an integrated, intelligent platform for NMR metabolomics analysis that consolidates key processes: Fourier transformation, baseline correction, automated metabolite identification, relative quantification, multivariate statistical analysis, and pathway enrichment. 

### Environment

python 3.10</br>
torch 1.10.0+cu113</br>
c/c++

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started


### 1.	Development environment preparation

First, you need to prepare the development environment. Mainly, you need to install the Python environment and the C/C++ environment.
It is recommended to install the Python environment via Anaconda. You need to go to the Anaconda official website(https://www.anaconda.com/download) to download and install Anaconda, and after installation, you will have the Python running environment.
As for the C/C++ running environment, it depends on the situation: if you are using a Windows system, you need to download the Visual Studio software(https://visualstudio.microsoft.com/); if you are using a Linux system, you can directly run sudo apt install gcc.

### 2.	Deploy the code
First, you need to download the code from GitHub; simply download the code zip package.
After downloading and extracting the code, you need to install the Python libraries required for the code to run by entering the following command in the terminal:
```
pip install -r requirements.txt
```
Additionally, run the following command to install the deep learning model runtime environment:
```
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cpuonly -c pytorch
```

We use the pypls library for PLS-DA and OPLS-DA analyses, so we need to follow the pypls library with reference to https://github.com/Omicometrics/pypls/tree/master.
Specifically, you first need to click the above link to download the pypls library code. After downloading, extract the compressed file. Then, navigate to the code directory, open the command line, and run the command</br>
```
python -m setup install
```

In addition to the above steps, you need to apply for openai API key to access the large model services. Specifically, follow these steps:
1.	Visit the openai homepage(https://platform.openai.com/docs/overview) to register and log in. Then create api key(https://platform.openai.com/api-keys).
2.	Open the app.py file and locate Line 110.
3.	Replace the placeholder with your newly obtained API key.
Finally, enter python app.py in the command line to run the platform code.</br>

Possible issues during the deployment process：
In FlaskUpload,
if you encounter the error cannot import secure_filename, change
```
from werkzeug
```
to 
```
from werkzeug.utils
```
If the error cannot import name 'FileStorage' appears after the change, split the import into two lines:
```
from werkzeug.utils import secure_filename  
from werkzeug.datastructures import FileStorage
```

<!-- USAGE EXAMPLES -->
## Usage
The front page of the file is as follows:</br>
![image](https://github.com/zza1211/NMRplatform/blob/master/tutorial_fig1.png)</br>
![image](https://github.com/zza1211/NMRplatform/blob/master/tutorial_fig2.png)</br>
Sample type, research content, research objects, and research purposes can be filled in according to the experiment. There are two options for file type: zip and csv. If the zip type is selected, upload the NMR data exported by the Bruker NMR instrument, with samples from different groups placed in separate compressed folders.<br>
Each time a new group is added, click the green plus sign once. If you choose to upload a CSV file, the first column of the CSV file must be chemical shifts, and starting from the second column are the sample spectra, with the column names being the sample names. Samples from different groups need to be stored in separate CSV files for upload.</br>
Then you can manipulate the data as needed.
