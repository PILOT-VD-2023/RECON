# RECON
# A Win-Win Game? Software Vulnerability Detection with Zero-Sum Game and Prototype Learning

## Dataset
To investigate the effectiveness of RECON, we adopt the vulnerability datasets from the paper:

* Fan et al. [1]
  
You can download the models by executing the following command or by visiting the following site：

    gdown https://drive.google.com/uc?id=12QVOd2fY9jYdJlqr5FfXW5yLBglR98OV
    
    https://drive.google.com/file/d/12QVOd2fY9jYdJlqr5FfXW5yLBglR98OV/view?usp=drive_link

## Environment Setup

    - Python: 3.8
    - Pytorch: 1.10.0+cu111
    - networkx: 2.8.5
    - numpy: 1.22.3
    - scikit-learn: 1.1.1
    - scipy: 1.8.1
    - tree-sitter: 0.20.0
    
## Tree-sitter 

    cd parserTool
    bash build.sh

## Train the model

    source train.sh

You can fine-tune the parameters for different RQs in this file.
  
## Test the model

You can download the models by executing the following command or by visiting the following site：

    gdown https://drive.google.com/uc?id=1bqqty2RxvjqNNE0iXVVzA3bGFILhP01P
  
    https://drive.google.com/file/d/1bqqty2RxvjqNNE0iXVVzA3bGFILhP01P/view?usp=drive_link

If you want to test the model quickly, you can execute the following command:

    source test.sh
    
## References
[1] Jiahao Fan, Yi Li, Shaohua Wang, and Tien Nguyen. 2020. A C/C++ Code Vulnerability Dataset with Code Changes and CVE Summaries. In The 2020 International Conference on Mining Software Repositories (MSR). IEEE.
