# SoftPower

This project examines the portability of deep-learning side-channel attacks against software discrepancies. Specifically, we examine the cases where training data and test data are discrepant in the context of side-channel attacks, and these discrepancies are primarily caused by different software settings between the training device and test deivce. 

We investigate four factors that can lead to software discrepancies, including random delays, instruction rewriting, optimization levels, and code obfuscation. Specifically, we simulate random delays by randomly shifting power measurements. We examine instruction rewriting at the assembly level and also over ELF (Executable and Linkable Format) files by leveraging a reverse engineering tool Ghidra. We investigate four optimization levels, including Os, O1, O2, and O3 given a cross-compiler. We explore three code obfuscations offered by a code obfuscation tool Tigress.

We collect a large-scale dataset, named SoftPower dataset. It consists of more than 3.2 million power traces (187 GBs) of AES-128 encryption from two types of microcontrollers, including AVR XMEGA (8-bit RISC) and ARM STM32 (32-bit Cortex-M4), using ChipWhisperer and various software settings associated with the four discrepancy factors we examine.

Experimental results suggest that every software discrepancy factor we examine leads to attack performance drops. Specifically, it takes a much greater number of test traces for a CNN (Convolutional Neural Network) to reveal encryption keys. Moreover, discrepancies caused by instruction rewriting, optimization levels, or code obfuscation can even result in the failure of recovering keys using a CNN.

Our results suggest that adjusting POI (Points of Interest) can improve attack performance against discrepancies caused by instruction rewriting, optimization levels, or code obfuscation. Domain adaptation is only effective in few examples associated with random delays. Multi-domain training can overcome discrepancies caused by every factor and effectively reveal keys. However, it has to significantly demote the attack assumption by requiring an attacker to know every possible software setting in advance. Although this is potentially feasible, it could be difficult to scale in practice given the diversity of software discrepancies between the training device and test device. 


## Reference
When reporting results that use the dataset or code in this repository, please cite the paper below:

*Chenggang Wang, Mabon Ninan, Shane Reilly, Joel Ward, William Hawkins, Boyang Wang, John M Emmert, "Portability of Deep-Learning Side-Channel Attacks against Software Discrepancies," In Proceedings of the 16th ACM Conference on Security and Privacy in Wireless and Mobile
Networks (WiSec’23), May 29-June 1, 2023, Guildford, United Kingdom.*

** The dataset and code are for research purpose only**

## Requirements
This project is written in Python 3.6, Tensorflow 2.3.1 and Pytorch 1.8.1. Our experiments is running with a GPU machine.

## Reproduce our results
### Code 
The codebased include 3 folders: cnn, triplet and tools
>
> - cnn: include the codes for runing cnn method
> - ada: include the codes for runing triplet method
> - tools: include the codes of all support functions
>

### Datasets
Our datasets used in this study can be accessed through the link below (last modified May 2023):

https://mailuc-my.sharepoint.com/:f:/g/personal/wang2ba_ucmail_uc_edu/EqBa62pr1EtPlW7PiiF0bSkBMavkyNOPfwOompfV3yf6ew?e=ki4Hoh

Note: the above link need to be updated every 6 months due to certain settings of OneDrive. If you find the links are expired and you cannot access the data, please feel free to email us (boyang.wang@uc.edu). We will be update the links as soon as we can. Thanks!


### How to Reproduce the results
1. For CNN based method, please follow the description in cnn/README.md
2. For ADA based method, please follow the description in ada/README.md


# Contacts
Mabon Ninan ninanmm@mail.uc.edu

Boyang Wang wang2ba@ucmail.uc.edu

Chenggang Wang wang2c9@mail.uc.edu

