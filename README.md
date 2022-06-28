# extra-lowRD
·  Why DETexT?

·  input & output

·  model structure

·  citation


·  Why DETexT?
Sequencing cost is positively correlated with the read depth. The detection of SNPs at extremely low read depth (1X ) is an arduous task pending. Specifically, the existing methods requires multiple read pairs support (at least 34X) and their accuracies are greatly reduced when the read depth is extremely low, and mutations that are orders of magnitude lower than the depth are excluded. However, the in-depth and accurate detection of SNP and other mutations in extremely low read depth is of great significance for reducing sequencing costs, early cancer screening and precise diagnosis and treatment, and is the basic requirement for the application of liquid biopsy. Here, we proposed a differential SNP detection method that is more suitable for the sequence information convolute, and use textCNN to make better use of the base sequence information from a text-like perspective. In addition, we considered mutational signatures as a priori information. We verified the method on the SNP detection of 1X simulating panel sequencing data of the ESCC, and the results demonstrated it can perform outstanding in a manner independent of sequencing depth and all three major innovations made sense. 

·  input & output
DETexT support the SAM file as the input and output filtered mutation in Variant Call Format (VCF file)
![image](https://user-images.githubusercontent.com/37039216/175777569-c2d5a6ba-0226-4ae4-9402-497ed5a6cf0a.png)

·  model structure
![image](https://user-images.githubusercontent.com/37039216/175775702-aa39a625-cd3f-41da-98da-25b398e933ef.png)

![image](https://user-images.githubusercontent.com/37039216/175777720-74378b60-10aa-48ae-bfd1-f87066ef4fa3.png)


Possible disadvantages of the method: 

	1) File processing time is not short enough 
	
	2) One-click operation is not fully integrated at the moment, you need to run the file in steps 
	
Ease of use for users are: 

	1) simple network, easy to use and understand for non-computer professional users 
	
	2) simple to use, not too many dependency packages 
	
	3) low computing resource requirements, excellent results
	


·  citation
