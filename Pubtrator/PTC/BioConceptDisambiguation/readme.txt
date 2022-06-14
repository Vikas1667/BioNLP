[Directory]
A. Introduction of folders
B. Installation
C. Instruction
D. FULL Usage

#======================================================================================#

A. [Introduction of folders]
	
	Library & Module:
		[Ab3P_Library]
		[model]
	Input data folder:
		[input]
	Output data folder:
		[output]
	TMP folder:
		[tmp]
	Corpus folder:
		[Corpus]
	
B. [Installation] 

	$ virtualenv -p python2.7 venv
	$ source venv/bin/activate
	$ pip install -r requirements.cpu.txt
					
C. [Instruction]

	$ python BD.py [inputfolder] [outputfolder] [tmpfolder]
	
	Example: python BD.py input output tmp
	
	[inputfolder] : User can provide the input data folder route (default is input)
	[outputfolder] : User can provide the output data folder route (default is output)
	[tmpfolder] : User can provide the tmp folder route (default is tmp)
	
D. [FULL Usage] 

	BioConceptDisambiguation is developed for resolving the annotation conflicts (overlapping annotations).

	INPUT:
	
		Input file folder. Each input file should follow the BioC format(http://bioc.sourceforge.net/) format. 

	RUN:
	
		python BD.py input output tmp
	
	OUTPUT:

		Output file folder. Each output file name is the same of the input file. 
	