
Commands
--------
python step1a_generate.py /hpc/users/chiup04/work/data/pf3

python combine.py /hpc/users/chiup04/work/data/pf3

python -W ignore cf.py /hpc/users/chiup04/work/data/pf3


1. python step1a_generate.py <project_path> 
     e.g. /Users/chiup04/Documents/work/data/pf2
     
     > generates BP predictions
     
     > step1a_generate is a modified version of generate.py; supports nested CV and flat CV
     
2. python combine.py <project_path>
   
   > this creates validation-<i>.csv.gz
    		      predictions-<i>.csv.gz
    		      i = 0 ~ outer_fold-1
    		      
   in stacking.py 
      > validation => train split
        predictions => test split
    
3. meta learner 
 
   a. stacker 
   
   b. cf



# Paper: ﻿Hu Y, Volinsky C, Koren Y. Collaborative filtering for implicit feedback datasets. 
In: Proceedings - IEEE International Conference on Data Mining, 
ICDM [Internet]. IEEE; 2008 [cited 2019 Jan 14]. p. 263–72. 
Available from: http://ieeexplore.ieee.org/document/4781121/

+ in Eq (3)
C<u,i> can be a function(al) of users/classifier and items/data
=> used a weights 
=> used a device for selecting appropriate classifiers and data (removing k outliers or kNN)


