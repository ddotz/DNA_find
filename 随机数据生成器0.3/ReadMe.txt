RandomDataGenerator 
Version: 0.2
Author: Qiang Yu, feqond@163.com

It supports the following discovery modes.
(1) OOPS
(2) ZOOPS
(3) TCM

It supports the follwoing planting modes, which determine the conservation of the planted motif.
(1) PLANTED_MODE_0_TO_D
(2) PLANTED_MODE_D_2_TO_D
(3) PLANTED_MODE_D: exactly d.
(4) PLANTED_MODE_RANDOM_REPLACE_D_POSITONS
(5) PLANTED_MODE_WEAK_CONSERVED: follwing the method in ref "A Cluster Refinement Algorithm for Motif Discovery" 
(6) PLANTED_MODE_HIGH_CONSERVED: follwing the method in ref "A Cluster Refinement Algorithm for Motif Discovery" 
(7) PLANTED_MODE_0_TO_D_HIGH_CONSERVED: follwing the method in ref BIBM manuscript of yu
(8) PLANTED_MODE_0_TO_D_MEDIA_CONSERVED: follwing the method in ref BIBM manuscript of yu
(9) PLANTED_MODE_0_TO_D_LOW_CONSERVED: follwing the method in ref BIBM manuscript of yu
	

Note:
(2) The generated random data is saved as data.txt in the standard fasta form.
(1) The supported maximum length of each sequence is 2000, and the maximum number of sequences is 1000,000.



