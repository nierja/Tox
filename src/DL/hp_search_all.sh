#!/bin/bash
cd /auto/brno2/home/nierja/Tox/src/DL
targets=("NR-AR") # "NR-AR-LBD" "NR-AhR" "NR-Aromatase" "NR-ER" "NR-ER-LBD" "NR-PPAR-gamma" "SR-ARE" "SR-ATAD5" "SR-HSE" "SR-MMP" "SR-p53")
# descriptors=('ecfp4')
descriptors=('ecfp4_maccs' 'maccs_rdk7' 'ecfp4_rdk7' 'ecfp4' 'ecfp6' 'fcfp4' 'fcfp6' 'maccs' 'hashap' 'hashtt' 'avalon' 'rdk7' 'eigenvals' 'rdkit_descr' 'CMat_600' 'mordred')
# descriptors=('ecfp4_maccs' 'maccs_rdk7' 'ecfp4_rdk7' 'ecfp0' 'ecfp2' 'ecfp4' 'ecfp6' 'fcfp2' 'fcfp4' 'fcfp6' 'maccs' 'hashap' 'hashtt' 'avalon' 'rdk5' 'rdk6' 'rdk7' 'eigenvals' 'rdkit_descr' 'CMat_400' 'CMat_600' 'mordred')
pcas=('0' '512') # '1024')
weights=('False') # 'True')
ensambles=('7') # '17' '37')

for target in ${targets[@]}; do
	for fp in ${descriptors[@]}; do
        	for pca in ${pcas[@]}; do
			for weight in ${weights[@]}; do
				for ensamble in ${ensambles[@]}; do
  		    			qsub -v fp=$fp,target=$target,pca=$pca,weight=$weight,ensamble=$ensamble hp_job.sh
				done
			done
        	done
	done
done
