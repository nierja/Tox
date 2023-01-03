#!/bin/bash
cd /auto/brno2/home/nierja/Tox/src/DL
targets=("NR-AR" "NR-AR-LBD" "NR-AhR" "NR-Aromatase" "NR-ER" "NR-ER-LBD" "NR-PPAR-gamma" "SR-ARE" "SR-ATAD5" "SR-HSE" "SR-MMP" "SR-p53")
targets=("SR-p53")
# descriptors=('mordred')
# descriptors=('ecfp4_maccs' 'maccs_rdk7' 'ecfp4_rdk7' 'ecfp4' 'ecfp6' 'fcfp4' 'fcfp6' 'maccs' 'hashap' 'hashtt' 'avalon' 'rdk7' 'eigenvals' 'rdkit_descr')
descriptors=('ecfp4_maccs' 'maccs_rdk7' 'ecfp4_rdk7' 'ecfp0' 'ecfp2' 'ecfp4' 'ecfp6' 'fcfp2' 'fcfp4' 'fcfp6' 'maccs' 'hashap' 'hashtt' 'avalon' 'rdk5' 'rdk6' 'rdk7' 'eigenvals' 'rdkit_descr' 'CMat_400' 'CMat_600' 'mordred')

short=('maccs' 'eigenvals' 'rdkit_descr' 'CMat_400')


i=1
pcas=('0' '512' '1024')
weights=('False' 'True')
ensambles=('7') # '3' '11')

for target in ${targets[@]}; do
	for fp in ${descriptors[@]}; do
		for pca in ${pcas[@]}; do
			for weight in ${weights[@]}; do
				for ensamble in ${ensambles[@]}; do

					# compute needed walltime
					time=6
					if [ "$target" = 'NR-Aromatase' ] || 
					   [ "$target" = 'SR-MMP' ]; then
						time=$(( time+4 ))
					fi

					if [ "$fp" = 'ecfp4_rdk7' ] || 
					   [ "$fp" = 'mordred' ]; then
						time=$(( 5*time ))
					fi

					if [ "$pca" = '1024' ] && 
					   [ "$fp" != 'mordred' ]; then
						continue
					fi

					too_short=0
					for short_fingerprint in "${short[@]}"; do
					    if [ $fp == "$short_fingerprint" ]; then
							too_short=1
						fi
					done

					if [ $too_short = "1" ] && 
					   (( pca > 0 )); then
						continue
					fi

					printf "job %s\t\t TARGET=%s\t FP=%s\t TIME=%s:00:00 ( %s, %s, %s )\n" $i $target $fp $time $pca $weight $ensamble

					qsub -l walltime=$time:00:00 -v fp=$fp,target=$target,pca=$pca,weight=$weight,ensamble=$ensamble hp_job.sh

					printf "\n"
					i=$(( i+1 ))
				done
			done
		done
	done
done

exit 0