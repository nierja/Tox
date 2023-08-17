#!/usr/bin/env python3
import os
import subprocess

# uncomment thete to generate all descriptors for all targets
targets = ["NR-AR", ] # "NR-AhR", "NR-AR-LBD", "NR-Aromatase", "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"]
descriptors = [ 'maccs', ] # 'ecfp4', 'ecfp4_maccs', 'maccs_rdk7', 'ecfp4_rdk7', 'ecfp0', 'ecfp2', 'ecfp6', 'fcfp2', 'fcfp4', 'fcfp6', 'hashap', 'hashtt', 'avalon', 'rdk5', 'rdk6', 'rdk7', 'eigenvals', 'rdkit_descr', 'CMat_400', 'CMat_600', 'mordred', ]
datasets = [ "training", "test", "eval" ]

spacer = '------------------------------------------------------------------------'

def main (  ):
    os.chdir('./src/descriptor_generation')
    try:
        os.mkdir('../../data/Tox21_descriptors')
    except OSError:
        pass

    for descriptor in descriptors:
        for target in targets:
            for dataset in datasets:
                print('\n', spacer)
                print(f'Running "python3 generate_{dataset}_data.py --target={target}"\n')

                pro = subprocess.Popen(
                    f'python3 generate_{dataset}_data.py --target={target} --fp={descriptor}', 
                    stdout=subprocess.PIPE, 
                    shell=True, 
                    preexec_fn=os.setsid
                ) 

if __name__ == "__main__":
    main (  )
