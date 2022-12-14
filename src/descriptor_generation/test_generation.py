#!/usr/bin/env python3
import os
import signal
import subprocess
import time

targets = ["NR-AR", "NR-AHR", "NR-AR-LBD", "NR-Aromatase", "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"]
descriptors = [ "maccs" ]
datasets = [ "training", "test", "eval" ]

spacer = '------------------------------------------------------------------------'

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
            time.sleep(10)
            os.killpg(os.getpgid(pro.pid), signal.SIGTERM) 

