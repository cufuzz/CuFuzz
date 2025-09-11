import sys
import os
import re
## adding parents dir
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir) 

from establish_graph_from_json import establish_graph

from datetime import datetime






    

def api_coverage_calcu(harness_root, minute_step):
    ###  read harness dir to be test.  traverse every haness(like  2025xxx_01xxx), read the code file,
    ###  gain api sequence leveraging re.
    ###  minute_step means how many minutes to record a point .

    count = 0
    no_c = 0
    for harness_dir in os.listdir(harness_root):
        if "_" in harness_dir:
            one_harness_dir = os.path.join(harness_root, harness_dir)

            if len(os.listdir(one_harness_dir)) <2:
                print(one_harness_dir)
                no_c+=1
            for item in os.listdir(one_harness_dir):
                if '_' in item and ('.cu' not in item) and ('sep' not in item):
                    
                    count += 1
                    

    print(f"{count} / {len(os.listdir(harness_root))} = {round(count / len(os.listdir(harness_root)), 3)}")
    print(no_c)
    
   


                     




if __name__ == "__main__":
    

    root_path = '../experiments/cu-rt/rt-lib/harness'
    api_coverage_calcu(root_path, minute_step=25)

