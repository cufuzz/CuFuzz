# Descripiton
This is a tool for fuzzing CUDA Libraries, named CuFuzz.
Currently supports cuRT(Runtime), cuBlas, cuFFT, cuNPP, cuSOLVER, cuSPARSE, nvJPEG, and cuRAND , total 8 libraries.

# Environment
Refer to the requirements and follow the Python packages inside. It is recommended to use conda and Python>=3.10

# Quick start
Configure in config.yaml, such as selecting the target library to test. 
Then run:
```
python gen_harness.py
```
During the fuzzing process, the results will be printed on the terminal and saved in log.txt

# Structure
gen_harness.py : It is mean stream and fuzzing entry.

gen_graph_from_cuda_sample.py and gen_graph_from_cuda_tutorial-2.py : These two are respectively used to extract knowledge graphs from official documents. I have already extracted the knowledge graphs and can use them directly

Folder c_factors : It is a pre compiled c-based mutation operator, which has been compiled and can be used directly.

Folder anlysis: Store experimental data analysis code.

Folder cublas et.al. : Store the fuzz results of each library, as well as their respective knowledge graphs, etc

# Experiment data
The experimental data and our POC are stored in the exp_data folder
