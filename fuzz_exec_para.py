import os
import threading
import time
from tqdm import tqdm
from CUDA import CudaTargetController, CudaAPIController
import random
from queue import Queue


# Define a task class to manage each fuzzing task
def fuzz_task(api, timelimit_per_program, cpu_semaphore, round,
			working_dir, result_queue, index):
	try:
		test = CudaAPIController(
			api=api,
			timelimit_per_program=timelimit_per_program,
			cpu_semaphore=cpu_semaphore,
			working_dir=working_dir,
			round=round
		)
		result = test.start_process()
		result_queue.put((index, result))
	except Exception as e:
		result_queue.put((index, -1))
	# finally:
	# tqdm.write(f"Task for API {api} completed.")


def fuzz(**kwargs):
	cpu_semaphore = threading.Semaphore(kwargs["pre_fuzz_cpu_limit"])           ##  5
	# cpu_semaphore2 = threading.Semaphore(kwargs["fuzz_cpu_limit"])
	api_controller_semaphore = threading.Semaphore(kwargs["fuzz_cpu_limit"])   ##  10

	if kwargs["target"] == "cuda":
		fuzzer = CudaTargetController(**kwargs)
	else:
		raise NotImplementedError(f"Fuzzing target {kwargs['target']} is not implemented.")

	os.makedirs(kwargs["working_dir"], exist_ok=True)
	# if kwargs["mode"] == 'nvjpeg':
	# 	os.system(f'cp {kwargs["working_dir"]}/../harness/OIP.jpg {kwargs["working_dir"]}')

	timelimit_per_program = kwargs['timelimit_per_program']

	threads = []
	result_queue = Queue()
	return_list = [0] * len(fuzzer.exec_list)
	for idx, api in enumerate(tqdm(fuzzer.exec_list, desc="exec_program Progress", leave=False, mininterval=0.1)):
		# 这里使用信号量来限制并发
		with api_controller_semaphore:  # 限制 API 同时执行数量
			thread = threading.Thread(target=fuzz_task, args=(
				api,
				timelimit_per_program,
				cpu_semaphore,
				fuzzer.round_dict[api],
				fuzzer.folder,
				result_queue,
				idx))
			threads.append(thread)
			thread.start()

		# 等待所有线程完成
	for thread in threads:
		thread.join()

	# 按索引排序结果
	result_list = [-1] * len(fuzzer.exec_list)
	while not result_queue.empty():
		idx, result = result_queue.get()
		result_list[idx] = result

	return result_list


if __name__ == "__main__":
	target = "cuda"

	current_directory = os.getcwd()
	working_dir = f"{current_directory}/nvjpeg/fuzz_output_threads"
	exec_list_path = [f"{current_directory}/nvjpeg/harness/20250509_215525"]
	round_list = [3]             #  round times for one harness

	timelimit_per_program = 30
	#timelimit_for_last_program = 86400
	verbose_level = 3
	fuzz_cpu_limit = 20
	pre_fuzz_cpu_limit = 5
	fuzz(
		target=target, ##
		exec_list_path=exec_list_path,  ##
		round_list=round_list,
		timelimit_per_program=timelimit_per_program,  ##
		working_dir=working_dir, ##
		verbose_level=verbose_level, ##
		fuzz_cpu_limit=fuzz_cpu_limit,
		pre_fuzz_cpu_limit=pre_fuzz_cpu_limit,
		mode='nvjpeg'
	)
