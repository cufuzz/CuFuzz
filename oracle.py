import subprocess
import time
import os
import numpy as np
import logging
import threading

logger = logging.getLogger()

from enum import Enum
class FResult(Enum):
    SAFE = 1  # validation returns okay
    FAILURE = 2  # validation contains error (something wrong with validation)
    ERROR = 3  # validation returns a potential error (look into)
    LLM_WEAKNESS = (
        4  # the generated input is ill-formed due to the weakness of the language model
    )
    TIMED_OUT = 10  # timed out, can be okay in certain targets


def oracle_report(exec_command, timeout):
    if not isinstance(exec_command, list):
        exec_command = [exec_command]

    try:
        # 启动子进程
        env = os.environ.copy()
        process = subprocess.Popen(
            exec_command,
            #shell=True,
            stdout=subprocess.PIPE,  # 捕获标准输出
            stderr=subprocess.PIPE,  # 捕获标准错误
            text=True,  # 输出以字符串形式处理
            bufsize=1,
            env=env,
            cwd=os.path.dirname(exec_command[0])
        )
        print(f"Started fuzzing with command: {exec_command}")
        logging.info(f"Started fuzzing with command: {exec_command}")
        print(f"Started fuzzing with PID: {process.pid}")

        def timeout_handler():
            print(f"TIMEOUT STOPPING fuzzing: {exec_command}")
            process.terminate()  # 终止子进程
            process.kill()  # 强制杀死子进程
            # raise TimeoutError(f"timeout! the preset: {timeout}s")

        # 设置定时器
        timer = threading.Timer(timeout, timeout_handler)
        timer.start()

        # 实时捕获子进程输出并处理
        while True:
            output = process.stdout.readline()
            if not output and process.poll() is not None:
                break
            if output:
                print(output.strip())

        # 等待子进程结束
        timer.cancel()  # 取消定时器
        process.wait()

        print("EXITCODE:", process.returncode)
        if process.returncode != 0:
            if process.returncode == 1:
                return FResult.FAILURE # that is 2
            elif process.returncode == -15:
                return FResult.TIMED_OUT
            else:
                return FResult.ERROR  # that is 3
        else:
            return FResult.SAFE  # that is 1

    except TimeoutError as e:
        print(str(e))
        return FResult.TIMED_OUT  # that is 10
    except Exception as e:
        print(f"ERROR: {e}")
        return FResult.FAILURE  # that is 2

    #     start_time = time.time()
    #
    #     # 实时捕获子进程输出并处理
    #     while process.poll() is None:  # 检查子进程是否结束
    #         # print(time.time() - start_time)
    #         if time.time() - start_time > timeout:  # 检测是否超时
    #             print(f"TIMEOUT STOPING fuzzing: {exec_command}")
    #             process.terminate()  # 终止子进程
    #             subprocess.run(f'kill -9 {process.pid}', shell=True)
    #             raise TimeoutError(f"timeout! the preset: {timeout}s")
    #
    #         output = process.stdout.readline()
    #         if output:
    #             print(output.strip())
    #         time.sleep(0.1)
    #
    #     print("EXITCODE:", process.returncode)
    #     if process.returncode !=0:
    #         return FResult.ERROR   ### that is 3
    #     else:
    #         return FResult.SAFE     ### that is 1
    # except TimeoutError as e:
    #     print(str(e))
    #     return FResult.TIMED_OUT    ### that is 10
    # except Exception as e:
    #     print(f"ERROR: {e}")
    #     return FResult.FAILURE       ### that is 2


class celibrated_harness():
    """
    celibrating a harness, giving the score according to the consistency of api_list and the execution result .
    """
    def __init__(self, the_time_flame, request_api_list, generated_api_list, exec_oracle, round):
        self.request_api_list = request_api_list
        self.generated_api_list = generated_api_list
        self.oracle = exec_oracle
        self.ID = the_time_flame
        self.round = round
        self.score = 1
        self.consistence = 1
        self.mutated_time = 0
        self.calculate_score()


    def calculate_score(self):
        if (set(self.request_api_list) == set(self.generated_api_list)):
            self.score = 2
        else:
            self.consistence = 0
        if (self.oracle == FResult.ERROR):
            self.score = 5
        if (self.oracle == FResult.TIMED_OUT):
            self.score = 3


    def update(self, new_oracle, new_round=None):
        if (new_oracle == FResult.ERROR):
            self.score = max(self.score, 6)
        if (new_oracle == FResult.TIMED_OUT):
            self.score = max(self.score, 4)

        if new_round:
            self.round += 1

    def mutate_add(self):
        self.mutated_time += 1

def get_top_indices(all_harness_list) -> list:

    if not all_harness_list:  # 确保列表不为空
        return []

    n = len(all_harness_list)
    if n < 5:
        return list(range(n))

    # First filter out harnesses with score > 3
    valid_harnesses = [harness for harness in all_harness_list if harness.score <= 3]
    valid_indices = [i for i, harness in enumerate(all_harness_list) if harness.score <= 3]

    # If after filtering we have less than 5, return all valid indices
    if len(valid_harnesses) < 5:
        return valid_indices

    ##  Logarithmic Normalized Probability Sampling
    scores = [len(harness.generated_api_list) * 0.5 for harness in all_harness_list if harness.score <= 3]
    exp_scores = np.exp(scores )
    probabilities = exp_scores / np.sum(exp_scores)

    # sample according to probabilities
    sampled_indices = np.random.choice(range(len(valid_harnesses)), size=min(5, len(valid_harnesses)),
                                       replace=False, p=probabilities)

    return [valid_indices[i] for i in sampled_indices]


def get_one_harness_candi_mutate(all_harness_list) -> list:
    if not all_harness_list:  # 确保列表不为空
        return []

        # First filter out harnesses with score > 3
    valid_indices = [i for i, harness in enumerate(all_harness_list)]

    ##  Logarithmic Normalized Probability Sampling
    scores = [len(harness.generated_api_list) * 0.5 for harness in all_harness_list]
    counts = [harness.mutated_time for harness in all_harness_list]
    counts = np.array(counts)
    exp_scores = np.exp(scores)

    decay_index = 1
    probabilities = exp_scores / np.sum(exp_scores) / (1 + counts) ** decay_index

    # normalization
    sum_new_probs = np.sum(probabilities)
    normalized_probs = probabilities / sum_new_probs

    # sample according to probabilities
    sampled_indices = np.random.choice(range(len(valid_indices)), size=1,
                                       replace=False, p=normalized_probs)

    return valid_indices[sampled_indices[0]]



if __name__ == '__main__':

    # command = f'/home/fanximing/fuzz4all/cuda-graph-llm/fuzz_output_threads/095227_sep_wrap_round1_0/095227_sep_wrap'
    # oracle_report(command, 30)

    all_harness_list = [
        celibrated_harness(the_time_flame=1, request_api_list=[1, 2], generated_api_list=[1, 2],
                           exec_oracle=FResult.SAFE, round=1),
        celibrated_harness(the_time_flame=2, request_api_list=[1, 2], generated_api_list=[1, 2, 3],
                           exec_oracle=FResult.SAFE, round=2),
        celibrated_harness(the_time_flame=3, request_api_list=[1, 2], generated_api_list=[1, 2],
                           exec_oracle=FResult.ERROR, round=1),
        celibrated_harness(the_time_flame=4, request_api_list=[1, 2], generated_api_list=[1, 2, 3, 4],
                           exec_oracle=FResult.SAFE, round=3),
        celibrated_harness(the_time_flame=5, request_api_list=[1, 2], generated_api_list=[1, 2],
                           exec_oracle=FResult.TIMED_OUT, round=1),
        celibrated_harness(the_time_flame=6, request_api_list=[1, 2], generated_api_list=[1, 2, 3, 4, 5],
                           exec_oracle=FResult.SAFE, round=2)
    ]

    # top_indices = get_top_indices(all_harness_list)
    # print("Top indices:", top_indices)
    ###  20250401_222211: EXIT_FAILURE    20250401_222314:segmentation fault

    aa = oracle_report(f'/home/fanximing/cuda-graph-llm/rt-lib/fuzz_output_threads/20250520_205726_sep_wrap_round1/20250520_205726_sep_wrap', 60)
    print(aa)
    print(11111)