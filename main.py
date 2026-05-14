import argparse
import os
import shutil
import subprocess
import json

import setproctitle
import torch

import convex_adversarial.examples.problems as pblm
from convex_adversarial.examples.trainer import evaluate_robust, robust_loss

print(torch.__file__)


# NOTE: This function assumes that whenever we run the sdp_verification, we check for all 
# adversarial classes even if the verifier returns non_robust against any earlier adversarial class
# This decision was taken to get a complete understanding of the bounds and time for each problem
def parse_log_file(filepath):
    with open(filepath, "r") as file:
        data = file.read()

    blocks = data.split("--------------------")

    inputs = {}
    num = 0
    for block in blocks:
        lines = [line.strip() for line in block.strip().split("\n") if line.strip()]
        if lines:
            test_input_path = [line for line in lines if line.startswith("test_input_path:")][0]
            input_key = test_input_path.split(":")[1].strip()

            row = [
                lines[0],  # true_class
                lines[1],  # test_input_path
                lines[2],  # adv_class
                lines[3],  # bound
                lines[4],  # solve time
            ]

            if input_key not in inputs:
                inputs[input_key] = []
            inputs[input_key].append(row)

    results = {}
    total_sum = 0
    total_sum_sdp = 0
    robust_samples_filepath = []

    for input_key, rows in inputs.items():
        total_time = 0
        total_time_sdp = 0
        found_positive = False

        for row in rows:
            if not found_positive and float(row[4]) != -1:
                total_time += float(row[4])
                if float(row[3]) > 0:
                    found_positive = True
                    num = num + 1
        
        if found_positive is False:
            robust_samples_filepath.append(input_key)

        results[input_key] = total_time
        total_sum += total_time

        for row_sdp in rows:
            if float(row[4]) != -1:
                total_time_sdp += float(row_sdp[4])  # Add the value in the fourth cell (row[4])

        results[input_key] = total_time_sdp
        total_sum_sdp += total_time_sdp

    for input_key, total_time in results.items():
        print(f"Input {input_key}: Total Time = {total_time:.2f}")

    average_time = (total_sum / len(results))/60
    print(f"\nAverage Time for all 50 inputs: {average_time:.2f}")

    average_time_sdp = (total_sum_sdp / 450)/60
    print(f"\nAverage sdp-Time for all 50 inputs: {average_time_sdp:.2f}")
    print(num)
    
    return robust_samples_filepath

def find_union_of_filepaths(list_1, list_2):
    union = set()
    
    for filepath in list_1:
        union.add(os.path.basename(filepath))
        
    for filepath in list_2:
        union.add(os.path.basename(filepath))
        
    return list(union)
        

if not torch.cuda.is_available():
    torch.Tensor.cuda = lambda self, device=None, non_blocking=False: self

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--starting_epsilon", type=float, default=0.05)
    parser.add_argument("--prefix")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--alpha_grad", action="store_true")
    parser.add_argument("--scatter_grad", action="store_true")
    parser.add_argument("--l1_proj", type=int, default=None)
    args = parser.parse_args()
    args.prefix = args.prefix or "mnist_conv_{:.4f}_{:.4f}_0".format(
        args.epsilon, args.lr
    ).replace(".", "_")
    setproctitle.setproctitle(args.prefix)

    train_log = open(args.prefix + "_train.log", "w")
    test_log = open(args.prefix + "_test.log", "w")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    model = pblm.mnist_model().cuda()

    lp_model_path = args.model_path
    print(lp_model_path)

    model.load_state_dict(torch.load(lp_model_path))

    # model.load_state_dict(torch.load("trained_models_lp_on_lp/mnist_model.pth"))

    # Set the model to evaluation mode (for inference)
    model.eval()

    # NOTE: All the inputs should of the form <input_name>_<label>.py
    test_loader, filepaths = pblm.custom_mnist_loaders("custom_inputs", 1)
        
    # This runs the code for LP verification 
    # Modified and extended from the original implementation (https://github.com/locuslab/convex_adversarial)
    robust, non_robust = evaluate_robust(
        test_loader, model, args.epsilon, -1, test_log, args.verbose
    )
    print("LP Robust")
    print(robust)
    
    ## Now, using the robust inputs list, we will create a folder structure needed run the SDP certification
    sdp_input_foldername = "sdp_inputs"
    os.makedirs(sdp_input_foldername, exist_ok=True)
    
    lp_robust_samples_filepaths = [filepaths[i] for i in robust]
    lp_non_robust_samples_filepaths = [filepaths[i] for i in non_robust]
    print(lp_non_robust_samples_filepaths)
    
    sdp_robust_samples_filepath = []
    

    # Ensures that we only go to SDP when some inputs have been verified as non-robust by LP
    if len(lp_non_robust_samples_filepaths)!=0:
        for filepath in lp_non_robust_samples_filepaths:
            filename = os.path.basename(filepath)
            label = filename.split("_")[1][0]
            
            if str(label) not in os.listdir(sdp_input_foldername):
                os.mkdir(os.path.join(sdp_input_foldername, str(label)))
            
            shutil.copy(filepath, os.path.join(sdp_input_foldername, str(label), filename))
            
        ## Now, we will initiate the sdp_verification through a shell script

        # SDP implementation is modified and extended from the original implementation (https://worksheets.codalab.org/worksheets/0x6933b8cdbbfd424584062cdf40865f30/)
        sdp_run_script_path = 'run_sdp.sh'
        subprocess.run(['chmod', "+x", sdp_run_script_path])
        subprocess.run(["bash", sdp_run_script_path])
        
        # After SDP script has finished running, we will use the generated logs folder to find 
        # out the samples that are robust
        sdp_robust_samples_filepath = parse_log_file('sdp/code/logs/output.txt')
    
    final_robust_samples = find_union_of_filepaths(sdp_robust_samples_filepath, lp_robust_samples_filepaths)
    
    # Calculate the final robust accuracy
    robust_accuracy = len(final_robust_samples)/len(filepaths)
    
    print(robust_accuracy)
