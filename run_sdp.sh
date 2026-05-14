#!/bin/bash
# Activate virtual environment
source ~/virtual-environments/sdp/bin/activate

# Navigate to the working directory
cd '<REPLACE WITH THE ABSOLUTE PATH TO /sdp/code>'

# Run the Python script with arguments
python certify.py \
  --matlab_folder '<REPLACE WITH THE ABSOLUTE PATH TO /sdp>' \
  --checkpoint '../models/nips_sdp.ckpt' \
  --model_json '../model_details/nips_sdp.json'
