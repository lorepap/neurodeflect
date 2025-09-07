#!/usr/bin/env python3

"""
RL Policy Results Extractor

This script extracts data from RL policy simulation results.
Adapted from extractor_shell_creator.py to handle RL-specific file naming patterns.
"""

import sys
import hashlib
from os import listdir, makedirs
from os.path import isfile, join, exists

RESULT_FILES_DIR = './'  # Current directory
TOPOLOGY = 2  # LEAF_SPINE
REP_NUM = 1
SIMS_DIR = '/home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims'
OUTPUT_FILE_DIRECTORY = f'{SIMS_DIR}/extracted_results/'

if len(sys.argv) > 1:
    add_category = sys.argv[1]
else:
    add_category = 'rl_policy'

category = add_category

print(f"RL Policy Extractor starting...")
print(f"Looking for RL policy result files in: {RESULT_FILES_DIR}")
print(f"Output directory: {OUTPUT_FILE_DIRECTORY}")
print(f"Category: {category}")

# Look for .vec files with _rl_policy pattern (instead of _rep_)
onlyfiles = [f for f in listdir(RESULT_FILES_DIR) if
             isfile(join(RESULT_FILES_DIR, f)) and f[-4:] == '.vec' and '_rl_policy' in f]

print(f"Found {len(onlyfiles)} RL policy .vec files:")
for f in onlyfiles:
    print(f"  - {f}")

if len(onlyfiles) == 0:
    print("ERROR: No RL policy .vec files found!")
    print("Expected files with pattern: *_rl_policy.vec")
    sys.exit(1)

# Ensure output directories exist
output_dirs = [
    'FLOW_ENDED', 'FLOW_STARTED', 'REQUEST_SENT', 'REPLY_LENGTH_ASKED',
    'REQUESTER_ID', 'SYN_SENT', 'SYN_SENT_IS_BURSTY', 'FIN_ACK_RCV',
    'OOO_SEG_NUM', 'UTILIZATION', 'PER_PACKET_FABRIC_DELAY_COUNTS',
    'PER_PACKET_FABRIC_DELAY_SUM', 'V2_RCVD_SOONER', 'V2_RCVD_CORRECTLY',
    'V2_RCVD_LATER', 'NUM_MARKING_TIMEOUTS', 'QUEUE_LEN', 'PACKET_ACTION'
]

for output_dir in output_dirs:
    full_dir = OUTPUT_FILE_DIRECTORY + output_dir
    if not exists(full_dir):
        makedirs(full_dir, exist_ok=True)
        print(f"Created directory: {full_dir}")

# Write extractor script
f = open('../../../extractor_rl_policy.sh', 'w')
f.write('#!/bin/bash\n')
f.write('# Auto-generated RL policy extractor script\n')
f.write(f'# Generated for {len(onlyfiles)} RL policy result files\n\n')

for file_name in onlyfiles:
    print(f"\nProcessing file: {file_name}")
    
    # Parse RL policy filename components
    # Example: 4_spines_8_aggs_40_servers_1_burstyapps_1_mice_40_reqPerBurst_11.85_bgintermult_1_bgfsizemult_0.11_burstyintermult_1_burstyfsizemult_250_ttl_0_rep_2_rndfwfactor_2_rndbouncefactor_20000_incastfsize_0.00120_mrktimer_0.00120_ordtimer_rl_policy.vec
    
    try:
        num_spines = file_name.split('_spines')[0]
        num_aggs = (file_name.split('_aggs')[0]).split('_')[-1]
        num_servers_under_each_rack = (file_name.split('_servers_')[0]).split('_')[-1]
        num_bursty_apps = (file_name.split('_burstyapps')[0]).split('_')[-1]
        num_req_per_burst = (file_name.split('_reqPerBurst')[0]).split('_')[-1]
        num_mice_flows_per_server = (file_name.split('_mice')[0]).split('_')[-1]
        bg_inter_arrival_multiplier = (file_name.split('_bgintermult')[0]).split('_')[-1]
        bg_flow_size_multiplier = (file_name.split('_bgfsizemult')[0]).split('_')[-1]
        bursty_inter_arrival_multiplier = (file_name.split('_burstyintermult')[0]).split('_')[-1]
        bursty_flow_size_multiplier = (file_name.split('_burstyfsizemult')[0]).split('_')[-1]
        ttl = (file_name.split('_ttl_')[0]).split('_')[-1]
        
        # For RL policy files, we need to handle the different ending pattern
        if '_rep_' in file_name:
            # Standard pattern with repetition number
            rep_part = file_name.split('_rep_')[1]
            rep_num = rep_part.split('_')[0]
            remaining = '_'.join(rep_part.split('_')[1:])
        else:
            # RL policy pattern - no rep number, but has other parameters
            rep_num = "0"  # Default rep number for RL policy
            remaining = file_name.split('_ttl_')[1]
        
        # Parse remaining parameters
        parts = remaining.replace('_rl_policy.vec', '').split('_')
        
        # Extract parameters with defaults
        random_power_factor = "2"
        random_power_bounce_factor = "2" 
        incast_flow_size = "20000"
        marking_timer = "0.00120"
        ordering_timer = "0.00120"
        
        # Try to extract actual values if they exist
        for i, part in enumerate(parts):
            if part == "rndfwfactor" and i+1 < len(parts):
                random_power_factor = parts[i+1]
            elif part == "rndbouncefactor" and i+1 < len(parts):
                random_power_bounce_factor = parts[i+1]
            elif part == "incastfsize" and i+1 < len(parts):
                incast_flow_size = parts[i+1]
            elif part == "mrktimer" and i+1 < len(parts):
                marking_timer = parts[i+1]
            elif part == "ordtimer" and i+1 < len(parts):
                ordering_timer = parts[i+1]
                
    except Exception as e:
        print(f"Error parsing filename {file_name}: {e}")
        # Use defaults for problematic files
        num_spines = "4"
        num_aggs = "8"
        num_servers_under_each_rack = "40"
        num_bursty_apps = "1"
        num_req_per_burst = "40"
        num_mice_flows_per_server = "1"
        bg_inter_arrival_multiplier = "11.85"
        bg_flow_size_multiplier = "1"
        bursty_inter_arrival_multiplier = "0.11"
        bursty_flow_size_multiplier = "1"
        ttl = "250"
        rep_num = "0"
        random_power_factor = "2"
        random_power_bounce_factor = "2"
        incast_flow_size = "20000"
        marking_timer = "0.00120"
        ordering_timer = "0.00120"

    # Use the actual filename instead of reconstructing it
    vector_file_name = file_name
    scalar_file_name = file_name.replace('.vec', '.sca')
    index_file_name = file_name.replace('.vec', '.vci')

    # Create a shorter output filename using hash
    full_name = file_name.replace('.vec', '')
    name_hash = hashlib.md5(full_name.encode()).hexdigest()[:8]
    
    # Create meaningful filename for RL policy
    short_name = f"rl_policy_{name_hash}_{num_spines}s_{num_aggs}a_{num_servers_under_each_rack}srv"
    output_file_name = f'{short_name}_{category}.csv'

    print(f"Output name: {short_name}")
    f.write(f'echo "Processing: {short_name}"\n')

    # ===== FLOW AND APPLICATION METRICS =====
    
    # Flow completion metrics
    output_dir_name = 'FLOW_ENDED/'
    command = "scavetool x --type v --filter \"module(**.server[*].app[*]) AND " \
              "\\\"flowEndedRequesterID:vector\\\"\" -o {} -F CSV-S {}\n".format(
        OUTPUT_FILE_DIRECTORY + output_dir_name + output_file_name, vector_file_name)
    f.write(f'echo "  {output_dir_name}"\n')
    f.write(command)

    output_dir_name = 'FLOW_STARTED/'
    command = "scavetool x --type v --filter \"module(**.server[*].app[*]) AND " \
               "\\\"flowStartedRequesterID:vector\\\"\" -o {} -F CSV-S {}\n".format(
        OUTPUT_FILE_DIRECTORY + output_dir_name + output_file_name, vector_file_name)
    f.write(f'echo "  {output_dir_name}"\n')
    f.write(command)

    output_dir_name = 'REQUEST_SENT/'
    command = "scavetool x --type v --filter \"module(**.server[*].app[*]) AND " \
              "\\\"requestSentRequesterID:vector\\\"\" -o {} -F CSV-S {}\n".format(
        OUTPUT_FILE_DIRECTORY + output_dir_name + output_file_name, vector_file_name)
    f.write(f'echo "  {output_dir_name}"\n')
    f.write(command)

    output_dir_name = 'REPLY_LENGTH_ASKED/'
    command = "scavetool x --type v --filter \"module(**.server[*].app[*]) AND " \
              "\\\"replyLengthAsked:vector\\\"\" -o {} -F CSV-S {}\n".format(
        OUTPUT_FILE_DIRECTORY + output_dir_name + output_file_name, vector_file_name)
    f.write(f'echo "  {output_dir_name}"\n')
    f.write(command)

    # ===== NETWORK SWITCH METRICS (KEY FOR RL ANALYSIS) =====
    
    # Requester ID per packet at switches  
    output_dir_name = 'REQUESTER_ID/'
    command = "scavetool x --type v --filter \"module(**.**.relayUnit) AND " \
              "\\\"RequesterID:vector\\\"\" -o {} -F CSV-S {}\n".format(
        OUTPUT_FILE_DIRECTORY + output_dir_name + output_file_name, vector_file_name)
    f.write(f'echo "  {output_dir_name}"\n')
    f.write(command)
    
    # Queue length at switches (important for RL state)
    output_dir_name = 'QUEUE_LEN/'
    command = "scavetool x --type v --filter \"module(**.**.eth[*].mac.queue) AND " \
              "\\\"queueLength:vector\\\"\" -o {} -F CSV-S {}\n".format(
        OUTPUT_FILE_DIRECTORY + output_dir_name + output_file_name, vector_file_name)
    f.write(f'echo "  {output_dir_name}"\n')
    f.write(command)
    
    # Packet actions/decisions at switches (deflection vs forward)
    output_dir_name = 'PACKET_ACTION/'
    command = "scavetool x --type v --filter \"module(**.**.relayUnit) AND " \
              "\\\"packetAction:vector\\\"\" -o {} -F CSV-S {}\n".format(
        OUTPUT_FILE_DIRECTORY + output_dir_name + output_file_name, vector_file_name)
    f.write(f'echo "  {output_dir_name}"\n')
    f.write(command)

    # ===== TCP METRICS =====
    
    output_dir_name = 'SYN_SENT/'
    command = "scavetool x --type v --filter \"module(**.server[*].tcp) AND " \
              "\\\"tcpConnectionSYNSent:vector\\\"\" -o {} -F CSV-S {}\n".format(
        OUTPUT_FILE_DIRECTORY + output_dir_name + output_file_name, vector_file_name)
    f.write(f'echo "  {output_dir_name}"\n')
    f.write(command)
    
    output_dir_name = 'SYN_SENT_IS_BURSTY/'
    command = "scavetool x --type v --filter \"module(**.server[*].tcp) AND " \
              "\\\"tcpConnectionSYNSentIsBursty:vector\\\"\" -o {} -F CSV-S {}\n".format(
        OUTPUT_FILE_DIRECTORY + output_dir_name + output_file_name, vector_file_name)
    f.write(f'echo "  {output_dir_name}"\n')
    f.write(command)
    
    output_dir_name = 'FIN_ACK_RCV/'
    command = "scavetool x --type v --filter \"module(**.server[*].tcp) AND " \
              "\\\"tcpConnectionFINRcv:vector\\\"\" -o {} -F CSV-S {}\n".format(
        OUTPUT_FILE_DIRECTORY + output_dir_name + output_file_name, vector_file_name)
    f.write(f'echo "  {output_dir_name}"\n')
    f.write(command)

    # ===== SCALAR METRICS =====
    
    output_dir_name = 'OOO_SEG_NUM/'
    command = "scavetool x --type s --filter \"module(**.server[*].tcp) AND " \
              "\\\"numReceivedOOOSegs\\\"\" -o {} -F CSV-S {}\n".format(
        OUTPUT_FILE_DIRECTORY + output_dir_name + output_file_name, scalar_file_name)
    f.write(f'echo "  {output_dir_name}"\n')
    f.write(command)

    output_dir_name = 'UTILIZATION/'
    command = "scavetool x --type s --filter \"module(**.**.eth[*].mac) AND " \
              "\\\"bits/sec *\\\"\" -o {} -F CSV-S {}\n".format(
        OUTPUT_FILE_DIRECTORY + output_dir_name + output_file_name, scalar_file_name)
    f.write(f'echo "  {output_dir_name}"\n')
    f.write(command)

    output_dir_name = 'PER_PACKET_FABRIC_DELAY_COUNTS/'
    command = "scavetool x --type s --filter \"module(**.**.eth[*].mac) AND " \
              "\\\"perPacketFabricDelayNum\\\"\" -o {} -F CSV-S {}\n".format(
        OUTPUT_FILE_DIRECTORY + output_dir_name + output_file_name, scalar_file_name)
    f.write(f'echo "  {output_dir_name}"\n')
    f.write(command)

    output_dir_name = 'PER_PACKET_FABRIC_DELAY_SUM/'
    command = "scavetool x --type s --filter \"module(**.**.eth[*].mac) AND " \
              "\\\"perPacketFabricDelaySum\\\"\" -o {} -F CSV-S {}\n".format(
        OUTPUT_FILE_DIRECTORY + output_dir_name + output_file_name, scalar_file_name)
    f.write(f'echo "  {output_dir_name}"\n')
    f.write(command)

    # ===== V2 ORDERING METRICS =====
    
    output_dir_name = 'V2_RCVD_SOONER/'
    command = "scavetool x --type s --filter \"module(**.ipv4.ip) AND " \
              "\\\"v2RcvdSoonerStored\\\"\" -o {} -F CSV-S {}\n".format(
        OUTPUT_FILE_DIRECTORY + output_dir_name + output_file_name, scalar_file_name)
    f.write(f'echo "  {output_dir_name}"\n')
    f.write(command)

    output_dir_name = 'V2_RCVD_CORRECTLY/'
    command = "scavetool x --type s --filter \"module(**.ipv4.ip) AND " \
              "\\\"v2RcvdCorrectlyPushed\\\"\" -o {} -F CSV-S {}\n".format(
        OUTPUT_FILE_DIRECTORY + output_dir_name + output_file_name, scalar_file_name)
    f.write(f'echo "  {output_dir_name}"\n')
    f.write(command)

    output_dir_name = 'V2_RCVD_LATER/'
    command = "scavetool x --type s --filter \"module(**.ipv4.ip) AND " \
              "\\\"v2RcvdLaterPushed\\\"\" -o {} -F CSV-S {}\n".format(
        OUTPUT_FILE_DIRECTORY + output_dir_name + output_file_name, scalar_file_name)
    f.write(f'echo "  {output_dir_name}"\n')
    f.write(command)

    output_dir_name = 'NUM_MARKING_TIMEOUTS/'
    command = "scavetool x --type s --filter \"module(**.ipv4.ip) AND " \
              "\\\"numTimeoutsMarking\\\"\" -o {} -F CSV-S {}\n".format(
        OUTPUT_FILE_DIRECTORY + output_dir_name + output_file_name, scalar_file_name)
    f.write(f'echo "  {output_dir_name}"\n')
    f.write(command)

    f.write('\n')

f.write('echo "RL policy extraction completed!"\n')
f.close()

print(f"\nExtractor script created: ../../../extractor_rl_policy.sh")
print(f"Total files to process: {len(onlyfiles)}")
print(f"Output directories prepared in: {OUTPUT_FILE_DIRECTORY}")
print("\nRun the extractor with:")
print("  bash ../../../extractor_rl_policy.sh")
