#!/usr/bin/env python3

import sys
import hashlib
from os import listdir
from os.path import isfile, join

UNDER_SAME_RACK = 1
LEAF_SPINE = 2
TOPOLOGY = LEAF_SPINE
REP_NUM = 1
SIMS_DIR = '/home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims'
OUTPUT_FILE_DIRECTORY = f'{SIMS_DIR}/extracted_results/'

if len(sys.argv) > 1:
    add_category = sys.argv[1]
    # Set the results directory based on the category
    RESULT_FILES_DIR = f'./results/{add_category}/'
else:
    add_category = input('What to add to category?')
    final_check = input('Your output directory is:{}\n and your category is: {}?'.format(OUTPUT_FILE_DIRECTORY, add_category))
    if final_check != 'yes':
        raise Exception('You did not accepted!')
    RESULT_FILES_DIR = f'./results/{add_category}/'
category = add_category

# Look for .vec files with _rep_ pattern in current directory
onlyfiles = [f for f in listdir(RESULT_FILES_DIR) if
             isfile(join(RESULT_FILES_DIR, f)) and f[-4:] == '.vec' and '_rep_' in f]

print(f"Found {len(onlyfiles)} .vec files")

# Write extractor script to correct location
f = open('extractor.sh', 'w')
f.write('#!/bin/bash\n')
f.write('# Auto-generated extractor script\n\n')

for file_name in onlyfiles:
    num_spines = file_name.split('_spines')[0]
    num_aggs = (file_name.split('_aggs')[0]).split('_')[-1]
    num_servers_under_each_rack = (file_name.split('_servers_')[0]).split('_')[-1]
    # num_bursty_servers = (file_name.split('_numBurstyServers')[0]).split('_')[-1]
    num_bursty_apps = (file_name.split('_burstyapps')[0]).split('_')[-1]
    num_req_per_burst = (file_name.split('_reqPerBurst')[0]).split('_')[-1]
    num_mice_flows_per_server = (file_name.split('_mice')[0]).split('_')[-1]
    bg_inter_arrival_multiplier = (file_name.split('_bgintermult')[0]).split('_')[-1]
    bg_flow_size_multiplier = (file_name.split('_bgfsizemult')[0]).split('_')[-1]
    bursty_inter_arrival_multiplier = (file_name.split('_burstyintermult')[0]).split('_')[-1]
    bursty_flow_size_multiplier = (file_name.split('_burstyfsizemult')[0]).split('_')[-1]
    ttl = (file_name.split('_ttl_')[0]).split('_')[-1]
    # gro_delay = (file_name.split('_groDelayInitialValue_')[0]).split('_')[-1]
    # micro_burst_queue_thresh = (file_name.split('_mburstqthresh_')[0]).split('_')[-1]
    random_power_factor = (file_name.split('_rndfwfactor')[0]).split('_')[-1]
    random_power_bounce_factor = (file_name.split('_rndbouncefactor')[0]).split('_')[-1]
    incast_flow_size = (file_name.split('_incastfsize')[0]).split('_')[-1]
    marking_timer = (file_name.split('_mrktimer')[0]).split('_')[-1]
    ordering_timer = (file_name.split('_ordtimer')[0]).split('_')[-1]


    for rep_num in range(REP_NUM):
        # Use the actual filename with proper path
        vector_file_name = f"{add_category}/{file_name}"
        scalar_file_name = f"{add_category}/{file_name.replace('.vec', '.sca')}"
        index_file_name = f"{add_category}/{file_name.replace('.vec', '.vci')}"

        # Create a shorter output filename to avoid filesystem limits
        import hashlib
        
        # Create a hash from the full filename for uniqueness
        full_name = file_name.replace('.vec', '')
        name_hash = hashlib.md5(full_name.encode()).hexdigest()[:8]
        
        # Extract key parameters for a readable filename
        threshold_part = ''
        if 'threshold_' in full_name:
            threshold_part = '_' + full_name.split('threshold_')[-1]

        # Extract beta, theta, eps if present
        if 'beta_' in full_name:
            beta_value = full_name.split('beta_')[1].split('_')[0]
            threshold_part += f'_beta{beta_value}'
        if 'theta_' in full_name:
            theta_value = full_name.split('theta_')[1].split('_')[0]
            threshold_part += f'_theta{theta_value}'
        if 'eps_' in full_name:
            eps_value = full_name.split('eps_')[1].split('_')[0]
            threshold_part += f'_eps{eps_value}'
        
        # Create shorter but meaningful filename
        short_name = f"sim_{name_hash}{threshold_part}"
        output_file_name = f'{short_name}_{category}.csv'

        print(short_name)
        f.write('echo \"{}\"\n'.format(short_name))

        '''
        # Flow- and requester-related extractions (enabled for RL dataset enrichment)

        output_dir_name = 'REQUEST_SENT/'
        command = "scavetool x --type v --filter \"module(**.server[*].app[*]) AND " \
                  "\\\"requestSentRequesterID:vector\\\"\" -o {} -F CSV-S {}\n".format(OUTPUT_FILE_DIRECTORY+output_dir_name+output_file_name, vector_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)

        output_dir_name = 'REPLY_LENGTH_ASKED/'
        command = "scavetool x --type v --filter \"module(**.server[*].app[*]) AND " \
                  "\\\"replyLengthAsked:vector\\\"\" -o {} -F CSV-S {}\n".format(OUTPUT_FILE_DIRECTORY+output_dir_name+output_file_name, vector_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)

        # Requester ID per packet at switches (note: signal is 'requesterID' in code)
        output_dir_name = 'SYN_SENT/'
        command = "scavetool x --type v --filter \"module(**.server[*].tcp) AND " \
                  "\\\"tcpConnectionSYNSent:vector\\\"\" -o {} -F CSV-S {}\n".format(OUTPUT_FILE_DIRECTORY+output_dir_name+output_file_name, vector_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)
        
        output_dir_name = 'SYN_SENT_IS_BURSTY/'
        command = "scavetool x --type v --filter \"module(**.server[*].tcp) AND " \
                  "\\\"tcpConnectionSYNSentIsBursty:vector\\\"\" -o {} -F CSV-S {}\n".format(OUTPUT_FILE_DIRECTORY+output_dir_name+output_file_name, vector_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)
        
        output_dir_name = 'FIN_ACK_RCV/'
        command = "scavetool x --type v --filter \"module(**.server[*].tcp) AND " \
                  "\\\"tcpConnectionFINRcv:vector\\\"\" -o {} -F CSV-S {}\n".format(OUTPUT_FILE_DIRECTORY+output_dir_name+output_file_name, vector_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)
        
        output_dir_name = 'OOO_SEG_NUM/'
        command = "scavetool x --type s --filter \"module(**.server[*].tcp) AND " \
                  "\\\"numReceivedOOOSegs\\\"\" -o {} -F CSV-S {}\n".format(OUTPUT_FILE_DIRECTORY+output_dir_name+output_file_name, scalar_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)

        output_dir_name = 'UTILIZATION/'
        command = "scavetool x --type s --filter \"module(**.**.eth[*].mac) AND " \
                  "\\\"bits/sec *\\\"\" -o {} -F CSV-S {}\n".format(OUTPUT_FILE_DIRECTORY+output_dir_name+output_file_name, scalar_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)

        output_dir_name = 'PER_PACKET_FABRIC_DELAY_COUNTS/'
        command = "scavetool x --type s --filter \"module(**.**.eth[*].mac) AND " \
                  "\\\"perPacketFabricDelayNum\\\"\" -o {} -F CSV-S {}\n".format(OUTPUT_FILE_DIRECTORY+output_dir_name+output_file_name, scalar_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)

        output_dir_name = 'PER_PACKET_FABRIC_DELAY_SUM/'
        command = "scavetool x --type s --filter \"module(**.**.eth[*].mac) AND " \
                  "\\\"perPacketFabricDelaySum\\\"\" -o {} -F CSV-S {}\n".format(OUTPUT_FILE_DIRECTORY+output_dir_name+output_file_name, scalar_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)

        output_dir_name = 'V2_RCVD_SOONER/'
        command = "scavetool x --type s --filter \"module(**.ipv4.ip) AND " \
                  "\\\"v2RcvdSoonerStored\\\"\" -o {} -F CSV-S {}\n".format(OUTPUT_FILE_DIRECTORY+output_dir_name+output_file_name, scalar_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)

        output_dir_name = 'V2_RCVD_CORRECTLY/'
        command = "scavetool x --type s --filter \"module(**.ipv4.ip) AND " \
                  "\\\"v2RcvdCorrectlyPushed\\\"\" -o {} -F CSV-S {}\n".format(OUTPUT_FILE_DIRECTORY+output_dir_name+output_file_name, scalar_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)

        output_dir_name = 'V2_RCVD_LATER/'
        command = "scavetool x --type s --filter \"module(**.ipv4.ip) AND " \
                  "\\\"v2RcvdLaterPushed\\\"\" -o {} -F CSV-S {}\n".format(OUTPUT_FILE_DIRECTORY+output_dir_name+output_file_name, scalar_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)

        output_dir_name = 'NUM_MARKING_TIMEOUTS/'
        command = "scavetool x --type s --filter \"module(**.ipv4.ip) AND " \
                  "\\\"numTimeoutsMarking\\\"\" -o {} -F CSV-S {}\n".format(OUTPUT_FILE_DIRECTORY+output_dir_name+output_file_name, scalar_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)

        output_dir_name = 'NUM_ORDERING_TIMEOUTS/'
        command = "scavetool x --type s --filter \"module(**.ipv4.ip) AND " \
                  "\\\"numTimeoutsOrdering\\\"\" -o {} -F CSV-S {}\n".format(OUTPUT_FILE_DIRECTORY+output_dir_name+output_file_name, scalar_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)

        output_dir_name = 'IP_PACKET_SENT_COUNTER/'
        command = "scavetool x --type s --filter \"module(**.ipv4.ip) AND " \
                  "\\\"IPPacketSentCounter\\\"\" -o {} -F CSV-S {}\n".format(OUTPUT_FILE_DIRECTORY+output_dir_name+output_file_name, scalar_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)

        output_dir_name = 'IP_DATA_PACKET_SENT_COUNTER/'
        command = "scavetool x --type s --filter \"module(**.ipv4.ip) AND " \
                  "\\\"IPDataPacketSentCounter\\\"\" -o {} -F CSV-S {}\n".format(OUTPUT_FILE_DIRECTORY+output_dir_name+output_file_name, scalar_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)

        output_dir_name = 'V2_QUEUEING_TIME/'
        command = "scavetool x --type s --filter \"module(**.ipv4.ip) AND " \
                  "\\\"v2QTime:*\\\"\" -o {} -F CSV-S {}\n".format(OUTPUT_FILE_DIRECTORY+output_dir_name+output_file_name, scalar_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)

        output_dir_name = 'NUM_RTO/'
        command = "scavetool x --type s --filter \"module(**.server[*].tcp) AND " \
                  "\\\"numRTOs\\\"\" -o {} -F CSV-S {}\n".format(OUTPUT_FILE_DIRECTORY+output_dir_name+output_file_name, scalar_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)

        output_dir_name = 'DUP_ACKS/'
        command = "scavetool x --type s --filter \"module(**.server[*].tcp) AND " \
                  "\\\"numDupAcks\\\"\" -o {} -F CSV-S {}\n".format(OUTPUT_FILE_DIRECTORY+output_dir_name+output_file_name, scalar_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)

        output_dir_name = 'DUP_ACKS_AFTER_FR/'
        command = "scavetool x --type s --filter \"module(**.server[*].tcp) AND " \
                  "\\\"numDupAcksAfterFastRetransmit\\\"\" -o {} -F CSV-S {}\n".format(OUTPUT_FILE_DIRECTORY+output_dir_name+output_file_name, scalar_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)

        output_dir_name = 'DUP_ACKS_FAST_RECOVERY/'
        command = "scavetool x --type s --filter \"module(**.server[*].tcp) AND " \
                  "\\\"numFastFastRecoveries\\\"\" -o {} -F CSV-S {}\n".format(OUTPUT_FILE_DIRECTORY+output_dir_name+output_file_name, scalar_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)

        output_dir_name = 'DUP_ACKS_FAST_RETRANSMISSION/'
        command = "scavetool x --type s --filter \"module(**.server[*].tcp) AND " \
                  "\\\"numFastRetransmits\\\"\" -o {} -F CSV-S {}\n".format(OUTPUT_FILE_DIRECTORY+output_dir_name+output_file_name, scalar_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)

        output_dir_name = 'NUM_PACKETS_MARKED_DCTCP/'
        command = "scavetool x --type s --filter \"module(**.server[*].tcp) AND " \
                  "\\\"DCTCPNumPacketsMarked\\\"\" -o {} -F CSV-S {}\n".format(OUTPUT_FILE_DIRECTORY+output_dir_name+output_file_name, scalar_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)

        output_dir_name = 'NUM_TCP_CONNECTIONS/'
        command = "scavetool x --type s --filter \"module(**.server[*].tcp) AND " \
                  "\\\"numConnections\\\"\" -o {} -F CSV-S {}\n".format(OUTPUT_FILE_DIRECTORY+output_dir_name+output_file_name, scalar_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)

        output_dir_name = 'NUM_SEGMENT_RETRANSMISSION/'
        command = "scavetool x --type s --filter \"module(**.server[*].tcp) AND " \
                  "\\\"numSegmentRetransmissions\\\"\" -o {} -F CSV-S {}\n".format(OUTPUT_FILE_DIRECTORY+output_dir_name+output_file_name, scalar_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)

        # output_dir_name = 'BYTES_REQUESTED_FROM_SERVER/'
        # command = "scavetool x --type v --filter \"module(**.server[*].app[*]) AND " \
        #           "\\\"bytesRequestedFromServer:vector\\\"\" -o {} -F CSV-S {}\n".format(OUTPUT_FILE_DIRECTORY+output_dir_name+output_file_name, vector_file_name)
        # print(command)
        # f.write('echo \"{}\"\n'.format(output_dir_name))
        # f.write(command)

        output_dir_name = 'PACKET_DROPPED_FORWARDING_DISABLED/'
        command = "scavetool x --type v --filter \"module(**.server[*].ipv4.ip) AND " \
                  "\\\"packetDropForwardingDisabled:vector\\\"\" -o {} -F CSV-S {}\n".format(OUTPUT_FILE_DIRECTORY+output_dir_name+output_file_name, vector_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)
        
        # output_dir_name = 'DROP_SENDER_NAMES/'
        # command = "scavetool x --type v --filter \"module(**.**.eth[*].mac.queue*) AND " \
        #           "\\\"packetDroppedSenderName:vector\\\"\" -o {} -F CSV-S {}\n".format(OUTPUT_FILE_DIRECTORY+output_dir_name+output_file_name, vector_file_name)
        # print(command)
        # f.write('echo \"{}\"\n'.format(output_dir_name))
        # f.write(command)
        #
        # output_dir_name = 'DROP_RECEIVER_NAMES/'
        # command = "scavetool x --type v --filter \"module(**.**.eth[*].mac.queue*) AND " \
        #           "\\\"packetDroppedReceiverName:vector\\\"\" -o {} -F CSV-S {}\n".format(OUTPUT_FILE_DIRECTORY+output_dir_name+output_file_name, vector_file_name)
        # print(command)
        # f.write('echo \"{}\"\n'.format(output_dir_name))
        # f.write(command)
        
        # output_dir_name = 'DROPS/'
        # command = "scavetool x --type v --filter \"module(**.**.eth[*].mac.queue*) AND " \
        #           "\\\"packetDropQueueOverflow:vector(packetBytes)\\\"\" -o {} -F CSV-S {}\n".format(OUTPUT_FILE_DIRECTORY+output_dir_name+output_file_name, vector_file_name)
        # print(command)
        # f.write('echo \"{}\"\n'.format(output_dir_name))
        # f.write(command)
        
        output_dir_name = 'LIGHT_DROPS/'
        command = "scavetool x --type s --filter \"module(**.**.eth[*].mac.queue*) AND " \
                  "\\\"packetDropQueueOverflow:count\\\"\" -o {} -F CSV-S {}\n".format(OUTPUT_FILE_DIRECTORY+output_dir_name+output_file_name, scalar_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)
        
        # output_dir_name = 'DROPS_IS_BURSTY/'
        # command = "scavetool x --type v --filter \"module(**.**.eth[*].mac.queue*) AND " \
        #           "\\\"packetDroppedIsBursty:vector\\\"\" -o {} -F CSV-S {}\n".format(OUTPUT_FILE_DIRECTORY+output_dir_name+output_file_name, vector_file_name)
        # print(command)
        # f.write('echo \"{}\"\n'.format(output_dir_name))
        # f.write(command)
        
        # output_dir_name = 'DROPS_SEQS/'
        # command = "scavetool x --type v --filter \"module(**.**.eth[*].mac.queue*) AND " \
        #           "\\\"droppktSeqs:vector\\\"\" -o {} -F CSV-S {}\n".format(OUTPUT_FILE_DIRECTORY+output_dir_name+output_file_name, vector_file_name)
        # print(command)
        # f.write('echo \"{}\"\n'.format(output_dir_name))
        # f.write(command)
        
        # output_dir_name = 'DROPS_RET_COUNTS/'
        # command = "scavetool x --type v --filter \"module(**.**.eth[*].mac.queue*) AND " \
        #           "\\\"droppktRetCount:vector\\\"\" -o {} -F CSV-S {}\n".format(OUTPUT_FILE_DIRECTORY+output_dir_name+output_file_name, vector_file_name)
        # print(command)
        # f.write('echo \"{}\"\n'.format(output_dir_name))
        # f.write(command)
        
        # output_dir_name = 'DROPS_RET_TOTAL_PAYLOAD_LEN/'
        # command = "scavetool x --type v --filter \"module(**.**.eth[*].mac.queue*) AND " \
        #           "\\\"pktDropTotalPayloadLen:vector\\\"\" -o {} -F CSV-S {}\n".format(OUTPUT_FILE_DIRECTORY+output_dir_name+output_file_name, vector_file_name)
        # print(command)
        # f.write('echo \"{}\"\n'.format(output_dir_name))
        # f.write(command)
        
        output_dir_name = 'LIGHT_IN_QUEUE_DROP_COUNTER/'
        command = "scavetool x --type s --filter \"module(**.**.eth[*].mac.queue*) AND " \
                  "\\\"lightInQueuePacketDropCount\\\"\" -o {} -F CSV-S {}\n".format(OUTPUT_FILE_DIRECTORY+output_dir_name+output_file_name, scalar_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)
        
        output_dir_name = 'LIGHT_ALL_QUEUEING_TIME/'
        command = "scavetool x --type s --filter \"module(**.**.eth[*].mac.queue*) AND " \
                  "\\\"lightAllQueueingTime\\\"\" -o {} -F CSV-S {}\n".format(OUTPUT_FILE_DIRECTORY+output_dir_name+output_file_name, scalar_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)
        
        output_dir_name = 'LIGHT_MICE_QUEUEING_TIME/'
        command = "scavetool x --type s --filter \"module(**.**.eth[*].mac.queue*) AND " \
                  "\\\"lightMiceQueueingTime\\\"\" -o {} -F CSV-S {}\n".format(OUTPUT_FILE_DIRECTORY+output_dir_name+output_file_name, scalar_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)
        
        output_dir_name = 'QUEUEING_TIME/'
        command = "scavetool x --type s --filter \"module(**.**.eth[*].mac.queue*) AND " \
                  "\\\"queueingTime:*\\\"\" -o {} -F CSV-S {}\n".format(OUTPUT_FILE_DIRECTORY+output_dir_name+output_file_name, scalar_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)
        
        # output_dir_name = 'QUEUEING_TIME_FLOW_SIZE/'
        # command = "scavetool x --type v --filter \"module(**.**.eth[*].mac.queue*) AND " \
        #           "\\\"queueingTimeFlowSize:vector\\\"\" -o {} -F CSV-S {}\n".format(OUTPUT_FILE_DIRECTORY+output_dir_name+output_file_name, vector_file_name)
        # print(command)
        # f.write('echo \"{}\"\n'.format(output_dir_name))
        # f.write(command)
        
        output_dir_name = 'QUEUEING_TIME_INCAST/'
        command = "scavetool x --type s --filter \"module(**.**.eth[*].mac.queue*) AND " \
                  "\\\"queueingTimeIncast:*\\\"\" -o {} -F CSV-S {}\n".format(OUTPUT_FILE_DIRECTORY+output_dir_name+output_file_name, scalar_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)
        
        output_dir_name = 'QUEUE_LENGTH/'
        command = "scavetool x --type s --filter \"module(**.**.eth[*].mac.queue*) AND " \
                  "\\\"DCQueueLength:*\\\"\" -o {} -F CSV-S {}\n".format(OUTPUT_FILE_DIRECTORY+output_dir_name+output_file_name, scalar_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)

        output_dir_name = 'QUEUE_LENGTH_PKT_BYTES/'
        command = "scavetool x --type s --filter \"module(**.**.eth[*].mac.queue*) AND " \
                  "\\\"DCQueueLengthPacketBytes:*\\\"\" -o {} -F CSV-S {}\n".format(
            OUTPUT_FILE_DIRECTORY + output_dir_name + output_file_name, scalar_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)

        # output_dir_name = 'Extra/'
        # command = "scavetool x --type v --filter \"module(**.**.eth[*].mac.queue*) AND " \
        #           "\\\"DCQueueLengthPacketBytes:*\\\"\" -o {} -F CSV-S {}\n".format(
        #     OUTPUT_FILE_DIRECTORY + output_dir_name + output_file_name, vector_file_name)
        # print(command)
        # f.write('echo \"{}\"\n'.format(output_dir_name))
        # f.write(command)
        
        # output_dir_name = 'FB_PACKET_GENERATED/'
        # command = "scavetool x --type v --filter \"module(**.**.eth[*].mac) AND " \
        #           "\\\"FBPacketGenerated:vector\\\"\" -o {} -F CSV-S {}\n".format(OUTPUT_FILE_DIRECTORY+output_dir_name+output_file_name, vector_file_name)
        # print(command)
        # f.write('echo \"{}\"\n'.format(output_dir_name))
        # f.write(command)
        
        # output_dir_name = 'FB_PACKET_GENERATED_Req_ID/'
        # command = "scavetool x --type v --filter \"module(**.**.eth[*].mac) AND " \
        #           "\\\"FBPacketGeneratedReqID:vector\\\"\" -o {} -F CSV-S {}\n".format(OUTPUT_FILE_DIRECTORY+output_dir_name+output_file_name, vector_file_name)
        # print(command)
        # f.write('echo \"{}\"\n'.format(output_dir_name))
        # f.write(command)
        
        #output_dir_name = 'FB_PACKET_DROPPED/'
        #command = "scavetool x --type v --filter \"module(**.**.eth[*].mac) AND " \
        #           "\\\"FBPacketDropped:vector\\\"\" -o {} -F CSV-S {}\n".format(OUTPUT_FILE_DIRECTORY+output_dir_name+output_file_name, vector_file_name)
        #print(command)
        #f.write('echo \"{}\"\n'.format(output_dir_name))
        #f.write(command)
        
        # output_dir_name = 'FB_BOUNCING_PASSED/'
        # command = "scavetool x --type v --filter \"module(**.**.eth[*].mac) AND " \
        #           "\\\"FBBouncePassed:vector\\\"\" -o {} -F CSV-S {}\n".format(OUTPUT_FILE_DIRECTORY+output_dir_name+output_file_name, vector_file_name)
        # print(command)
        # f.write('echo \"{}\"\n'.format(output_dir_name))
        # f.write(command)

        output_dir_name = 'HOP_COUNT/'
        command = "scavetool x --type s --filter \"module(**.**.eth[*].mac) AND " \
                  "\\\"packetHopCount:*\\\"\" -o {} -F CSV-S {}\n".format(OUTPUT_FILE_DIRECTORY+output_dir_name+output_file_name, scalar_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)
        
        output_dir_name = 'FB_PACKET_GENERATED/'
        command = "scavetool x --type s --filter \"module(**.**.relayUnit) AND " \
                  "\\\"FBPacketGenerated:count\\\"\" -o {} -F CSV-S {}\n".format(OUTPUT_FILE_DIRECTORY+output_dir_name+output_file_name, scalar_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)
        
        #output_dir_name = 'FB_PACKET_DROPPED/'
        #command = "scavetool x --type v --filter \"module(**.**.relayUnit) AND " \
        #           "\\\"FBPacketDropped:vector\\\"\" -o {} -F CSV-S {}\n".format(OUTPUT_FILE_DIRECTORY+output_dir_name+output_file_name, vector_file_name)
        #print(command)
        #f.write('echo \"{}\"\n'.format(output_dir_name))
        #f.write(command)
        
        # output_dir_name = 'FB_PACKET_DROPPED_PORT/'
        # command = "scavetool x --type v --filter \"module(**.**.relayUnit) AND " \
        #           "\\\"FBPacketDroppedPort:vector\\\"\" -o {} -F CSV-S {}\n".format(OUTPUT_FILE_DIRECTORY+output_dir_name+output_file_name, vector_file_name)
        # print(command)
        # f.write('echo \"{}\"\n'.format(output_dir_name))
        # f.write(command)
        
        # output_dir_name = 'FB_BOUNCING_PASSED/'
        # command = "scavetool x --type s --filter \"module(**.**.relayUnit) AND " \
        #           "\\\"FBBouncePassed:count\\\"\" -o {} -F CSV-S {}\n".format(OUTPUT_FILE_DIRECTORY+output_dir_name+output_file_name, scalar_file_name)
        # print(command)
        # f.write('echo \"{}\"\n'.format(output_dir_name))
        # f.write(command)
        
        # output_dir_name = 'BURSTY_PKT_RCVED/'
        # command = "scavetool x --type v --filter \"module(**.**.relayUnit) AND " \
        #           "\\\"BurstyPacketReceived:vector\\\"\" -o {} -F CSV-S {}\n".format(OUTPUT_FILE_DIRECTORY+output_dir_name+output_file_name, vector_file_name)
        # print(command)
        # f.write('echo \"{}\"\n'.format(output_dir_name))
        # f.write(command)
        
        output_dir_name = 'LIGHT_IN_RELAY_DROP_COUNTER/'
        command = "scavetool x --type s --filter \"module(**.**.relayUnit) AND " \
                  "\\\"lightInRelayPacketDropCounter\\\"\" -o {} -F CSV-S {}\n".format(OUTPUT_FILE_DIRECTORY+output_dir_name+output_file_name, scalar_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)

        output_dir_name = 'SENT_PKTS_SEQ_NUMS/'
        command = "scavetool x --type v --filter \"module(**.server[*].tcp.*) AND " \
                  "\\\"sndNxt:vector\\\"\" -o {} -F CSV-S {}\n".format(OUTPUT_FILE_DIRECTORY + output_dir_name + output_file_name, vector_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)

        # output_dir_name = 'RTT/'
        # command = "scavetool x --type v --filter \"module(**.server[*].tcp.*) AND " \
        #           "\\\"rtt:vector\\\"\" -o {} -F CSV-S {}\n".format(OUTPUT_FILE_DIRECTORY + output_dir_name + output_file_name, vector_file_name)
        # print(command)
        # f.write('echo \"{}\"\n'.format(output_dir_name))
        # f.write(command)

        output_dir_name = 'LIGHT_RTT_SUMS/'
        command = "scavetool x --type s --filter \"module(**.server[*].tcp) AND " \
                  "\\\"RTTSum\\\"\" -o {} -F CSV-S {}\n".format(OUTPUT_FILE_DIRECTORY+output_dir_name+output_file_name, scalar_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)

        output_dir_name = 'LIGHT_RTT_RECORD_COUNTS/'
        command = "scavetool x --type s --filter \"module(**.server[*].tcp) AND " \
                  "\\\"numRTTRecords\\\"\" -o {} -F CSV-S {}\n".format(OUTPUT_FILE_DIRECTORY+output_dir_name+output_file_name, scalar_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)

        output_dir_name = 'LIGHT_GOODPUT_NUM_BITS_SENT_TO_APP/'
        command = "scavetool x --type s --filter \"module(**.server[*].tcp) AND " \
                  "\\\"lightGoodputNumBitsSentToApp\\\"\" -o {} -F CSV-S {}\n".format(OUTPUT_FILE_DIRECTORY+output_dir_name+output_file_name, scalar_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)

        output_dir_name = 'LIGHT_THROUGHPUT_BITS_SENT/'
        command = "scavetool x --type s --filter \"module(**.**.eth[*].mac) AND " \
                  "\\\"lightThroughputBitsSent\\\"\" -o {} -F CSV-S {}\n".format(OUTPUT_FILE_DIRECTORY+output_dir_name+output_file_name, scalar_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)

        output_dir_name = 'LIGHT_THROUGHPUT_BITS_RCVD/'
        command = "scavetool x --type s --filter \"module(**.**.eth[*].mac) AND " \
                  "\\\"lightThroughputBitsRcvd\\\"\" -o {} -F CSV-S {}\n".format(OUTPUT_FILE_DIRECTORY+output_dir_name+output_file_name, scalar_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)
        
    '''
        output_dir_name = 'QUEUE_LEN/'
        command = (
                "scavetool x --type v --filter "
                "\"(module(**.agg[*].relayUnit) OR module(**.spine[*].relayUnit)) AND name(QueueLen:vector)\" "
                "-o {} -F CSV-R {}\n"
            ).format(OUTPUT_FILE_DIRECTORY + output_dir_name + output_file_name, vector_file_name)

        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)
        
        command = '\n\n'# output_dir_name = 'RTT/'
        # command = "scavetool x --type v --filter \"module(**.server[*].tcp.*) AND " \
        #           "\\\"rtt:vector\\\"\" -o {} -F CSV-S {}\n".format(OUTPUT_FILE_DIRECTORY + output_dir_name + output_file_name, vector_file_name)
        # print(command)
        # f.write('echo \"{}\"\n'.format(output_dir_name))
        # f.write(comman
        print(command)
        f.write(command)
        
        output_dir_name = 'QUEUES_TOT_LEN/'
        command = (
            "scavetool x --type v --filter "
            "\"(module(**.agg[*].relayUnit) OR module(**.spine[*].relayUnit)) AND name(QueuesTotLen:vector)\" "
            "-o {} -F CSV-R {}\n"
        ).format(OUTPUT_FILE_DIRECTORY + output_dir_name + output_file_name, vector_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)
        
        command = '\n\n'
        print(command)
        f.write(command)
        
        output_dir_name = 'QUEUE_CAPACITY/'
        command = (
            "scavetool x --type v --filter "
            "\"(module(**.agg[*].relayUnit) OR module(**.spine[*].relayUnit)) AND name(QueueCapacity:vector)\" "
            "-o {} -F CSV-R {}\n"
        ).format(OUTPUT_FILE_DIRECTORY + output_dir_name + output_file_name, vector_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)
        
        command = '\n\n'
        print(command)
        f.write(command)
        
        output_dir_name = 'QUEUES_TOT_CAPACITY/'
        command = (
            "scavetool x --type v --filter "
            "\"(module(**.agg[*].relayUnit) OR module(**.spine[*].relayUnit)) AND name(QueuesTotCapacity:vector)\" "
            "-o {} -F CSV-R {}\n"
        ).format(OUTPUT_FILE_DIRECTORY + output_dir_name + output_file_name, vector_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)
        
        command = '\n\n'
        print(command)
        f.write(command)
        output_dir_name = 'PACKET_ACTION/'
        command = (
            "scavetool x --type v --filter "
            "\"(module(**.agg[*].relayUnit) OR module(**.spine[*].relayUnit)) AND name(PacketAction:vector)\" "
            "-o {} -F CSV-R {}\n"
        ).format(OUTPUT_FILE_DIRECTORY + output_dir_name + output_file_name, vector_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)
        
        command = '\n\n'
        print(command)
        f.write(command)
        
        #output_dir_name = 'RTT/'
        #command = "scavetool x --type v --filter \"module(**.server[*].tcp.*) AND " \
        #          "\\\"rtt:vector\\\"\" -o {} -F CSV-S {}\n".format(OUTPUT_FILE_DIRECTORY + output_dir_name + output_file_name, vector_file_name)
        #print(command)
        #f.write('echo \"{}\"\n'.format(output_dir_name))
        #f.write(command)
        
        #command = '\n\n'
        #print(command)
        #f.write(command)
        
        #output_dir_name = 'CLIENT_SEQ_NUM/'
        #command = "scavetool x --type v --filter \"module(**.server[*].tcp.*) AND " \
        #          "\\\"seqNo:vector\\\"\" -o {} -F CSV-S {}\n".format(OUTPUT_FILE_DIRECTORY + output_dir_name + output_file_name, vector_file_name)
        #print(command)
        #f.write('echo \"{}\"\n'.format(output_dir_name))
        #f.write(command)
        
        #command = '\n\n'
        #print(command)
        #f.write(command)
        
        output_dir_name = 'FLOW_ENDED_QUERY_ID/'
        command = "scavetool x --type v --filter \"module(**.server[*].app[*]) AND " \
                  "\\\"flowEndedQueryID:vector\\\"\" -o {} -F CSV-R {}\n".format(OUTPUT_FILE_DIRECTORY + output_dir_name + output_file_name, vector_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)

        command = '\n\n'
        print(command)
        f.write(command)

        output_dir_name = 'RCV_TS_SEQ_NUM/'
        command = "scavetool x --type v --filter \"module(LeafSpine1G) AND " \
                  "\\\"rcvSeq:vector\\\"\" -o {} -F CSV-R {}\n".format(OUTPUT_FILE_DIRECTORY + output_dir_name + output_file_name, vector_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)
        
        command = '\n\n'
        print(command)
        f.write(command)
        
        output_dir_name = 'SND_TS_SEQ_NUM/'
        command = "scavetool x --type v --filter \"module(LeafSpine1G) AND " \
                  "\\\"sndNxt:vector\\\"\" -o {} -F CSV-R {}\n".format(OUTPUT_FILE_DIRECTORY + output_dir_name + output_file_name, vector_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)
        
        command = '\n\n'
        print(command)
        f.write(command)

        output_dir_name = 'OOO_SEG/'
        command = "scavetool x --type v --filter \"module(LeafSpine1G) AND " \
                  "\\\"rcvOooSeg:vector\\\"\" -o {} -F CSV-R {}\n".format(OUTPUT_FILE_DIRECTORY + output_dir_name + output_file_name, vector_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)
        
        command = '\n\n'
        print(command)
        f.write(command)
        
        output_dir_name = 'SWITCH_SEQ_NUM/'
        command = "scavetool x --type v --filter \"module(LeafSpine1G) AND " \
                  "\\\"switchSeqNum:vector\\\"\" -o {} -F CSV-R {}\n".format(OUTPUT_FILE_DIRECTORY + output_dir_name + output_file_name, vector_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)
        
        command = '\n\n'
        print(command)
        f.write(command)

        output_dir_name = 'RETRANSMITTED/'
        command = "scavetool x --type v --filter \"module(LeafSpine1G) AND " \
                  "\\\"retransmitted:vector\\\"\" -o {} -F CSV-R {}\n".format(OUTPUT_FILE_DIRECTORY + output_dir_name + output_file_name, vector_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)

        command = '\n\n'
        print(command)
        f.write(command)
        
        output_dir_name = 'TTL/'
        command = "scavetool x --type v --filter \"module(LeafSpine1G) AND " \
                  "\\\"switchTtl:vector\\\"\" -o {} -F CSV-R {}\n".format(OUTPUT_FILE_DIRECTORY + output_dir_name + output_file_name, vector_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)
        
        command = '\n\n'
        print(command)
        f.write(command)

        output_dir_name = 'ACTION_SEQ_NUM/'
        command = "scavetool x --type v --filter \"module(LeafSpine1G) AND " \
                  "\\\"actionSeqNum:vector\\\"\" -o {} -F CSV-R {}\n".format(OUTPUT_FILE_DIRECTORY + output_dir_name + output_file_name, vector_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)

        command = '\n\n'
        print(command)
        f.write(command)
        
        
        output_dir_name = 'SWITCH_ID/'
        command = "scavetool x --type v --filter \"module(**.relayUnit) AND " \
                  "\\\"switchId:vector\\\"\" -o {} -F CSV-R {}\n".format(OUTPUT_FILE_DIRECTORY + output_dir_name + output_file_name, vector_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)
        
        command = '\n\n'
        print(command)
        f.write(command)
        
        
        output_dir_name = 'SWITCH_ID_ACTION/'
        command = "scavetool x --type v --filter \"module(**.relayUnit) AND " \
                  "\\\"switchIdAction:vector\\\"\" -o {} -F CSV-R {}\n".format(OUTPUT_FILE_DIRECTORY + output_dir_name + output_file_name, vector_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)

        command = '\n\n'
        print(command)
        f.write(command)
        
        
        output_dir_name = 'INTERFACE_ID/'
        command = "scavetool x --type v --filter \"module(**.relayUnit) AND " \
                  "\\\"interfaceId:vector\\\"\" -o {} -F CSV-R {}\n".format(OUTPUT_FILE_DIRECTORY + output_dir_name + output_file_name, vector_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)

        command = '\n\n'
        print(command)
        f.write(command)

        # New: requester and flow-level metadata for RL datasets
        output_dir_name = 'REQUESTER_ID/'
        # Note: the statistic name in NED is 'RequesterID' (capital R), scavetool matches the statistic name, not the signal
        command = "scavetool x --type v --filter \"module(**.**.relayUnit) AND " \
            "\\\"RequesterID:vector\\\"\" -o {} -F CSV-R {}\n".format(OUTPUT_FILE_DIRECTORY + output_dir_name + output_file_name, vector_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)

        command = '\n\n'
        print(command)
        f.write(command)

        output_dir_name = 'FLOW_STARTED/'
        command = "scavetool x --type v --filter \"module(**.server[*].app[*]) AND " \
            "\\\"flowStartedRequesterID:vector\\\"\" -o {} -F CSV-R {}\n".format(OUTPUT_FILE_DIRECTORY + output_dir_name + output_file_name, vector_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)

        command = '\n\n'
        print(command)
        f.write(command)

        output_dir_name = 'FLOW_ENDED/'
        command = "scavetool x --type v --filter \"module(**.server[*].app[*]) AND " \
            "\\\"flowEndedRequesterID:vector\\\"\" -o {} -F CSV-R {}\n".format(OUTPUT_FILE_DIRECTORY + output_dir_name + output_file_name, vector_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)

        command = '\n\n'
        print(command)
        f.write(command)

        output_dir_name = 'REQUEST_SENT/'
        command = "scavetool x --type v --filter \"module(**.server[*].app[*]) AND " \
            "\\\"requestSentRequesterID:vector\\\"\" -o {} -F CSV-R {}\n".format(OUTPUT_FILE_DIRECTORY + output_dir_name + output_file_name, vector_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)

        command = '\n\n'
        print(command)
        f.write(command)

        output_dir_name = 'REPLY_LENGTH_ASKED/'
        command = "scavetool x --type v --filter \"module(**.server[*].app[*]) AND " \
            "\\\"replyLengthAsked:vector\\\"\" -o {} -F CSV-R {}\n".format(OUTPUT_FILE_DIRECTORY + output_dir_name + output_file_name, vector_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)

        command = '\n\n'
        print(command)
        f.write(command)

        # Add FLOW_ID extraction for true flow identification
        output_dir_name = 'FLOW_ID/'
        command = "scavetool x --type v --filter \"module(**.**.relayUnit) AND " \
            "\\\"FlowID:vector\\\"\" -o {} -F CSV-R {}\n".format(OUTPUT_FILE_DIRECTORY + output_dir_name + output_file_name, vector_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)

        command = '\n\n'
        print(command)
        f.write(command)

        # Add PACKET_SIZE extraction for packet size information - now network-level aggregated
        output_dir_name = 'PACKET_SIZE/'
        command = "scavetool x --type v --filter \"module(LeafSpine1G) AND " \
            "\\\"PacketSize:vector\\\"\" -o {} -F CSV-R {}\n".format(OUTPUT_FILE_DIRECTORY + output_dir_name + output_file_name, vector_file_name)
        print(command)
        f.write('echo \"{}\"\n'.format(output_dir_name))
        f.write(command)

f.close()