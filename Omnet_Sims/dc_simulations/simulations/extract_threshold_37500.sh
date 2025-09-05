#!/bin/bash
# Extract data for threshold 37500

BASE_DIR="/home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations"
VEC_FILE="sims/results/threshold_37500/4_spines_8_aggs_40_servers_1_burstyapps_1_mice_40_reqPerBurst_11.85_bgintermult_1_bgfsizemult_0.11_burstyintermult_1_burstyfsizemult_250_ttl_0_rep_2_rndfwfactor_2_rndbouncefactor_20000_incastfsize_0.00120_mrktimer_0.00120_ordtimer_threshold_37500.vec"
OUTPUT_PREFIX="sims/extracted_results"
FILE_PREFIX="sim_99ffc3ea_37500_37500"

cd "$BASE_DIR"

# Create output directories
mkdir -p "$OUTPUT_PREFIX"/{QUEUE_LEN,QUEUES_TOT_LEN,QUEUE_CAPACITY,QUEUES_TOT_CAPACITY,PACKET_ACTION,RCV_TS_SEQ_NUM,SND_TS_SEQ_NUM,OOO_SEG,SWITCH_SEQ_NUM,RETRANSMITTED,TTL,ACTION_SEQ_NUM,SWITCH_ID,SWITCH_ID_ACTION,INTERFACE_ID,REQUESTER_ID,FLOW_STARTED,FLOW_ENDED,REQUEST_SENT,REPLY_LENGTH_ASKED,FLOW_ID,PACKET_SIZE}

echo "Extracting QUEUE_LEN..."
scavetool x --type v --filter "module(LeafSpine1G) AND \"QueueLen:vector\"" -o "$OUTPUT_PREFIX/QUEUE_LEN/${FILE_PREFIX}.csv" -F CSV-S "$VEC_FILE"

echo "Extracting QUEUES_TOT_LEN..."
scavetool x --type v --filter "module(LeafSpine1G) AND \"QueuesTotLen:vector\"" -o "$OUTPUT_PREFIX/QUEUES_TOT_LEN/${FILE_PREFIX}.csv" -F CSV-S "$VEC_FILE"

echo "Extracting QUEUE_CAPACITY..."
scavetool x --type v --filter "module(LeafSpine1G) AND \"QueueCapacity:vector\"" -o "$OUTPUT_PREFIX/QUEUE_CAPACITY/${FILE_PREFIX}.csv" -F CSV-S "$VEC_FILE"

echo "Extracting QUEUES_TOT_CAPACITY..."
scavetool x --type v --filter "module(LeafSpine1G) AND \"QueuesTotCapacity:vector\"" -o "$OUTPUT_PREFIX/QUEUES_TOT_CAPACITY/${FILE_PREFIX}.csv" -F CSV-S "$VEC_FILE"

echo "Extracting PACKET_ACTION..."
scavetool x --type v --filter "module(LeafSpine1G) AND \"PacketAction:vector\"" -o "$OUTPUT_PREFIX/PACKET_ACTION/${FILE_PREFIX}.csv" -F CSV-S "$VEC_FILE"

echo "Extracting RCV_TS_SEQ_NUM..."
scavetool x --type v --filter "module(LeafSpine1G) AND \"rcvSeq:vector\"" -o "$OUTPUT_PREFIX/RCV_TS_SEQ_NUM/${FILE_PREFIX}.csv" -F CSV-S "$VEC_FILE"

echo "Extracting SND_TS_SEQ_NUM..."
scavetool x --type v --filter "module(LeafSpine1G) AND \"sndNxt:vector\"" -o "$OUTPUT_PREFIX/SND_TS_SEQ_NUM/${FILE_PREFIX}.csv" -F CSV-S "$VEC_FILE"

echo "Extracting OOO_SEG..."
scavetool x --type v --filter "module(LeafSpine1G) AND \"rcvOooSeg:vector\"" -o "$OUTPUT_PREFIX/OOO_SEG/${FILE_PREFIX}.csv" -F CSV-S "$VEC_FILE"

echo "Extracting SWITCH_SEQ_NUM..."
scavetool x --type v --filter "module(LeafSpine1G) AND \"switchSeqNum:vector\"" -o "$OUTPUT_PREFIX/SWITCH_SEQ_NUM/${FILE_PREFIX}.csv" -F CSV-S "$VEC_FILE"

echo "Extracting RETRANSMITTED..."
scavetool x --type v --filter "module(LeafSpine1G) AND \"retransmitted:vector\"" -o "$OUTPUT_PREFIX/RETRANSMITTED/${FILE_PREFIX}.csv" -F CSV-S "$VEC_FILE"

echo "Extracting TTL..."
scavetool x --type v --filter "module(LeafSpine1G) AND \"switchTtl:vector\"" -o "$OUTPUT_PREFIX/TTL/${FILE_PREFIX}.csv" -F CSV-S "$VEC_FILE"

echo "Extracting ACTION_SEQ_NUM..."
scavetool x --type v --filter "module(LeafSpine1G) AND \"actionSeqNum:vector\"" -o "$OUTPUT_PREFIX/ACTION_SEQ_NUM/${FILE_PREFIX}.csv" -F CSV-S "$VEC_FILE"

echo "Extracting PACKET_SIZE..."
scavetool x --type v --filter "module(LeafSpine1G) AND \"PacketSize:vector\"" -o "$OUTPUT_PREFIX/PACKET_SIZE/${FILE_PREFIX}.csv" -F CSV-S "$VEC_FILE"

echo "Extracting SWITCH_ID..."
scavetool x --type v --filter "module(**.relayUnit) AND \"switchId:vector\"" -o "$OUTPUT_PREFIX/SWITCH_ID/${FILE_PREFIX}.csv" -F CSV-R "$VEC_FILE"

echo "Extracting SWITCH_ID_ACTION..."
scavetool x --type v --filter "module(**.relayUnit) AND \"switchIdAction:vector\"" -o "$OUTPUT_PREFIX/SWITCH_ID_ACTION/${FILE_PREFIX}.csv" -F CSV-R "$VEC_FILE"

echo "Extracting INTERFACE_ID..."
scavetool x --type v --filter "module(**.relayUnit) AND \"interfaceId:vector\"" -o "$OUTPUT_PREFIX/INTERFACE_ID/${FILE_PREFIX}.csv" -F CSV-R "$VEC_FILE"

echo "Extracting FLOW_ID..."
scavetool x --type v --filter "module(**.**.relayUnit) AND \"FlowID:vector\"" -o "$OUTPUT_PREFIX/FLOW_ID/${FILE_PREFIX}.csv" -F CSV-R "$VEC_FILE"

echo "Extraction complete!"
