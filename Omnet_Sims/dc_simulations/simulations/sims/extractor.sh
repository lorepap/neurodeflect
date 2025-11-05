#!/bin/bash
# Auto-generated extractor script

echo "sim_8dc66ba9"
echo "QUEUE_LEN/"
scavetool x --type v --filter "(module(**.agg[*].relayUnit) OR module(**.spine[*].relayUnit)) AND name(QueueLen:vector)" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUE_LEN/sim_8dc66ba9_rl_policy.csv -F CSV-R rl_policy/4_spines_8_aggs_40_servers_1_burstyapps_1_mice_40_reqPerBurst_11.85_bgintermult_1_bgfsizemult_0.11_burstyintermult_1_burstyfsizemult_250_ttl_0_rep_2_rndfwfactor_2_rndbouncefactor_20000_incastfsize_0.00120_mrktimer_0.00120_ordtimer_rl_policy.vec


echo "QUEUES_TOT_LEN/"
scavetool x --type v --filter "(module(**.agg[*].relayUnit) OR module(**.spine[*].relayUnit)) AND name(QueuesTotLen:vector)" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUES_TOT_LEN/sim_8dc66ba9_rl_policy.csv -F CSV-R rl_policy/4_spines_8_aggs_40_servers_1_burstyapps_1_mice_40_reqPerBurst_11.85_bgintermult_1_bgfsizemult_0.11_burstyintermult_1_burstyfsizemult_250_ttl_0_rep_2_rndfwfactor_2_rndbouncefactor_20000_incastfsize_0.00120_mrktimer_0.00120_ordtimer_rl_policy.vec


echo "QUEUE_CAPACITY/"
scavetool x --type v --filter "(module(**.agg[*].relayUnit) OR module(**.spine[*].relayUnit)) AND name(QueueCapacity:vector)" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUE_CAPACITY/sim_8dc66ba9_rl_policy.csv -F CSV-R rl_policy/4_spines_8_aggs_40_servers_1_burstyapps_1_mice_40_reqPerBurst_11.85_bgintermult_1_bgfsizemult_0.11_burstyintermult_1_burstyfsizemult_250_ttl_0_rep_2_rndfwfactor_2_rndbouncefactor_20000_incastfsize_0.00120_mrktimer_0.00120_ordtimer_rl_policy.vec


echo "QUEUES_TOT_CAPACITY/"
scavetool x --type v --filter "(module(**.agg[*].relayUnit) OR module(**.spine[*].relayUnit)) AND name(QueuesTotCapacity:vector)" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUES_TOT_CAPACITY/sim_8dc66ba9_rl_policy.csv -F CSV-R rl_policy/4_spines_8_aggs_40_servers_1_burstyapps_1_mice_40_reqPerBurst_11.85_bgintermult_1_bgfsizemult_0.11_burstyintermult_1_burstyfsizemult_250_ttl_0_rep_2_rndfwfactor_2_rndbouncefactor_20000_incastfsize_0.00120_mrktimer_0.00120_ordtimer_rl_policy.vec


echo "PACKET_ACTION/"
scavetool x --type v --filter "(module(**.agg[*].relayUnit) OR module(**.spine[*].relayUnit)) AND name(PacketAction:vector)" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/PACKET_ACTION/sim_8dc66ba9_rl_policy.csv -F CSV-R rl_policy/4_spines_8_aggs_40_servers_1_burstyapps_1_mice_40_reqPerBurst_11.85_bgintermult_1_bgfsizemult_0.11_burstyintermult_1_burstyfsizemult_250_ttl_0_rep_2_rndfwfactor_2_rndbouncefactor_20000_incastfsize_0.00120_mrktimer_0.00120_ordtimer_rl_policy.vec


echo "FLOW_ENDED_QUERY_ID/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"flowEndedQueryID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_ENDED_QUERY_ID/sim_8dc66ba9_rl_policy.csv -F CSV-R rl_policy/4_spines_8_aggs_40_servers_1_burstyapps_1_mice_40_reqPerBurst_11.85_bgintermult_1_bgfsizemult_0.11_burstyintermult_1_burstyfsizemult_250_ttl_0_rep_2_rndfwfactor_2_rndbouncefactor_20000_incastfsize_0.00120_mrktimer_0.00120_ordtimer_rl_policy.vec


echo "RCV_TS_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"rcvSeq:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/RCV_TS_SEQ_NUM/sim_8dc66ba9_rl_policy.csv -F CSV-R rl_policy/4_spines_8_aggs_40_servers_1_burstyapps_1_mice_40_reqPerBurst_11.85_bgintermult_1_bgfsizemult_0.11_burstyintermult_1_burstyfsizemult_250_ttl_0_rep_2_rndfwfactor_2_rndbouncefactor_20000_incastfsize_0.00120_mrktimer_0.00120_ordtimer_rl_policy.vec


echo "SND_TS_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"sndNxt:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SND_TS_SEQ_NUM/sim_8dc66ba9_rl_policy.csv -F CSV-R rl_policy/4_spines_8_aggs_40_servers_1_burstyapps_1_mice_40_reqPerBurst_11.85_bgintermult_1_bgfsizemult_0.11_burstyintermult_1_burstyfsizemult_250_ttl_0_rep_2_rndfwfactor_2_rndbouncefactor_20000_incastfsize_0.00120_mrktimer_0.00120_ordtimer_rl_policy.vec


echo "OOO_SEG/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"rcvOooSeg:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/OOO_SEG/sim_8dc66ba9_rl_policy.csv -F CSV-R rl_policy/4_spines_8_aggs_40_servers_1_burstyapps_1_mice_40_reqPerBurst_11.85_bgintermult_1_bgfsizemult_0.11_burstyintermult_1_burstyfsizemult_250_ttl_0_rep_2_rndfwfactor_2_rndbouncefactor_20000_incastfsize_0.00120_mrktimer_0.00120_ordtimer_rl_policy.vec


echo "SWITCH_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"switchSeqNum:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SWITCH_SEQ_NUM/sim_8dc66ba9_rl_policy.csv -F CSV-R rl_policy/4_spines_8_aggs_40_servers_1_burstyapps_1_mice_40_reqPerBurst_11.85_bgintermult_1_bgfsizemult_0.11_burstyintermult_1_burstyfsizemult_250_ttl_0_rep_2_rndfwfactor_2_rndbouncefactor_20000_incastfsize_0.00120_mrktimer_0.00120_ordtimer_rl_policy.vec


echo "RETRANSMITTED/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"retransmitted:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/RETRANSMITTED/sim_8dc66ba9_rl_policy.csv -F CSV-R rl_policy/4_spines_8_aggs_40_servers_1_burstyapps_1_mice_40_reqPerBurst_11.85_bgintermult_1_bgfsizemult_0.11_burstyintermult_1_burstyfsizemult_250_ttl_0_rep_2_rndfwfactor_2_rndbouncefactor_20000_incastfsize_0.00120_mrktimer_0.00120_ordtimer_rl_policy.vec


echo "TTL/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"switchTtl:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/TTL/sim_8dc66ba9_rl_policy.csv -F CSV-R rl_policy/4_spines_8_aggs_40_servers_1_burstyapps_1_mice_40_reqPerBurst_11.85_bgintermult_1_bgfsizemult_0.11_burstyintermult_1_burstyfsizemult_250_ttl_0_rep_2_rndfwfactor_2_rndbouncefactor_20000_incastfsize_0.00120_mrktimer_0.00120_ordtimer_rl_policy.vec


echo "ACTION_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"actionSeqNum:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/ACTION_SEQ_NUM/sim_8dc66ba9_rl_policy.csv -F CSV-R rl_policy/4_spines_8_aggs_40_servers_1_burstyapps_1_mice_40_reqPerBurst_11.85_bgintermult_1_bgfsizemult_0.11_burstyintermult_1_burstyfsizemult_250_ttl_0_rep_2_rndfwfactor_2_rndbouncefactor_20000_incastfsize_0.00120_mrktimer_0.00120_ordtimer_rl_policy.vec


echo "SWITCH_ID/"
scavetool x --type v --filter "module(**.relayUnit) AND \"switchId:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SWITCH_ID/sim_8dc66ba9_rl_policy.csv -F CSV-R rl_policy/4_spines_8_aggs_40_servers_1_burstyapps_1_mice_40_reqPerBurst_11.85_bgintermult_1_bgfsizemult_0.11_burstyintermult_1_burstyfsizemult_250_ttl_0_rep_2_rndfwfactor_2_rndbouncefactor_20000_incastfsize_0.00120_mrktimer_0.00120_ordtimer_rl_policy.vec


echo "SWITCH_ID_ACTION/"
scavetool x --type v --filter "module(**.relayUnit) AND \"switchIdAction:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SWITCH_ID_ACTION/sim_8dc66ba9_rl_policy.csv -F CSV-R rl_policy/4_spines_8_aggs_40_servers_1_burstyapps_1_mice_40_reqPerBurst_11.85_bgintermult_1_bgfsizemult_0.11_burstyintermult_1_burstyfsizemult_250_ttl_0_rep_2_rndfwfactor_2_rndbouncefactor_20000_incastfsize_0.00120_mrktimer_0.00120_ordtimer_rl_policy.vec


echo "INTERFACE_ID/"
scavetool x --type v --filter "module(**.relayUnit) AND \"interfaceId:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/INTERFACE_ID/sim_8dc66ba9_rl_policy.csv -F CSV-R rl_policy/4_spines_8_aggs_40_servers_1_burstyapps_1_mice_40_reqPerBurst_11.85_bgintermult_1_bgfsizemult_0.11_burstyintermult_1_burstyfsizemult_250_ttl_0_rep_2_rndfwfactor_2_rndbouncefactor_20000_incastfsize_0.00120_mrktimer_0.00120_ordtimer_rl_policy.vec


echo "REQUESTER_ID/"
scavetool x --type v --filter "module(**.**.relayUnit) AND \"RequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/REQUESTER_ID/sim_8dc66ba9_rl_policy.csv -F CSV-R rl_policy/4_spines_8_aggs_40_servers_1_burstyapps_1_mice_40_reqPerBurst_11.85_bgintermult_1_bgfsizemult_0.11_burstyintermult_1_burstyfsizemult_250_ttl_0_rep_2_rndfwfactor_2_rndbouncefactor_20000_incastfsize_0.00120_mrktimer_0.00120_ordtimer_rl_policy.vec


echo "FLOW_STARTED/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"flowStartedRequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_STARTED/sim_8dc66ba9_rl_policy.csv -F CSV-R rl_policy/4_spines_8_aggs_40_servers_1_burstyapps_1_mice_40_reqPerBurst_11.85_bgintermult_1_bgfsizemult_0.11_burstyintermult_1_burstyfsizemult_250_ttl_0_rep_2_rndfwfactor_2_rndbouncefactor_20000_incastfsize_0.00120_mrktimer_0.00120_ordtimer_rl_policy.vec


echo "FLOW_ENDED/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"flowEndedRequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_ENDED/sim_8dc66ba9_rl_policy.csv -F CSV-R rl_policy/4_spines_8_aggs_40_servers_1_burstyapps_1_mice_40_reqPerBurst_11.85_bgintermult_1_bgfsizemult_0.11_burstyintermult_1_burstyfsizemult_250_ttl_0_rep_2_rndfwfactor_2_rndbouncefactor_20000_incastfsize_0.00120_mrktimer_0.00120_ordtimer_rl_policy.vec


echo "REQUEST_SENT/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"requestSentRequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/REQUEST_SENT/sim_8dc66ba9_rl_policy.csv -F CSV-R rl_policy/4_spines_8_aggs_40_servers_1_burstyapps_1_mice_40_reqPerBurst_11.85_bgintermult_1_bgfsizemult_0.11_burstyintermult_1_burstyfsizemult_250_ttl_0_rep_2_rndfwfactor_2_rndbouncefactor_20000_incastfsize_0.00120_mrktimer_0.00120_ordtimer_rl_policy.vec


echo "REPLY_LENGTH_ASKED/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"replyLengthAsked:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/REPLY_LENGTH_ASKED/sim_8dc66ba9_rl_policy.csv -F CSV-R rl_policy/4_spines_8_aggs_40_servers_1_burstyapps_1_mice_40_reqPerBurst_11.85_bgintermult_1_bgfsizemult_0.11_burstyintermult_1_burstyfsizemult_250_ttl_0_rep_2_rndfwfactor_2_rndbouncefactor_20000_incastfsize_0.00120_mrktimer_0.00120_ordtimer_rl_policy.vec


echo "FLOW_ID/"
scavetool x --type v --filter "module(**.**.relayUnit) AND \"FlowID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_ID/sim_8dc66ba9_rl_policy.csv -F CSV-R rl_policy/4_spines_8_aggs_40_servers_1_burstyapps_1_mice_40_reqPerBurst_11.85_bgintermult_1_bgfsizemult_0.11_burstyintermult_1_burstyfsizemult_250_ttl_0_rep_2_rndfwfactor_2_rndbouncefactor_20000_incastfsize_0.00120_mrktimer_0.00120_ordtimer_rl_policy.vec


echo "PACKET_SIZE/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"PacketSize:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/PACKET_SIZE/sim_8dc66ba9_rl_policy.csv -F CSV-R rl_policy/4_spines_8_aggs_40_servers_1_burstyapps_1_mice_40_reqPerBurst_11.85_bgintermult_1_bgfsizemult_0.11_burstyintermult_1_burstyfsizemult_250_ttl_0_rep_2_rndfwfactor_2_rndbouncefactor_20000_incastfsize_0.00120_mrktimer_0.00120_ordtimer_rl_policy.vec
