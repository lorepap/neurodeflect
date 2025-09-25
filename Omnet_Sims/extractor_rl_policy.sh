#!/bin/bash
# Auto-generated RL policy extractor script
# Generated for 1 RL policy result files
# Working directory: ./results/rl_policy/

echo "Processing: rl_policy_8dc66ba9_4s_8a_40srv"
echo "  FLOW_ENDED/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"flowEndedRequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_ENDED/rl_policy_8dc66ba9_4s_8a_40srv_rl_policy.csv -F CSV-S ./results/rl_policy/4_spines_8_aggs_40_servers_1_burstyapps_1_mice_40_reqPerBurst_11.85_bgintermult_1_bgfsizemult_0.11_burstyintermult_1_burstyfsizemult_250_ttl_0_rep_2_rndfwfactor_2_rndbouncefactor_20000_incastfsize_0.00120_mrktimer_0.00120_ordtimer_rl_policy.vec
echo "  FLOW_STARTED/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"flowStartedRequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_STARTED/rl_policy_8dc66ba9_4s_8a_40srv_rl_policy.csv -F CSV-S ./results/rl_policy/4_spines_8_aggs_40_servers_1_burstyapps_1_mice_40_reqPerBurst_11.85_bgintermult_1_bgfsizemult_0.11_burstyintermult_1_burstyfsizemult_250_ttl_0_rep_2_rndfwfactor_2_rndbouncefactor_20000_incastfsize_0.00120_mrktimer_0.00120_ordtimer_rl_policy.vec
echo "FLOW_ID/"
scavetool x --type v --filter "module(**.**.relayUnit) AND \"FlowID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_ID/rl_policy_8dc66ba9_4s_8a_40srv_rl_policy.csv -F CSV-R ./results/rl_policy/4_spines_8_aggs_40_servers_1_burstyapps_1_mice_40_reqPerBurst_11.85_bgintermult_1_bgfsizemult_0.11_burstyintermult_1_burstyfsizemult_250_ttl_0_rep_2_rndfwfactor_2_rndbouncefactor_20000_incastfsize_0.00120_mrktimer_0.00120_ordtimer_rl_policy.vec
echo "  REQUEST_SENT/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"requestSentRequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/REQUEST_SENT/rl_policy_8dc66ba9_4s_8a_40srv_rl_policy.csv -F CSV-S ./results/rl_policy/4_spines_8_aggs_40_servers_1_burstyapps_1_mice_40_reqPerBurst_11.85_bgintermult_1_bgfsizemult_0.11_burstyintermult_1_burstyfsizemult_250_ttl_0_rep_2_rndfwfactor_2_rndbouncefactor_20000_incastfsize_0.00120_mrktimer_0.00120_ordtimer_rl_policy.vec
echo "  REPLY_LENGTH_ASKED/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"replyLengthAsked:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/REPLY_LENGTH_ASKED/rl_policy_8dc66ba9_4s_8a_40srv_rl_policy.csv -F CSV-S ./results/rl_policy/4_spines_8_aggs_40_servers_1_burstyapps_1_mice_40_reqPerBurst_11.85_bgintermult_1_bgfsizemult_0.11_burstyintermult_1_burstyfsizemult_250_ttl_0_rep_2_rndfwfactor_2_rndbouncefactor_20000_incastfsize_0.00120_mrktimer_0.00120_ordtimer_rl_policy.vec
echo "  REQUESTER_ID/"
scavetool x --type v --filter "module(**.**.relayUnit) AND \"RequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/REQUESTER_ID/rl_policy_8dc66ba9_4s_8a_40srv_rl_policy.csv -F CSV-S ./results/rl_policy/4_spines_8_aggs_40_servers_1_burstyapps_1_mice_40_reqPerBurst_11.85_bgintermult_1_bgfsizemult_0.11_burstyintermult_1_burstyfsizemult_250_ttl_0_rep_2_rndfwfactor_2_rndbouncefactor_20000_incastfsize_0.00120_mrktimer_0.00120_ordtimer_rl_policy.vec
echo "  QUEUE_LEN/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"QueueLen:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUE_LEN/rl_policy_8dc66ba9_4s_8a_40srv_rl_policy.csv -F CSV-S ./results/rl_policy/4_spines_8_aggs_40_servers_1_burstyapps_1_mice_40_reqPerBurst_11.85_bgintermult_1_bgfsizemult_0.11_burstyintermult_1_burstyfsizemult_250_ttl_0_rep_2_rndfwfactor_2_rndbouncefactor_20000_incastfsize_0.00120_mrktimer_0.00120_ordtimer_rl_policy.vec
echo "QUEUES_TOT_LEN/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"QueuesTotLen:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUES_TOT_LEN/rl_policy_8dc66ba9_4s_8a_40srv_rl_policy.csv -F CSV-S ./results/rl_policy/4_spines_8_aggs_40_servers_1_burstyapps_1_mice_40_reqPerBurst_11.85_bgintermult_1_bgfsizemult_0.11_burstyintermult_1_burstyfsizemult_250_ttl_0_rep_2_rndfwfactor_2_rndbouncefactor_20000_incastfsize_0.00120_mrktimer_0.00120_ordtimer_rl_policy.vec
echo "  PACKET_ACTION/"
scavetool x --type v --filter "module(**) AND \"PacketAction:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/PACKET_ACTION/rl_policy_8dc66ba9_4s_8a_40srv_rl_policy.csv -F CSV-S ./results/rl_policy/4_spines_8_aggs_40_servers_1_burstyapps_1_mice_40_reqPerBurst_11.85_bgintermult_1_bgfsizemult_0.11_burstyintermult_1_burstyfsizemult_250_ttl_0_rep_2_rndfwfactor_2_rndbouncefactor_20000_incastfsize_0.00120_mrktimer_0.00120_ordtimer_rl_policy.vec
echo "RCV_TS_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"rcvSeq:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/RCV_TS_SEQ_NUM/rl_policy_8dc66ba9_4s_8a_40srv_rl_policy.csv -F CSV-S ./results/rl_policy/4_spines_8_aggs_40_servers_1_burstyapps_1_mice_40_reqPerBurst_11.85_bgintermult_1_bgfsizemult_0.11_burstyintermult_1_burstyfsizemult_250_ttl_0_rep_2_rndfwfactor_2_rndbouncefactor_20000_incastfsize_0.00120_mrktimer_0.00120_ordtimer_rl_policy.vec


echo "SND_TS_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"sndNxt:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SND_TS_SEQ_NUM/rl_policy_8dc66ba9_4s_8a_40srv_rl_policy.csv -F CSV-S ./results/rl_policy/4_spines_8_aggs_40_servers_1_burstyapps_1_mice_40_reqPerBurst_11.85_bgintermult_1_bgfsizemult_0.11_burstyintermult_1_burstyfsizemult_250_ttl_0_rep_2_rndfwfactor_2_rndbouncefactor_20000_incastfsize_0.00120_mrktimer_0.00120_ordtimer_rl_policy.vec
echo "  OOO_SEG_NUM/"
scavetool x --type s --filter "module(LeafSpine1G) AND \"numReceivedOOOSegs\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/OOO_SEG_NUM/rl_policy_8dc66ba9_4s_8a_40srv_rl_policy.csv -F CSV-S ./results/rl_policy/4_spines_8_aggs_40_servers_1_burstyapps_1_mice_40_reqPerBurst_11.85_bgintermult_1_bgfsizemult_0.11_burstyintermult_1_burstyfsizemult_250_ttl_0_rep_2_rndfwfactor_2_rndbouncefactor_20000_incastfsize_0.00120_mrktimer_0.00120_ordtimer_rl_policy.sca

echo "RL policy extraction completed!"
