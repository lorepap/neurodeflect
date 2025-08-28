#!/bin/bash
# Auto-generated extractor script

echo "sim_c2a2613f_50000"
echo "QUEUE_LEN/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"QueueLen:vector\"" -o ../../extracted_results/QUEUE_LEN/sim_c2a2613f_50000_dctcp_sd_threshold_threshold_50000.csv -F CSV-S 4_spines_8_aggs_40_servers_1_burstyapps_1_mice_40_reqPerBurst_11.85_bgintermult_1_bgfsizemult_0.11_burstyintermult_1_burstyfsizemult_250_ttl_0_rep_2_rndfwfactor_2_rndbouncefactor_20000_incastfsize_0.00120_mrktimer_0.00120_ordtimer_threshold_50000.vec


echo "QUEUES_TOT_LEN/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"QueuesTotLen:vector\"" -o ../../extracted_results/QUEUES_TOT_LEN/sim_c2a2613f_50000_dctcp_sd_threshold_threshold_50000.csv -F CSV-S 4_spines_8_aggs_40_servers_1_burstyapps_1_mice_40_reqPerBurst_11.85_bgintermult_1_bgfsizemult_0.11_burstyintermult_1_burstyfsizemult_250_ttl_0_rep_2_rndfwfactor_2_rndbouncefactor_20000_incastfsize_0.00120_mrktimer_0.00120_ordtimer_threshold_50000.vec


echo "QUEUE_CAPACITY/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"QueueCapacity:vector\"" -o ../../extracted_results/QUEUE_CAPACITY/sim_c2a2613f_50000_dctcp_sd_threshold_threshold_50000.csv -F CSV-S 4_spines_8_aggs_40_servers_1_burstyapps_1_mice_40_reqPerBurst_11.85_bgintermult_1_bgfsizemult_0.11_burstyintermult_1_burstyfsizemult_250_ttl_0_rep_2_rndfwfactor_2_rndbouncefactor_20000_incastfsize_0.00120_mrktimer_0.00120_ordtimer_threshold_50000.vec


echo "QUEUES_TOT_CAPACITY/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"QueuesTotCapacity:vector\"" -o ../../extracted_results/QUEUES_TOT_CAPACITY/sim_c2a2613f_50000_dctcp_sd_threshold_threshold_50000.csv -F CSV-S 4_spines_8_aggs_40_servers_1_burstyapps_1_mice_40_reqPerBurst_11.85_bgintermult_1_bgfsizemult_0.11_burstyintermult_1_burstyfsizemult_250_ttl_0_rep_2_rndfwfactor_2_rndbouncefactor_20000_incastfsize_0.00120_mrktimer_0.00120_ordtimer_threshold_50000.vec


echo "PACKET_ACTION/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"PacketAction:vector\"" -o ../../extracted_results/PACKET_ACTION/sim_c2a2613f_50000_dctcp_sd_threshold_threshold_50000.csv -F CSV-S 4_spines_8_aggs_40_servers_1_burstyapps_1_mice_40_reqPerBurst_11.85_bgintermult_1_bgfsizemult_0.11_burstyintermult_1_burstyfsizemult_250_ttl_0_rep_2_rndfwfactor_2_rndbouncefactor_20000_incastfsize_0.00120_mrktimer_0.00120_ordtimer_threshold_50000.vec


echo "RCV_TS_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"rcvSeq:vector\"" -o ../../extracted_results/RCV_TS_SEQ_NUM/sim_c2a2613f_50000_dctcp_sd_threshold_threshold_50000.csv -F CSV-S 4_spines_8_aggs_40_servers_1_burstyapps_1_mice_40_reqPerBurst_11.85_bgintermult_1_bgfsizemult_0.11_burstyintermult_1_burstyfsizemult_250_ttl_0_rep_2_rndfwfactor_2_rndbouncefactor_20000_incastfsize_0.00120_mrktimer_0.00120_ordtimer_threshold_50000.vec


echo "SND_TS_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"sndNxt:vector\"" -o ../../extracted_results/SND_TS_SEQ_NUM/sim_c2a2613f_50000_dctcp_sd_threshold_threshold_50000.csv -F CSV-S 4_spines_8_aggs_40_servers_1_burstyapps_1_mice_40_reqPerBurst_11.85_bgintermult_1_bgfsizemult_0.11_burstyintermult_1_burstyfsizemult_250_ttl_0_rep_2_rndfwfactor_2_rndbouncefactor_20000_incastfsize_0.00120_mrktimer_0.00120_ordtimer_threshold_50000.vec


echo "OOO_SEG/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"rcvOooSeg:vector\"" -o ../../extracted_results/OOO_SEG/sim_c2a2613f_50000_dctcp_sd_threshold_threshold_50000.csv -F CSV-S 4_spines_8_aggs_40_servers_1_burstyapps_1_mice_40_reqPerBurst_11.85_bgintermult_1_bgfsizemult_0.11_burstyintermult_1_burstyfsizemult_250_ttl_0_rep_2_rndfwfactor_2_rndbouncefactor_20000_incastfsize_0.00120_mrktimer_0.00120_ordtimer_threshold_50000.vec


echo "SWITCH_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"switchSeqNum:vector\"" -o ../../extracted_results/SWITCH_SEQ_NUM/sim_c2a2613f_50000_dctcp_sd_threshold_threshold_50000.csv -F CSV-S 4_spines_8_aggs_40_servers_1_burstyapps_1_mice_40_reqPerBurst_11.85_bgintermult_1_bgfsizemult_0.11_burstyintermult_1_burstyfsizemult_250_ttl_0_rep_2_rndfwfactor_2_rndbouncefactor_20000_incastfsize_0.00120_mrktimer_0.00120_ordtimer_threshold_50000.vec


echo "RETRANSMITTED/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"retransmitted:vector\"" -o ../../extracted_results/RETRANSMITTED/sim_c2a2613f_50000_dctcp_sd_threshold_threshold_50000.csv -F CSV-S 4_spines_8_aggs_40_servers_1_burstyapps_1_mice_40_reqPerBurst_11.85_bgintermult_1_bgfsizemult_0.11_burstyintermult_1_burstyfsizemult_250_ttl_0_rep_2_rndfwfactor_2_rndbouncefactor_20000_incastfsize_0.00120_mrktimer_0.00120_ordtimer_threshold_50000.vec


echo "TTL/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"switchTtl:vector\"" -o ../../extracted_results/TTL/sim_c2a2613f_50000_dctcp_sd_threshold_threshold_50000.csv -F CSV-S 4_spines_8_aggs_40_servers_1_burstyapps_1_mice_40_reqPerBurst_11.85_bgintermult_1_bgfsizemult_0.11_burstyintermult_1_burstyfsizemult_250_ttl_0_rep_2_rndfwfactor_2_rndbouncefactor_20000_incastfsize_0.00120_mrktimer_0.00120_ordtimer_threshold_50000.vec


echo "ACTION_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"actionSeqNum:vector\"" -o ../../extracted_results/ACTION_SEQ_NUM/sim_c2a2613f_50000_dctcp_sd_threshold_threshold_50000.csv -F CSV-S 4_spines_8_aggs_40_servers_1_burstyapps_1_mice_40_reqPerBurst_11.85_bgintermult_1_bgfsizemult_0.11_burstyintermult_1_burstyfsizemult_250_ttl_0_rep_2_rndfwfactor_2_rndbouncefactor_20000_incastfsize_0.00120_mrktimer_0.00120_ordtimer_threshold_50000.vec


echo "SWITCH_ID/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"switchId:vector\"" -o ../../extracted_results/SWITCH_ID/sim_c2a2613f_50000_dctcp_sd_threshold_threshold_50000.csv -F CSV-S 4_spines_8_aggs_40_servers_1_burstyapps_1_mice_40_reqPerBurst_11.85_bgintermult_1_bgfsizemult_0.11_burstyintermult_1_burstyfsizemult_250_ttl_0_rep_2_rndfwfactor_2_rndbouncefactor_20000_incastfsize_0.00120_mrktimer_0.00120_ordtimer_threshold_50000.vec


echo "SWITCH_ID_ACTION/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"switchIdAction:vector\"" -o ../../extracted_results/SWITCH_ID_ACTION/sim_c2a2613f_50000_dctcp_sd_threshold_threshold_50000.csv -F CSV-S 4_spines_8_aggs_40_servers_1_burstyapps_1_mice_40_reqPerBurst_11.85_bgintermult_1_bgfsizemult_0.11_burstyintermult_1_burstyfsizemult_250_ttl_0_rep_2_rndfwfactor_2_rndbouncefactor_20000_incastfsize_0.00120_mrktimer_0.00120_ordtimer_threshold_50000.vec


echo "INTERFACE_ID/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"interfaceId:vector\"" -o ../../extracted_results/INTERFACE_ID/sim_c2a2613f_50000_dctcp_sd_threshold_threshold_50000.csv -F CSV-S 4_spines_8_aggs_40_servers_1_burstyapps_1_mice_40_reqPerBurst_11.85_bgintermult_1_bgfsizemult_0.11_burstyintermult_1_burstyfsizemult_250_ttl_0_rep_2_rndfwfactor_2_rndbouncefactor_20000_incastfsize_0.00120_mrktimer_0.00120_ordtimer_threshold_50000.vec


