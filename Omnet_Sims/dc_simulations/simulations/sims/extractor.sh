#!/bin/bash
# Auto-generated extractor script

echo "sim_0fb2bd7b_beta0.15_theta0.75"
echo "QUEUE_LEN/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"QueueLen:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUE_LEN/sim_0fb2bd7b_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_10000_tbBurst_1000_rep_0.vec


echo "QUEUES_TOT_LEN/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"QueuesTotLen:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUES_TOT_LEN/sim_0fb2bd7b_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_10000_tbBurst_1000_rep_0.vec


echo "QUEUE_CAPACITY/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"QueueCapacity:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUE_CAPACITY/sim_0fb2bd7b_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_10000_tbBurst_1000_rep_0.vec


echo "QUEUES_TOT_CAPACITY/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"QueuesTotCapacity:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUES_TOT_CAPACITY/sim_0fb2bd7b_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_10000_tbBurst_1000_rep_0.vec


echo "PACKET_ACTION/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"PacketAction:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/PACKET_ACTION/sim_0fb2bd7b_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_10000_tbBurst_1000_rep_0.vec


echo "RCV_TS_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"rcvSeq:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/RCV_TS_SEQ_NUM/sim_0fb2bd7b_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_10000_tbBurst_1000_rep_0.vec


echo "SND_TS_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"sndNxt:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SND_TS_SEQ_NUM/sim_0fb2bd7b_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_10000_tbBurst_1000_rep_0.vec


echo "OOO_SEG/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"rcvOooSeg:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/OOO_SEG/sim_0fb2bd7b_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_10000_tbBurst_1000_rep_0.vec


echo "SWITCH_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"switchSeqNum:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SWITCH_SEQ_NUM/sim_0fb2bd7b_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_10000_tbBurst_1000_rep_0.vec


echo "RETRANSMITTED/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"retransmitted:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/RETRANSMITTED/sim_0fb2bd7b_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_10000_tbBurst_1000_rep_0.vec


echo "TTL/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"switchTtl:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/TTL/sim_0fb2bd7b_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_10000_tbBurst_1000_rep_0.vec


echo "ACTION_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"actionSeqNum:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/ACTION_SEQ_NUM/sim_0fb2bd7b_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_10000_tbBurst_1000_rep_0.vec


echo "SWITCH_ID/"
scavetool x --type v --filter "module(**.relayUnit) AND \"switchId:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SWITCH_ID/sim_0fb2bd7b_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-R probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_10000_tbBurst_1000_rep_0.vec


echo "SWITCH_ID_ACTION/"
scavetool x --type v --filter "module(**.relayUnit) AND \"switchIdAction:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SWITCH_ID_ACTION/sim_0fb2bd7b_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-R probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_10000_tbBurst_1000_rep_0.vec


echo "INTERFACE_ID/"
scavetool x --type v --filter "module(**.relayUnit) AND \"interfaceId:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/INTERFACE_ID/sim_0fb2bd7b_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-R probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_10000_tbBurst_1000_rep_0.vec


echo "REQUESTER_ID/"
scavetool x --type v --filter "module(**.**.relayUnit) AND \"RequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/REQUESTER_ID/sim_0fb2bd7b_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_10000_tbBurst_1000_rep_0.vec


echo "FLOW_STARTED/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"flowStartedRequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_STARTED/sim_0fb2bd7b_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_10000_tbBurst_1000_rep_0.vec


echo "FLOW_ENDED/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"flowEndedRequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_ENDED/sim_0fb2bd7b_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_10000_tbBurst_1000_rep_0.vec


echo "REQUEST_SENT/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"requestSentRequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/REQUEST_SENT/sim_0fb2bd7b_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-R probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_10000_tbBurst_1000_rep_0.vec


echo "REPLY_LENGTH_ASKED/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"replyLengthAsked:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/REPLY_LENGTH_ASKED/sim_0fb2bd7b_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-R probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_10000_tbBurst_1000_rep_0.vec


echo "FLOW_ID/"
scavetool x --type v --filter "module(**.**.relayUnit) AND \"FlowID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_ID/sim_0fb2bd7b_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-R probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_10000_tbBurst_1000_rep_0.vec


echo "PACKET_SIZE/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"PacketSize:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/PACKET_SIZE/sim_0fb2bd7b_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_10000_tbBurst_1000_rep_0.vec
echo "sim_80dcefcf_beta0.15_theta0.75"
echo "QUEUE_LEN/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"QueueLen:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUE_LEN/sim_80dcefcf_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_5000_tbBurst_1000_rep_0.vec


echo "QUEUES_TOT_LEN/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"QueuesTotLen:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUES_TOT_LEN/sim_80dcefcf_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_5000_tbBurst_1000_rep_0.vec


echo "QUEUE_CAPACITY/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"QueueCapacity:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUE_CAPACITY/sim_80dcefcf_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_5000_tbBurst_1000_rep_0.vec


echo "QUEUES_TOT_CAPACITY/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"QueuesTotCapacity:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUES_TOT_CAPACITY/sim_80dcefcf_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_5000_tbBurst_1000_rep_0.vec


echo "PACKET_ACTION/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"PacketAction:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/PACKET_ACTION/sim_80dcefcf_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_5000_tbBurst_1000_rep_0.vec


echo "RCV_TS_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"rcvSeq:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/RCV_TS_SEQ_NUM/sim_80dcefcf_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_5000_tbBurst_1000_rep_0.vec


echo "SND_TS_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"sndNxt:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SND_TS_SEQ_NUM/sim_80dcefcf_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_5000_tbBurst_1000_rep_0.vec


echo "OOO_SEG/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"rcvOooSeg:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/OOO_SEG/sim_80dcefcf_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_5000_tbBurst_1000_rep_0.vec


echo "SWITCH_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"switchSeqNum:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SWITCH_SEQ_NUM/sim_80dcefcf_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_5000_tbBurst_1000_rep_0.vec


echo "RETRANSMITTED/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"retransmitted:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/RETRANSMITTED/sim_80dcefcf_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_5000_tbBurst_1000_rep_0.vec


echo "TTL/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"switchTtl:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/TTL/sim_80dcefcf_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_5000_tbBurst_1000_rep_0.vec


echo "ACTION_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"actionSeqNum:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/ACTION_SEQ_NUM/sim_80dcefcf_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_5000_tbBurst_1000_rep_0.vec


echo "SWITCH_ID/"
scavetool x --type v --filter "module(**.relayUnit) AND \"switchId:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SWITCH_ID/sim_80dcefcf_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-R probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_5000_tbBurst_1000_rep_0.vec


echo "SWITCH_ID_ACTION/"
scavetool x --type v --filter "module(**.relayUnit) AND \"switchIdAction:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SWITCH_ID_ACTION/sim_80dcefcf_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-R probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_5000_tbBurst_1000_rep_0.vec


echo "INTERFACE_ID/"
scavetool x --type v --filter "module(**.relayUnit) AND \"interfaceId:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/INTERFACE_ID/sim_80dcefcf_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-R probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_5000_tbBurst_1000_rep_0.vec


echo "REQUESTER_ID/"
scavetool x --type v --filter "module(**.**.relayUnit) AND \"RequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/REQUESTER_ID/sim_80dcefcf_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_5000_tbBurst_1000_rep_0.vec


echo "FLOW_STARTED/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"flowStartedRequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_STARTED/sim_80dcefcf_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_5000_tbBurst_1000_rep_0.vec


echo "FLOW_ENDED/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"flowEndedRequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_ENDED/sim_80dcefcf_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_5000_tbBurst_1000_rep_0.vec


echo "REQUEST_SENT/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"requestSentRequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/REQUEST_SENT/sim_80dcefcf_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-R probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_5000_tbBurst_1000_rep_0.vec


echo "REPLY_LENGTH_ASKED/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"replyLengthAsked:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/REPLY_LENGTH_ASKED/sim_80dcefcf_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-R probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_5000_tbBurst_1000_rep_0.vec


echo "FLOW_ID/"
scavetool x --type v --filter "module(**.**.relayUnit) AND \"FlowID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_ID/sim_80dcefcf_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-R probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_5000_tbBurst_1000_rep_0.vec


echo "PACKET_SIZE/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"PacketSize:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/PACKET_SIZE/sim_80dcefcf_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_5000_tbBurst_1000_rep_0.vec
echo "sim_466096dc_beta0.15_theta0.75"
echo "QUEUE_LEN/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"QueueLen:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUE_LEN/sim_466096dc_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_50_tbBurst_25_rep_0.vec


echo "QUEUES_TOT_LEN/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"QueuesTotLen:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUES_TOT_LEN/sim_466096dc_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_50_tbBurst_25_rep_0.vec


echo "QUEUE_CAPACITY/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"QueueCapacity:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUE_CAPACITY/sim_466096dc_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_50_tbBurst_25_rep_0.vec


echo "QUEUES_TOT_CAPACITY/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"QueuesTotCapacity:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUES_TOT_CAPACITY/sim_466096dc_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_50_tbBurst_25_rep_0.vec


echo "PACKET_ACTION/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"PacketAction:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/PACKET_ACTION/sim_466096dc_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_50_tbBurst_25_rep_0.vec


echo "RCV_TS_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"rcvSeq:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/RCV_TS_SEQ_NUM/sim_466096dc_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_50_tbBurst_25_rep_0.vec


echo "SND_TS_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"sndNxt:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SND_TS_SEQ_NUM/sim_466096dc_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_50_tbBurst_25_rep_0.vec


echo "OOO_SEG/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"rcvOooSeg:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/OOO_SEG/sim_466096dc_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_50_tbBurst_25_rep_0.vec


echo "SWITCH_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"switchSeqNum:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SWITCH_SEQ_NUM/sim_466096dc_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_50_tbBurst_25_rep_0.vec


echo "RETRANSMITTED/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"retransmitted:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/RETRANSMITTED/sim_466096dc_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_50_tbBurst_25_rep_0.vec


echo "TTL/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"switchTtl:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/TTL/sim_466096dc_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_50_tbBurst_25_rep_0.vec


echo "ACTION_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"actionSeqNum:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/ACTION_SEQ_NUM/sim_466096dc_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_50_tbBurst_25_rep_0.vec


echo "SWITCH_ID/"
scavetool x --type v --filter "module(**.relayUnit) AND \"switchId:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SWITCH_ID/sim_466096dc_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-R probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_50_tbBurst_25_rep_0.vec


echo "SWITCH_ID_ACTION/"
scavetool x --type v --filter "module(**.relayUnit) AND \"switchIdAction:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SWITCH_ID_ACTION/sim_466096dc_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-R probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_50_tbBurst_25_rep_0.vec


echo "INTERFACE_ID/"
scavetool x --type v --filter "module(**.relayUnit) AND \"interfaceId:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/INTERFACE_ID/sim_466096dc_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-R probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_50_tbBurst_25_rep_0.vec


echo "REQUESTER_ID/"
scavetool x --type v --filter "module(**.**.relayUnit) AND \"RequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/REQUESTER_ID/sim_466096dc_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_50_tbBurst_25_rep_0.vec


echo "FLOW_STARTED/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"flowStartedRequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_STARTED/sim_466096dc_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_50_tbBurst_25_rep_0.vec


echo "FLOW_ENDED/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"flowEndedRequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_ENDED/sim_466096dc_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_50_tbBurst_25_rep_0.vec


echo "REQUEST_SENT/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"requestSentRequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/REQUEST_SENT/sim_466096dc_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-R probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_50_tbBurst_25_rep_0.vec


echo "REPLY_LENGTH_ASKED/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"replyLengthAsked:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/REPLY_LENGTH_ASKED/sim_466096dc_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-R probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_50_tbBurst_25_rep_0.vec


echo "FLOW_ID/"
scavetool x --type v --filter "module(**.**.relayUnit) AND \"FlowID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_ID/sim_466096dc_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-R probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_50_tbBurst_25_rep_0.vec


echo "PACKET_SIZE/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"PacketSize:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/PACKET_SIZE/sim_466096dc_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_50_tbBurst_25_rep_0.vec
echo "sim_76aa311b_beta0.15_theta0.75"
echo "QUEUE_LEN/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"QueueLen:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUE_LEN/sim_76aa311b_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_5000_tbBurst_2000_rep_0.vec


echo "QUEUES_TOT_LEN/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"QueuesTotLen:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUES_TOT_LEN/sim_76aa311b_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_5000_tbBurst_2000_rep_0.vec


echo "QUEUE_CAPACITY/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"QueueCapacity:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUE_CAPACITY/sim_76aa311b_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_5000_tbBurst_2000_rep_0.vec


echo "QUEUES_TOT_CAPACITY/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"QueuesTotCapacity:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUES_TOT_CAPACITY/sim_76aa311b_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_5000_tbBurst_2000_rep_0.vec


echo "PACKET_ACTION/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"PacketAction:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/PACKET_ACTION/sim_76aa311b_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_5000_tbBurst_2000_rep_0.vec


echo "RCV_TS_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"rcvSeq:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/RCV_TS_SEQ_NUM/sim_76aa311b_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_5000_tbBurst_2000_rep_0.vec


echo "SND_TS_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"sndNxt:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SND_TS_SEQ_NUM/sim_76aa311b_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_5000_tbBurst_2000_rep_0.vec


echo "OOO_SEG/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"rcvOooSeg:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/OOO_SEG/sim_76aa311b_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_5000_tbBurst_2000_rep_0.vec


echo "SWITCH_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"switchSeqNum:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SWITCH_SEQ_NUM/sim_76aa311b_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_5000_tbBurst_2000_rep_0.vec


echo "RETRANSMITTED/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"retransmitted:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/RETRANSMITTED/sim_76aa311b_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_5000_tbBurst_2000_rep_0.vec


echo "TTL/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"switchTtl:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/TTL/sim_76aa311b_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_5000_tbBurst_2000_rep_0.vec


echo "ACTION_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"actionSeqNum:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/ACTION_SEQ_NUM/sim_76aa311b_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_5000_tbBurst_2000_rep_0.vec


echo "SWITCH_ID/"
scavetool x --type v --filter "module(**.relayUnit) AND \"switchId:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SWITCH_ID/sim_76aa311b_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-R probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_5000_tbBurst_2000_rep_0.vec


echo "SWITCH_ID_ACTION/"
scavetool x --type v --filter "module(**.relayUnit) AND \"switchIdAction:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SWITCH_ID_ACTION/sim_76aa311b_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-R probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_5000_tbBurst_2000_rep_0.vec


echo "INTERFACE_ID/"
scavetool x --type v --filter "module(**.relayUnit) AND \"interfaceId:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/INTERFACE_ID/sim_76aa311b_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-R probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_5000_tbBurst_2000_rep_0.vec


echo "REQUESTER_ID/"
scavetool x --type v --filter "module(**.**.relayUnit) AND \"RequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/REQUESTER_ID/sim_76aa311b_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_5000_tbBurst_2000_rep_0.vec


echo "FLOW_STARTED/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"flowStartedRequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_STARTED/sim_76aa311b_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_5000_tbBurst_2000_rep_0.vec


echo "FLOW_ENDED/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"flowEndedRequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_ENDED/sim_76aa311b_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_5000_tbBurst_2000_rep_0.vec


echo "REQUEST_SENT/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"requestSentRequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/REQUEST_SENT/sim_76aa311b_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-R probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_5000_tbBurst_2000_rep_0.vec


echo "REPLY_LENGTH_ASKED/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"replyLengthAsked:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/REPLY_LENGTH_ASKED/sim_76aa311b_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-R probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_5000_tbBurst_2000_rep_0.vec


echo "FLOW_ID/"
scavetool x --type v --filter "module(**.**.relayUnit) AND \"FlowID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_ID/sim_76aa311b_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-R probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_5000_tbBurst_2000_rep_0.vec


echo "PACKET_SIZE/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"PacketSize:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/PACKET_SIZE/sim_76aa311b_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_5000_tbBurst_2000_rep_0.vec
echo "sim_a405a8cd_beta0.05_theta0.7"
echo "QUEUE_LEN/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"QueueLen:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUE_LEN/sim_a405a8cd_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_10000_tbBurst_2000_rep_0.vec


echo "QUEUES_TOT_LEN/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"QueuesTotLen:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUES_TOT_LEN/sim_a405a8cd_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_10000_tbBurst_2000_rep_0.vec


echo "QUEUE_CAPACITY/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"QueueCapacity:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUE_CAPACITY/sim_a405a8cd_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_10000_tbBurst_2000_rep_0.vec


echo "QUEUES_TOT_CAPACITY/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"QueuesTotCapacity:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUES_TOT_CAPACITY/sim_a405a8cd_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_10000_tbBurst_2000_rep_0.vec


echo "PACKET_ACTION/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"PacketAction:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/PACKET_ACTION/sim_a405a8cd_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_10000_tbBurst_2000_rep_0.vec


echo "RCV_TS_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"rcvSeq:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/RCV_TS_SEQ_NUM/sim_a405a8cd_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_10000_tbBurst_2000_rep_0.vec


echo "SND_TS_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"sndNxt:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SND_TS_SEQ_NUM/sim_a405a8cd_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_10000_tbBurst_2000_rep_0.vec


echo "OOO_SEG/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"rcvOooSeg:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/OOO_SEG/sim_a405a8cd_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_10000_tbBurst_2000_rep_0.vec


echo "SWITCH_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"switchSeqNum:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SWITCH_SEQ_NUM/sim_a405a8cd_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_10000_tbBurst_2000_rep_0.vec


echo "RETRANSMITTED/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"retransmitted:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/RETRANSMITTED/sim_a405a8cd_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_10000_tbBurst_2000_rep_0.vec


echo "TTL/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"switchTtl:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/TTL/sim_a405a8cd_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_10000_tbBurst_2000_rep_0.vec


echo "ACTION_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"actionSeqNum:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/ACTION_SEQ_NUM/sim_a405a8cd_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_10000_tbBurst_2000_rep_0.vec


echo "SWITCH_ID/"
scavetool x --type v --filter "module(**.relayUnit) AND \"switchId:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SWITCH_ID/sim_a405a8cd_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-R probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_10000_tbBurst_2000_rep_0.vec


echo "SWITCH_ID_ACTION/"
scavetool x --type v --filter "module(**.relayUnit) AND \"switchIdAction:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SWITCH_ID_ACTION/sim_a405a8cd_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-R probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_10000_tbBurst_2000_rep_0.vec


echo "INTERFACE_ID/"
scavetool x --type v --filter "module(**.relayUnit) AND \"interfaceId:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/INTERFACE_ID/sim_a405a8cd_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-R probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_10000_tbBurst_2000_rep_0.vec


echo "REQUESTER_ID/"
scavetool x --type v --filter "module(**.**.relayUnit) AND \"RequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/REQUESTER_ID/sim_a405a8cd_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_10000_tbBurst_2000_rep_0.vec


echo "FLOW_STARTED/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"flowStartedRequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_STARTED/sim_a405a8cd_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_10000_tbBurst_2000_rep_0.vec


echo "FLOW_ENDED/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"flowEndedRequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_ENDED/sim_a405a8cd_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_10000_tbBurst_2000_rep_0.vec


echo "REQUEST_SENT/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"requestSentRequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/REQUEST_SENT/sim_a405a8cd_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-R probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_10000_tbBurst_2000_rep_0.vec


echo "REPLY_LENGTH_ASKED/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"replyLengthAsked:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/REPLY_LENGTH_ASKED/sim_a405a8cd_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-R probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_10000_tbBurst_2000_rep_0.vec


echo "FLOW_ID/"
scavetool x --type v --filter "module(**.**.relayUnit) AND \"FlowID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_ID/sim_a405a8cd_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-R probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_10000_tbBurst_2000_rep_0.vec


echo "PACKET_SIZE/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"PacketSize:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/PACKET_SIZE/sim_a405a8cd_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_10000_tbBurst_2000_rep_0.vec
echo "sim_77c20184_beta0.15_theta0.75"
echo "QUEUE_LEN/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"QueueLen:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUE_LEN/sim_77c20184_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_100_tbBurst_100_rep_0.vec


echo "QUEUES_TOT_LEN/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"QueuesTotLen:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUES_TOT_LEN/sim_77c20184_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_100_tbBurst_100_rep_0.vec


echo "QUEUE_CAPACITY/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"QueueCapacity:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUE_CAPACITY/sim_77c20184_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_100_tbBurst_100_rep_0.vec


echo "QUEUES_TOT_CAPACITY/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"QueuesTotCapacity:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUES_TOT_CAPACITY/sim_77c20184_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_100_tbBurst_100_rep_0.vec


echo "PACKET_ACTION/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"PacketAction:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/PACKET_ACTION/sim_77c20184_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_100_tbBurst_100_rep_0.vec


echo "RCV_TS_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"rcvSeq:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/RCV_TS_SEQ_NUM/sim_77c20184_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_100_tbBurst_100_rep_0.vec


echo "SND_TS_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"sndNxt:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SND_TS_SEQ_NUM/sim_77c20184_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_100_tbBurst_100_rep_0.vec


echo "OOO_SEG/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"rcvOooSeg:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/OOO_SEG/sim_77c20184_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_100_tbBurst_100_rep_0.vec


echo "SWITCH_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"switchSeqNum:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SWITCH_SEQ_NUM/sim_77c20184_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_100_tbBurst_100_rep_0.vec


echo "RETRANSMITTED/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"retransmitted:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/RETRANSMITTED/sim_77c20184_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_100_tbBurst_100_rep_0.vec


echo "TTL/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"switchTtl:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/TTL/sim_77c20184_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_100_tbBurst_100_rep_0.vec


echo "ACTION_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"actionSeqNum:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/ACTION_SEQ_NUM/sim_77c20184_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_100_tbBurst_100_rep_0.vec


echo "SWITCH_ID/"
scavetool x --type v --filter "module(**.relayUnit) AND \"switchId:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SWITCH_ID/sim_77c20184_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-R probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_100_tbBurst_100_rep_0.vec


echo "SWITCH_ID_ACTION/"
scavetool x --type v --filter "module(**.relayUnit) AND \"switchIdAction:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SWITCH_ID_ACTION/sim_77c20184_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-R probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_100_tbBurst_100_rep_0.vec


echo "INTERFACE_ID/"
scavetool x --type v --filter "module(**.relayUnit) AND \"interfaceId:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/INTERFACE_ID/sim_77c20184_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-R probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_100_tbBurst_100_rep_0.vec


echo "REQUESTER_ID/"
scavetool x --type v --filter "module(**.**.relayUnit) AND \"RequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/REQUESTER_ID/sim_77c20184_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_100_tbBurst_100_rep_0.vec


echo "FLOW_STARTED/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"flowStartedRequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_STARTED/sim_77c20184_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_100_tbBurst_100_rep_0.vec


echo "FLOW_ENDED/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"flowEndedRequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_ENDED/sim_77c20184_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_100_tbBurst_100_rep_0.vec


echo "REQUEST_SENT/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"requestSentRequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/REQUEST_SENT/sim_77c20184_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-R probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_100_tbBurst_100_rep_0.vec


echo "REPLY_LENGTH_ASKED/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"replyLengthAsked:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/REPLY_LENGTH_ASKED/sim_77c20184_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-R probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_100_tbBurst_100_rep_0.vec


echo "FLOW_ID/"
scavetool x --type v --filter "module(**.**.relayUnit) AND \"FlowID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_ID/sim_77c20184_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-R probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_100_tbBurst_100_rep_0.vec


echo "PACKET_SIZE/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"PacketSize:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/PACKET_SIZE/sim_77c20184_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_100_tbBurst_100_rep_0.vec
echo "sim_6b15cac1_beta0.15_theta0.75"
echo "QUEUE_LEN/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"QueueLen:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUE_LEN/sim_6b15cac1_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_10000_tbBurst_2000_rep_0.vec


echo "QUEUES_TOT_LEN/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"QueuesTotLen:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUES_TOT_LEN/sim_6b15cac1_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_10000_tbBurst_2000_rep_0.vec


echo "QUEUE_CAPACITY/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"QueueCapacity:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUE_CAPACITY/sim_6b15cac1_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_10000_tbBurst_2000_rep_0.vec


echo "QUEUES_TOT_CAPACITY/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"QueuesTotCapacity:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUES_TOT_CAPACITY/sim_6b15cac1_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_10000_tbBurst_2000_rep_0.vec


echo "PACKET_ACTION/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"PacketAction:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/PACKET_ACTION/sim_6b15cac1_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_10000_tbBurst_2000_rep_0.vec


echo "RCV_TS_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"rcvSeq:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/RCV_TS_SEQ_NUM/sim_6b15cac1_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_10000_tbBurst_2000_rep_0.vec


echo "SND_TS_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"sndNxt:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SND_TS_SEQ_NUM/sim_6b15cac1_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_10000_tbBurst_2000_rep_0.vec


echo "OOO_SEG/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"rcvOooSeg:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/OOO_SEG/sim_6b15cac1_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_10000_tbBurst_2000_rep_0.vec


echo "SWITCH_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"switchSeqNum:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SWITCH_SEQ_NUM/sim_6b15cac1_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_10000_tbBurst_2000_rep_0.vec


echo "RETRANSMITTED/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"retransmitted:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/RETRANSMITTED/sim_6b15cac1_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_10000_tbBurst_2000_rep_0.vec


echo "TTL/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"switchTtl:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/TTL/sim_6b15cac1_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_10000_tbBurst_2000_rep_0.vec


echo "ACTION_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"actionSeqNum:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/ACTION_SEQ_NUM/sim_6b15cac1_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_10000_tbBurst_2000_rep_0.vec


echo "SWITCH_ID/"
scavetool x --type v --filter "module(**.relayUnit) AND \"switchId:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SWITCH_ID/sim_6b15cac1_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-R probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_10000_tbBurst_2000_rep_0.vec


echo "SWITCH_ID_ACTION/"
scavetool x --type v --filter "module(**.relayUnit) AND \"switchIdAction:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SWITCH_ID_ACTION/sim_6b15cac1_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-R probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_10000_tbBurst_2000_rep_0.vec


echo "INTERFACE_ID/"
scavetool x --type v --filter "module(**.relayUnit) AND \"interfaceId:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/INTERFACE_ID/sim_6b15cac1_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-R probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_10000_tbBurst_2000_rep_0.vec


echo "REQUESTER_ID/"
scavetool x --type v --filter "module(**.**.relayUnit) AND \"RequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/REQUESTER_ID/sim_6b15cac1_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_10000_tbBurst_2000_rep_0.vec


echo "FLOW_STARTED/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"flowStartedRequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_STARTED/sim_6b15cac1_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_10000_tbBurst_2000_rep_0.vec


echo "FLOW_ENDED/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"flowEndedRequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_ENDED/sim_6b15cac1_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_10000_tbBurst_2000_rep_0.vec


echo "REQUEST_SENT/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"requestSentRequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/REQUEST_SENT/sim_6b15cac1_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-R probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_10000_tbBurst_2000_rep_0.vec


echo "REPLY_LENGTH_ASKED/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"replyLengthAsked:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/REPLY_LENGTH_ASKED/sim_6b15cac1_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-R probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_10000_tbBurst_2000_rep_0.vec


echo "FLOW_ID/"
scavetool x --type v --filter "module(**.**.relayUnit) AND \"FlowID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_ID/sim_6b15cac1_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-R probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_10000_tbBurst_2000_rep_0.vec


echo "PACKET_SIZE/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"PacketSize:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/PACKET_SIZE/sim_6b15cac1_beta0.15_theta0.75_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.75_beta_0.15_tbRate_10000_tbBurst_2000_rep_0.vec
echo "sim_bb6aeba3_beta0.05_theta0.7"
echo "QUEUE_LEN/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"QueueLen:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUE_LEN/sim_bb6aeba3_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_5000_tbBurst_2000_rep_0.vec


echo "QUEUES_TOT_LEN/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"QueuesTotLen:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUES_TOT_LEN/sim_bb6aeba3_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_5000_tbBurst_2000_rep_0.vec


echo "QUEUE_CAPACITY/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"QueueCapacity:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUE_CAPACITY/sim_bb6aeba3_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_5000_tbBurst_2000_rep_0.vec


echo "QUEUES_TOT_CAPACITY/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"QueuesTotCapacity:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUES_TOT_CAPACITY/sim_bb6aeba3_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_5000_tbBurst_2000_rep_0.vec


echo "PACKET_ACTION/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"PacketAction:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/PACKET_ACTION/sim_bb6aeba3_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_5000_tbBurst_2000_rep_0.vec


echo "RCV_TS_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"rcvSeq:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/RCV_TS_SEQ_NUM/sim_bb6aeba3_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_5000_tbBurst_2000_rep_0.vec


echo "SND_TS_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"sndNxt:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SND_TS_SEQ_NUM/sim_bb6aeba3_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_5000_tbBurst_2000_rep_0.vec


echo "OOO_SEG/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"rcvOooSeg:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/OOO_SEG/sim_bb6aeba3_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_5000_tbBurst_2000_rep_0.vec


echo "SWITCH_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"switchSeqNum:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SWITCH_SEQ_NUM/sim_bb6aeba3_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_5000_tbBurst_2000_rep_0.vec


echo "RETRANSMITTED/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"retransmitted:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/RETRANSMITTED/sim_bb6aeba3_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_5000_tbBurst_2000_rep_0.vec


echo "TTL/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"switchTtl:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/TTL/sim_bb6aeba3_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_5000_tbBurst_2000_rep_0.vec


echo "ACTION_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"actionSeqNum:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/ACTION_SEQ_NUM/sim_bb6aeba3_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_5000_tbBurst_2000_rep_0.vec


echo "SWITCH_ID/"
scavetool x --type v --filter "module(**.relayUnit) AND \"switchId:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SWITCH_ID/sim_bb6aeba3_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-R probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_5000_tbBurst_2000_rep_0.vec


echo "SWITCH_ID_ACTION/"
scavetool x --type v --filter "module(**.relayUnit) AND \"switchIdAction:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SWITCH_ID_ACTION/sim_bb6aeba3_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-R probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_5000_tbBurst_2000_rep_0.vec


echo "INTERFACE_ID/"
scavetool x --type v --filter "module(**.relayUnit) AND \"interfaceId:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/INTERFACE_ID/sim_bb6aeba3_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-R probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_5000_tbBurst_2000_rep_0.vec


echo "REQUESTER_ID/"
scavetool x --type v --filter "module(**.**.relayUnit) AND \"RequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/REQUESTER_ID/sim_bb6aeba3_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_5000_tbBurst_2000_rep_0.vec


echo "FLOW_STARTED/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"flowStartedRequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_STARTED/sim_bb6aeba3_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_5000_tbBurst_2000_rep_0.vec


echo "FLOW_ENDED/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"flowEndedRequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_ENDED/sim_bb6aeba3_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_5000_tbBurst_2000_rep_0.vec


echo "REQUEST_SENT/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"requestSentRequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/REQUEST_SENT/sim_bb6aeba3_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-R probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_5000_tbBurst_2000_rep_0.vec


echo "REPLY_LENGTH_ASKED/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"replyLengthAsked:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/REPLY_LENGTH_ASKED/sim_bb6aeba3_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-R probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_5000_tbBurst_2000_rep_0.vec


echo "FLOW_ID/"
scavetool x --type v --filter "module(**.**.relayUnit) AND \"FlowID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_ID/sim_bb6aeba3_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-R probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_5000_tbBurst_2000_rep_0.vec


echo "PACKET_SIZE/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"PacketSize:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/PACKET_SIZE/sim_bb6aeba3_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_5000_tbBurst_2000_rep_0.vec
echo "sim_998ea558_beta0.05_theta0.7"
echo "QUEUE_LEN/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"QueueLen:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUE_LEN/sim_998ea558_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_5000_tbBurst_1000_rep_0.vec


echo "QUEUES_TOT_LEN/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"QueuesTotLen:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUES_TOT_LEN/sim_998ea558_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_5000_tbBurst_1000_rep_0.vec


echo "QUEUE_CAPACITY/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"QueueCapacity:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUE_CAPACITY/sim_998ea558_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_5000_tbBurst_1000_rep_0.vec


echo "QUEUES_TOT_CAPACITY/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"QueuesTotCapacity:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUES_TOT_CAPACITY/sim_998ea558_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_5000_tbBurst_1000_rep_0.vec


echo "PACKET_ACTION/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"PacketAction:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/PACKET_ACTION/sim_998ea558_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_5000_tbBurst_1000_rep_0.vec


echo "RCV_TS_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"rcvSeq:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/RCV_TS_SEQ_NUM/sim_998ea558_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_5000_tbBurst_1000_rep_0.vec


echo "SND_TS_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"sndNxt:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SND_TS_SEQ_NUM/sim_998ea558_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_5000_tbBurst_1000_rep_0.vec


echo "OOO_SEG/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"rcvOooSeg:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/OOO_SEG/sim_998ea558_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_5000_tbBurst_1000_rep_0.vec


echo "SWITCH_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"switchSeqNum:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SWITCH_SEQ_NUM/sim_998ea558_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_5000_tbBurst_1000_rep_0.vec


echo "RETRANSMITTED/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"retransmitted:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/RETRANSMITTED/sim_998ea558_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_5000_tbBurst_1000_rep_0.vec


echo "TTL/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"switchTtl:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/TTL/sim_998ea558_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_5000_tbBurst_1000_rep_0.vec


echo "ACTION_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"actionSeqNum:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/ACTION_SEQ_NUM/sim_998ea558_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_5000_tbBurst_1000_rep_0.vec


echo "SWITCH_ID/"
scavetool x --type v --filter "module(**.relayUnit) AND \"switchId:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SWITCH_ID/sim_998ea558_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-R probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_5000_tbBurst_1000_rep_0.vec


echo "SWITCH_ID_ACTION/"
scavetool x --type v --filter "module(**.relayUnit) AND \"switchIdAction:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SWITCH_ID_ACTION/sim_998ea558_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-R probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_5000_tbBurst_1000_rep_0.vec


echo "INTERFACE_ID/"
scavetool x --type v --filter "module(**.relayUnit) AND \"interfaceId:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/INTERFACE_ID/sim_998ea558_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-R probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_5000_tbBurst_1000_rep_0.vec


echo "REQUESTER_ID/"
scavetool x --type v --filter "module(**.**.relayUnit) AND \"RequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/REQUESTER_ID/sim_998ea558_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_5000_tbBurst_1000_rep_0.vec


echo "FLOW_STARTED/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"flowStartedRequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_STARTED/sim_998ea558_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_5000_tbBurst_1000_rep_0.vec


echo "FLOW_ENDED/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"flowEndedRequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_ENDED/sim_998ea558_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_5000_tbBurst_1000_rep_0.vec


echo "REQUEST_SENT/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"requestSentRequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/REQUEST_SENT/sim_998ea558_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-R probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_5000_tbBurst_1000_rep_0.vec


echo "REPLY_LENGTH_ASKED/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"replyLengthAsked:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/REPLY_LENGTH_ASKED/sim_998ea558_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-R probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_5000_tbBurst_1000_rep_0.vec


echo "FLOW_ID/"
scavetool x --type v --filter "module(**.**.relayUnit) AND \"FlowID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_ID/sim_998ea558_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-R probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_5000_tbBurst_1000_rep_0.vec


echo "PACKET_SIZE/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"PacketSize:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/PACKET_SIZE/sim_998ea558_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_5000_tbBurst_1000_rep_0.vec
echo "sim_ed73b9c6_beta0.05_theta0.7"
echo "QUEUE_LEN/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"QueueLen:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUE_LEN/sim_ed73b9c6_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_10000_tbBurst_1000_rep_0.vec


echo "QUEUES_TOT_LEN/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"QueuesTotLen:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUES_TOT_LEN/sim_ed73b9c6_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_10000_tbBurst_1000_rep_0.vec


echo "QUEUE_CAPACITY/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"QueueCapacity:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUE_CAPACITY/sim_ed73b9c6_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_10000_tbBurst_1000_rep_0.vec


echo "QUEUES_TOT_CAPACITY/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"QueuesTotCapacity:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUES_TOT_CAPACITY/sim_ed73b9c6_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_10000_tbBurst_1000_rep_0.vec


echo "PACKET_ACTION/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"PacketAction:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/PACKET_ACTION/sim_ed73b9c6_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_10000_tbBurst_1000_rep_0.vec


echo "RCV_TS_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"rcvSeq:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/RCV_TS_SEQ_NUM/sim_ed73b9c6_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_10000_tbBurst_1000_rep_0.vec


echo "SND_TS_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"sndNxt:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SND_TS_SEQ_NUM/sim_ed73b9c6_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_10000_tbBurst_1000_rep_0.vec


echo "OOO_SEG/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"rcvOooSeg:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/OOO_SEG/sim_ed73b9c6_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_10000_tbBurst_1000_rep_0.vec


echo "SWITCH_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"switchSeqNum:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SWITCH_SEQ_NUM/sim_ed73b9c6_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_10000_tbBurst_1000_rep_0.vec


echo "RETRANSMITTED/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"retransmitted:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/RETRANSMITTED/sim_ed73b9c6_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_10000_tbBurst_1000_rep_0.vec


echo "TTL/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"switchTtl:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/TTL/sim_ed73b9c6_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_10000_tbBurst_1000_rep_0.vec


echo "ACTION_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"actionSeqNum:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/ACTION_SEQ_NUM/sim_ed73b9c6_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_10000_tbBurst_1000_rep_0.vec


echo "SWITCH_ID/"
scavetool x --type v --filter "module(**.relayUnit) AND \"switchId:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SWITCH_ID/sim_ed73b9c6_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-R probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_10000_tbBurst_1000_rep_0.vec


echo "SWITCH_ID_ACTION/"
scavetool x --type v --filter "module(**.relayUnit) AND \"switchIdAction:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SWITCH_ID_ACTION/sim_ed73b9c6_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-R probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_10000_tbBurst_1000_rep_0.vec


echo "INTERFACE_ID/"
scavetool x --type v --filter "module(**.relayUnit) AND \"interfaceId:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/INTERFACE_ID/sim_ed73b9c6_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-R probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_10000_tbBurst_1000_rep_0.vec


echo "REQUESTER_ID/"
scavetool x --type v --filter "module(**.**.relayUnit) AND \"RequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/REQUESTER_ID/sim_ed73b9c6_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_10000_tbBurst_1000_rep_0.vec


echo "FLOW_STARTED/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"flowStartedRequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_STARTED/sim_ed73b9c6_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_10000_tbBurst_1000_rep_0.vec


echo "FLOW_ENDED/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"flowEndedRequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_ENDED/sim_ed73b9c6_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_10000_tbBurst_1000_rep_0.vec


echo "REQUEST_SENT/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"requestSentRequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/REQUEST_SENT/sim_ed73b9c6_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-R probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_10000_tbBurst_1000_rep_0.vec


echo "REPLY_LENGTH_ASKED/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"replyLengthAsked:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/REPLY_LENGTH_ASKED/sim_ed73b9c6_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-R probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_10000_tbBurst_1000_rep_0.vec


echo "FLOW_ID/"
scavetool x --type v --filter "module(**.**.relayUnit) AND \"FlowID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_ID/sim_ed73b9c6_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-R probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_10000_tbBurst_1000_rep_0.vec


echo "PACKET_SIZE/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"PacketSize:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/PACKET_SIZE/sim_ed73b9c6_beta0.05_theta0.7_probabilistic_tb.csv -F CSV-S probabilistic_tb/prob_theta_0.7_beta_0.05_tbRate_10000_tbBurst_1000_rep_0.vec
