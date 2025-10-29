#!/bin/bash
# Auto-generated extractor script

echo "sim_f9567cb9_eps0.05"
echo "QUEUE_LEN/"
scavetool x --type v --filter "(module(**.agg[*].relayUnit) OR module(**.spine[*].relayUnit)) AND name(QueueLen:vector)" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUE_LEN/sim_f9567cb9_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_64_tbBurst_12000_rep_0.vec


echo "QUEUES_TOT_LEN/"
scavetool x --type v --filter "(module(**.agg[*].relayUnit) OR module(**.spine[*].relayUnit)) AND name(QueuesTotLen:vector)" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUES_TOT_LEN/sim_f9567cb9_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_64_tbBurst_12000_rep_0.vec


echo "QUEUE_CAPACITY/"
scavetool x --type v --filter "(module(**.agg[*].relayUnit) OR module(**.spine[*].relayUnit)) AND name(QueueCapacity:vector)" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUE_CAPACITY/sim_f9567cb9_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_64_tbBurst_12000_rep_0.vec


echo "QUEUES_TOT_CAPACITY/"
scavetool x --type v --filter "(module(**.agg[*].relayUnit) OR module(**.spine[*].relayUnit)) AND name(QueuesTotCapacity:vector)" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUES_TOT_CAPACITY/sim_f9567cb9_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_64_tbBurst_12000_rep_0.vec


echo "PACKET_ACTION/"
scavetool x --type v --filter "(module(**.agg[*].relayUnit) OR module(**.spine[*].relayUnit)) AND name(PacketAction:vector)" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/PACKET_ACTION/sim_f9567cb9_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_64_tbBurst_12000_rep_0.vec


echo "FLOW_ENDED_QUERY_ID/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"flowEndedQueryID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_ENDED_QUERY_ID/sim_f9567cb9_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_64_tbBurst_12000_rep_0.vec


echo "RCV_TS_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"rcvSeq:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/RCV_TS_SEQ_NUM/sim_f9567cb9_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_64_tbBurst_12000_rep_0.vec


echo "SND_TS_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"sndNxt:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SND_TS_SEQ_NUM/sim_f9567cb9_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_64_tbBurst_12000_rep_0.vec


echo "OOO_SEG/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"rcvOooSeg:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/OOO_SEG/sim_f9567cb9_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_64_tbBurst_12000_rep_0.vec


echo "SWITCH_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"switchSeqNum:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SWITCH_SEQ_NUM/sim_f9567cb9_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_64_tbBurst_12000_rep_0.vec


echo "RETRANSMITTED/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"retransmitted:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/RETRANSMITTED/sim_f9567cb9_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_64_tbBurst_12000_rep_0.vec


echo "TTL/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"switchTtl:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/TTL/sim_f9567cb9_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_64_tbBurst_12000_rep_0.vec


echo "ACTION_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"actionSeqNum:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/ACTION_SEQ_NUM/sim_f9567cb9_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_64_tbBurst_12000_rep_0.vec


echo "SWITCH_ID/"
scavetool x --type v --filter "module(**.relayUnit) AND \"switchId:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SWITCH_ID/sim_f9567cb9_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_64_tbBurst_12000_rep_0.vec


echo "SWITCH_ID_ACTION/"
scavetool x --type v --filter "module(**.relayUnit) AND \"switchIdAction:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SWITCH_ID_ACTION/sim_f9567cb9_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_64_tbBurst_12000_rep_0.vec


echo "INTERFACE_ID/"
scavetool x --type v --filter "module(**.relayUnit) AND \"interfaceId:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/INTERFACE_ID/sim_f9567cb9_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_64_tbBurst_12000_rep_0.vec


echo "REQUESTER_ID/"
scavetool x --type v --filter "module(**.**.relayUnit) AND \"RequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/REQUESTER_ID/sim_f9567cb9_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_64_tbBurst_12000_rep_0.vec


echo "FLOW_STARTED/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"flowStartedRequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_STARTED/sim_f9567cb9_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_64_tbBurst_12000_rep_0.vec


echo "FLOW_ENDED/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"flowEndedRequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_ENDED/sim_f9567cb9_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_64_tbBurst_12000_rep_0.vec


echo "REQUEST_SENT/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"requestSentRequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/REQUEST_SENT/sim_f9567cb9_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_64_tbBurst_12000_rep_0.vec


echo "REPLY_LENGTH_ASKED/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"replyLengthAsked:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/REPLY_LENGTH_ASKED/sim_f9567cb9_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_64_tbBurst_12000_rep_0.vec


echo "FLOW_ID/"
scavetool x --type v --filter "module(**.**.relayUnit) AND \"FlowID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_ID/sim_f9567cb9_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_64_tbBurst_12000_rep_0.vec


echo "PACKET_SIZE/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"PacketSize:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/PACKET_SIZE/sim_f9567cb9_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_64_tbBurst_12000_rep_0.vec
echo "sim_eacbd704_eps0.2"
echo "QUEUE_LEN/"
scavetool x --type v --filter "(module(**.agg[*].relayUnit) OR module(**.spine[*].relayUnit)) AND name(QueueLen:vector)" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUE_LEN/sim_eacbd704_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_16_tbBurst_3000_rep_0.vec


echo "QUEUES_TOT_LEN/"
scavetool x --type v --filter "(module(**.agg[*].relayUnit) OR module(**.spine[*].relayUnit)) AND name(QueuesTotLen:vector)" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUES_TOT_LEN/sim_eacbd704_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_16_tbBurst_3000_rep_0.vec


echo "QUEUE_CAPACITY/"
scavetool x --type v --filter "(module(**.agg[*].relayUnit) OR module(**.spine[*].relayUnit)) AND name(QueueCapacity:vector)" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUE_CAPACITY/sim_eacbd704_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_16_tbBurst_3000_rep_0.vec


echo "QUEUES_TOT_CAPACITY/"
scavetool x --type v --filter "(module(**.agg[*].relayUnit) OR module(**.spine[*].relayUnit)) AND name(QueuesTotCapacity:vector)" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUES_TOT_CAPACITY/sim_eacbd704_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_16_tbBurst_3000_rep_0.vec


echo "PACKET_ACTION/"
scavetool x --type v --filter "(module(**.agg[*].relayUnit) OR module(**.spine[*].relayUnit)) AND name(PacketAction:vector)" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/PACKET_ACTION/sim_eacbd704_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_16_tbBurst_3000_rep_0.vec


echo "FLOW_ENDED_QUERY_ID/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"flowEndedQueryID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_ENDED_QUERY_ID/sim_eacbd704_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_16_tbBurst_3000_rep_0.vec


echo "RCV_TS_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"rcvSeq:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/RCV_TS_SEQ_NUM/sim_eacbd704_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_16_tbBurst_3000_rep_0.vec


echo "SND_TS_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"sndNxt:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SND_TS_SEQ_NUM/sim_eacbd704_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_16_tbBurst_3000_rep_0.vec


echo "OOO_SEG/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"rcvOooSeg:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/OOO_SEG/sim_eacbd704_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_16_tbBurst_3000_rep_0.vec


echo "SWITCH_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"switchSeqNum:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SWITCH_SEQ_NUM/sim_eacbd704_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_16_tbBurst_3000_rep_0.vec


echo "RETRANSMITTED/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"retransmitted:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/RETRANSMITTED/sim_eacbd704_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_16_tbBurst_3000_rep_0.vec


echo "TTL/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"switchTtl:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/TTL/sim_eacbd704_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_16_tbBurst_3000_rep_0.vec


echo "ACTION_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"actionSeqNum:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/ACTION_SEQ_NUM/sim_eacbd704_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_16_tbBurst_3000_rep_0.vec


echo "SWITCH_ID/"
scavetool x --type v --filter "module(**.relayUnit) AND \"switchId:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SWITCH_ID/sim_eacbd704_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_16_tbBurst_3000_rep_0.vec


echo "SWITCH_ID_ACTION/"
scavetool x --type v --filter "module(**.relayUnit) AND \"switchIdAction:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SWITCH_ID_ACTION/sim_eacbd704_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_16_tbBurst_3000_rep_0.vec


echo "INTERFACE_ID/"
scavetool x --type v --filter "module(**.relayUnit) AND \"interfaceId:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/INTERFACE_ID/sim_eacbd704_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_16_tbBurst_3000_rep_0.vec


echo "REQUESTER_ID/"
scavetool x --type v --filter "module(**.**.relayUnit) AND \"RequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/REQUESTER_ID/sim_eacbd704_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_16_tbBurst_3000_rep_0.vec


echo "FLOW_STARTED/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"flowStartedRequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_STARTED/sim_eacbd704_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_16_tbBurst_3000_rep_0.vec


echo "FLOW_ENDED/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"flowEndedRequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_ENDED/sim_eacbd704_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_16_tbBurst_3000_rep_0.vec


echo "REQUEST_SENT/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"requestSentRequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/REQUEST_SENT/sim_eacbd704_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_16_tbBurst_3000_rep_0.vec


echo "REPLY_LENGTH_ASKED/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"replyLengthAsked:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/REPLY_LENGTH_ASKED/sim_eacbd704_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_16_tbBurst_3000_rep_0.vec


echo "FLOW_ID/"
scavetool x --type v --filter "module(**.**.relayUnit) AND \"FlowID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_ID/sim_eacbd704_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_16_tbBurst_3000_rep_0.vec


echo "PACKET_SIZE/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"PacketSize:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/PACKET_SIZE/sim_eacbd704_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_16_tbBurst_3000_rep_0.vec
echo "sim_4881146d_eps0.1"
echo "QUEUE_LEN/"
scavetool x --type v --filter "(module(**.agg[*].relayUnit) OR module(**.spine[*].relayUnit)) AND name(QueueLen:vector)" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUE_LEN/sim_4881146d_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_32_tbBurst_6000_rep_0.vec


echo "QUEUES_TOT_LEN/"
scavetool x --type v --filter "(module(**.agg[*].relayUnit) OR module(**.spine[*].relayUnit)) AND name(QueuesTotLen:vector)" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUES_TOT_LEN/sim_4881146d_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_32_tbBurst_6000_rep_0.vec


echo "QUEUE_CAPACITY/"
scavetool x --type v --filter "(module(**.agg[*].relayUnit) OR module(**.spine[*].relayUnit)) AND name(QueueCapacity:vector)" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUE_CAPACITY/sim_4881146d_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_32_tbBurst_6000_rep_0.vec


echo "QUEUES_TOT_CAPACITY/"
scavetool x --type v --filter "(module(**.agg[*].relayUnit) OR module(**.spine[*].relayUnit)) AND name(QueuesTotCapacity:vector)" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUES_TOT_CAPACITY/sim_4881146d_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_32_tbBurst_6000_rep_0.vec


echo "PACKET_ACTION/"
scavetool x --type v --filter "(module(**.agg[*].relayUnit) OR module(**.spine[*].relayUnit)) AND name(PacketAction:vector)" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/PACKET_ACTION/sim_4881146d_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_32_tbBurst_6000_rep_0.vec


echo "FLOW_ENDED_QUERY_ID/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"flowEndedQueryID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_ENDED_QUERY_ID/sim_4881146d_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_32_tbBurst_6000_rep_0.vec


echo "RCV_TS_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"rcvSeq:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/RCV_TS_SEQ_NUM/sim_4881146d_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_32_tbBurst_6000_rep_0.vec


echo "SND_TS_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"sndNxt:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SND_TS_SEQ_NUM/sim_4881146d_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_32_tbBurst_6000_rep_0.vec


echo "OOO_SEG/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"rcvOooSeg:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/OOO_SEG/sim_4881146d_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_32_tbBurst_6000_rep_0.vec


echo "SWITCH_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"switchSeqNum:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SWITCH_SEQ_NUM/sim_4881146d_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_32_tbBurst_6000_rep_0.vec


echo "RETRANSMITTED/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"retransmitted:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/RETRANSMITTED/sim_4881146d_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_32_tbBurst_6000_rep_0.vec


echo "TTL/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"switchTtl:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/TTL/sim_4881146d_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_32_tbBurst_6000_rep_0.vec


echo "ACTION_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"actionSeqNum:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/ACTION_SEQ_NUM/sim_4881146d_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_32_tbBurst_6000_rep_0.vec


echo "SWITCH_ID/"
scavetool x --type v --filter "module(**.relayUnit) AND \"switchId:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SWITCH_ID/sim_4881146d_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_32_tbBurst_6000_rep_0.vec


echo "SWITCH_ID_ACTION/"
scavetool x --type v --filter "module(**.relayUnit) AND \"switchIdAction:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SWITCH_ID_ACTION/sim_4881146d_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_32_tbBurst_6000_rep_0.vec


echo "INTERFACE_ID/"
scavetool x --type v --filter "module(**.relayUnit) AND \"interfaceId:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/INTERFACE_ID/sim_4881146d_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_32_tbBurst_6000_rep_0.vec


echo "REQUESTER_ID/"
scavetool x --type v --filter "module(**.**.relayUnit) AND \"RequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/REQUESTER_ID/sim_4881146d_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_32_tbBurst_6000_rep_0.vec


echo "FLOW_STARTED/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"flowStartedRequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_STARTED/sim_4881146d_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_32_tbBurst_6000_rep_0.vec


echo "FLOW_ENDED/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"flowEndedRequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_ENDED/sim_4881146d_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_32_tbBurst_6000_rep_0.vec


echo "REQUEST_SENT/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"requestSentRequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/REQUEST_SENT/sim_4881146d_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_32_tbBurst_6000_rep_0.vec


echo "REPLY_LENGTH_ASKED/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"replyLengthAsked:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/REPLY_LENGTH_ASKED/sim_4881146d_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_32_tbBurst_6000_rep_0.vec


echo "FLOW_ID/"
scavetool x --type v --filter "module(**.**.relayUnit) AND \"FlowID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_ID/sim_4881146d_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_32_tbBurst_6000_rep_0.vec


echo "PACKET_SIZE/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"PacketSize:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/PACKET_SIZE/sim_4881146d_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_32_tbBurst_6000_rep_0.vec
echo "sim_66759b7d_eps0.2"
echo "QUEUE_LEN/"
scavetool x --type v --filter "(module(**.agg[*].relayUnit) OR module(**.spine[*].relayUnit)) AND name(QueueLen:vector)" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUE_LEN/sim_66759b7d_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_32_tbBurst_6000_rep_0.vec


echo "QUEUES_TOT_LEN/"
scavetool x --type v --filter "(module(**.agg[*].relayUnit) OR module(**.spine[*].relayUnit)) AND name(QueuesTotLen:vector)" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUES_TOT_LEN/sim_66759b7d_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_32_tbBurst_6000_rep_0.vec


echo "QUEUE_CAPACITY/"
scavetool x --type v --filter "(module(**.agg[*].relayUnit) OR module(**.spine[*].relayUnit)) AND name(QueueCapacity:vector)" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUE_CAPACITY/sim_66759b7d_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_32_tbBurst_6000_rep_0.vec


echo "QUEUES_TOT_CAPACITY/"
scavetool x --type v --filter "(module(**.agg[*].relayUnit) OR module(**.spine[*].relayUnit)) AND name(QueuesTotCapacity:vector)" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUES_TOT_CAPACITY/sim_66759b7d_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_32_tbBurst_6000_rep_0.vec


echo "PACKET_ACTION/"
scavetool x --type v --filter "(module(**.agg[*].relayUnit) OR module(**.spine[*].relayUnit)) AND name(PacketAction:vector)" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/PACKET_ACTION/sim_66759b7d_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_32_tbBurst_6000_rep_0.vec


echo "FLOW_ENDED_QUERY_ID/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"flowEndedQueryID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_ENDED_QUERY_ID/sim_66759b7d_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_32_tbBurst_6000_rep_0.vec


echo "RCV_TS_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"rcvSeq:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/RCV_TS_SEQ_NUM/sim_66759b7d_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_32_tbBurst_6000_rep_0.vec


echo "SND_TS_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"sndNxt:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SND_TS_SEQ_NUM/sim_66759b7d_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_32_tbBurst_6000_rep_0.vec


echo "OOO_SEG/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"rcvOooSeg:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/OOO_SEG/sim_66759b7d_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_32_tbBurst_6000_rep_0.vec


echo "SWITCH_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"switchSeqNum:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SWITCH_SEQ_NUM/sim_66759b7d_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_32_tbBurst_6000_rep_0.vec


echo "RETRANSMITTED/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"retransmitted:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/RETRANSMITTED/sim_66759b7d_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_32_tbBurst_6000_rep_0.vec


echo "TTL/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"switchTtl:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/TTL/sim_66759b7d_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_32_tbBurst_6000_rep_0.vec


echo "ACTION_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"actionSeqNum:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/ACTION_SEQ_NUM/sim_66759b7d_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_32_tbBurst_6000_rep_0.vec


echo "SWITCH_ID/"
scavetool x --type v --filter "module(**.relayUnit) AND \"switchId:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SWITCH_ID/sim_66759b7d_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_32_tbBurst_6000_rep_0.vec


echo "SWITCH_ID_ACTION/"
scavetool x --type v --filter "module(**.relayUnit) AND \"switchIdAction:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SWITCH_ID_ACTION/sim_66759b7d_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_32_tbBurst_6000_rep_0.vec


echo "INTERFACE_ID/"
scavetool x --type v --filter "module(**.relayUnit) AND \"interfaceId:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/INTERFACE_ID/sim_66759b7d_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_32_tbBurst_6000_rep_0.vec


echo "REQUESTER_ID/"
scavetool x --type v --filter "module(**.**.relayUnit) AND \"RequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/REQUESTER_ID/sim_66759b7d_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_32_tbBurst_6000_rep_0.vec


echo "FLOW_STARTED/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"flowStartedRequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_STARTED/sim_66759b7d_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_32_tbBurst_6000_rep_0.vec


echo "FLOW_ENDED/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"flowEndedRequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_ENDED/sim_66759b7d_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_32_tbBurst_6000_rep_0.vec


echo "REQUEST_SENT/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"requestSentRequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/REQUEST_SENT/sim_66759b7d_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_32_tbBurst_6000_rep_0.vec


echo "REPLY_LENGTH_ASKED/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"replyLengthAsked:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/REPLY_LENGTH_ASKED/sim_66759b7d_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_32_tbBurst_6000_rep_0.vec


echo "FLOW_ID/"
scavetool x --type v --filter "module(**.**.relayUnit) AND \"FlowID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_ID/sim_66759b7d_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_32_tbBurst_6000_rep_0.vec


echo "PACKET_SIZE/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"PacketSize:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/PACKET_SIZE/sim_66759b7d_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_32_tbBurst_6000_rep_0.vec
echo "sim_2e95fac9_eps0.1"
echo "QUEUE_LEN/"
scavetool x --type v --filter "(module(**.agg[*].relayUnit) OR module(**.spine[*].relayUnit)) AND name(QueueLen:vector)" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUE_LEN/sim_2e95fac9_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_16_tbBurst_3000_rep_0.vec


echo "QUEUES_TOT_LEN/"
scavetool x --type v --filter "(module(**.agg[*].relayUnit) OR module(**.spine[*].relayUnit)) AND name(QueuesTotLen:vector)" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUES_TOT_LEN/sim_2e95fac9_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_16_tbBurst_3000_rep_0.vec


echo "QUEUE_CAPACITY/"
scavetool x --type v --filter "(module(**.agg[*].relayUnit) OR module(**.spine[*].relayUnit)) AND name(QueueCapacity:vector)" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUE_CAPACITY/sim_2e95fac9_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_16_tbBurst_3000_rep_0.vec


echo "QUEUES_TOT_CAPACITY/"
scavetool x --type v --filter "(module(**.agg[*].relayUnit) OR module(**.spine[*].relayUnit)) AND name(QueuesTotCapacity:vector)" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUES_TOT_CAPACITY/sim_2e95fac9_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_16_tbBurst_3000_rep_0.vec


echo "PACKET_ACTION/"
scavetool x --type v --filter "(module(**.agg[*].relayUnit) OR module(**.spine[*].relayUnit)) AND name(PacketAction:vector)" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/PACKET_ACTION/sim_2e95fac9_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_16_tbBurst_3000_rep_0.vec


echo "FLOW_ENDED_QUERY_ID/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"flowEndedQueryID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_ENDED_QUERY_ID/sim_2e95fac9_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_16_tbBurst_3000_rep_0.vec


echo "RCV_TS_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"rcvSeq:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/RCV_TS_SEQ_NUM/sim_2e95fac9_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_16_tbBurst_3000_rep_0.vec


echo "SND_TS_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"sndNxt:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SND_TS_SEQ_NUM/sim_2e95fac9_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_16_tbBurst_3000_rep_0.vec


echo "OOO_SEG/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"rcvOooSeg:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/OOO_SEG/sim_2e95fac9_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_16_tbBurst_3000_rep_0.vec


echo "SWITCH_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"switchSeqNum:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SWITCH_SEQ_NUM/sim_2e95fac9_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_16_tbBurst_3000_rep_0.vec


echo "RETRANSMITTED/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"retransmitted:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/RETRANSMITTED/sim_2e95fac9_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_16_tbBurst_3000_rep_0.vec


echo "TTL/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"switchTtl:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/TTL/sim_2e95fac9_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_16_tbBurst_3000_rep_0.vec


echo "ACTION_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"actionSeqNum:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/ACTION_SEQ_NUM/sim_2e95fac9_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_16_tbBurst_3000_rep_0.vec


echo "SWITCH_ID/"
scavetool x --type v --filter "module(**.relayUnit) AND \"switchId:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SWITCH_ID/sim_2e95fac9_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_16_tbBurst_3000_rep_0.vec


echo "SWITCH_ID_ACTION/"
scavetool x --type v --filter "module(**.relayUnit) AND \"switchIdAction:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SWITCH_ID_ACTION/sim_2e95fac9_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_16_tbBurst_3000_rep_0.vec


echo "INTERFACE_ID/"
scavetool x --type v --filter "module(**.relayUnit) AND \"interfaceId:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/INTERFACE_ID/sim_2e95fac9_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_16_tbBurst_3000_rep_0.vec


echo "REQUESTER_ID/"
scavetool x --type v --filter "module(**.**.relayUnit) AND \"RequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/REQUESTER_ID/sim_2e95fac9_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_16_tbBurst_3000_rep_0.vec


echo "FLOW_STARTED/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"flowStartedRequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_STARTED/sim_2e95fac9_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_16_tbBurst_3000_rep_0.vec


echo "FLOW_ENDED/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"flowEndedRequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_ENDED/sim_2e95fac9_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_16_tbBurst_3000_rep_0.vec


echo "REQUEST_SENT/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"requestSentRequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/REQUEST_SENT/sim_2e95fac9_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_16_tbBurst_3000_rep_0.vec


echo "REPLY_LENGTH_ASKED/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"replyLengthAsked:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/REPLY_LENGTH_ASKED/sim_2e95fac9_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_16_tbBurst_3000_rep_0.vec


echo "FLOW_ID/"
scavetool x --type v --filter "module(**.**.relayUnit) AND \"FlowID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_ID/sim_2e95fac9_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_16_tbBurst_3000_rep_0.vec


echo "PACKET_SIZE/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"PacketSize:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/PACKET_SIZE/sim_2e95fac9_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_16_tbBurst_3000_rep_0.vec
echo "sim_47f03c17_eps0.05"
echo "QUEUE_LEN/"
scavetool x --type v --filter "(module(**.agg[*].relayUnit) OR module(**.spine[*].relayUnit)) AND name(QueueLen:vector)" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUE_LEN/sim_47f03c17_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_16_tbBurst_3000_rep_0.vec


echo "QUEUES_TOT_LEN/"
scavetool x --type v --filter "(module(**.agg[*].relayUnit) OR module(**.spine[*].relayUnit)) AND name(QueuesTotLen:vector)" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUES_TOT_LEN/sim_47f03c17_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_16_tbBurst_3000_rep_0.vec


echo "QUEUE_CAPACITY/"
scavetool x --type v --filter "(module(**.agg[*].relayUnit) OR module(**.spine[*].relayUnit)) AND name(QueueCapacity:vector)" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUE_CAPACITY/sim_47f03c17_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_16_tbBurst_3000_rep_0.vec


echo "QUEUES_TOT_CAPACITY/"
scavetool x --type v --filter "(module(**.agg[*].relayUnit) OR module(**.spine[*].relayUnit)) AND name(QueuesTotCapacity:vector)" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUES_TOT_CAPACITY/sim_47f03c17_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_16_tbBurst_3000_rep_0.vec


echo "PACKET_ACTION/"
scavetool x --type v --filter "(module(**.agg[*].relayUnit) OR module(**.spine[*].relayUnit)) AND name(PacketAction:vector)" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/PACKET_ACTION/sim_47f03c17_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_16_tbBurst_3000_rep_0.vec


echo "FLOW_ENDED_QUERY_ID/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"flowEndedQueryID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_ENDED_QUERY_ID/sim_47f03c17_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_16_tbBurst_3000_rep_0.vec


echo "RCV_TS_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"rcvSeq:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/RCV_TS_SEQ_NUM/sim_47f03c17_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_16_tbBurst_3000_rep_0.vec


echo "SND_TS_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"sndNxt:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SND_TS_SEQ_NUM/sim_47f03c17_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_16_tbBurst_3000_rep_0.vec


echo "OOO_SEG/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"rcvOooSeg:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/OOO_SEG/sim_47f03c17_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_16_tbBurst_3000_rep_0.vec


echo "SWITCH_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"switchSeqNum:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SWITCH_SEQ_NUM/sim_47f03c17_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_16_tbBurst_3000_rep_0.vec


echo "RETRANSMITTED/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"retransmitted:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/RETRANSMITTED/sim_47f03c17_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_16_tbBurst_3000_rep_0.vec


echo "TTL/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"switchTtl:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/TTL/sim_47f03c17_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_16_tbBurst_3000_rep_0.vec


echo "ACTION_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"actionSeqNum:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/ACTION_SEQ_NUM/sim_47f03c17_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_16_tbBurst_3000_rep_0.vec


echo "SWITCH_ID/"
scavetool x --type v --filter "module(**.relayUnit) AND \"switchId:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SWITCH_ID/sim_47f03c17_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_16_tbBurst_3000_rep_0.vec


echo "SWITCH_ID_ACTION/"
scavetool x --type v --filter "module(**.relayUnit) AND \"switchIdAction:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SWITCH_ID_ACTION/sim_47f03c17_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_16_tbBurst_3000_rep_0.vec


echo "INTERFACE_ID/"
scavetool x --type v --filter "module(**.relayUnit) AND \"interfaceId:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/INTERFACE_ID/sim_47f03c17_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_16_tbBurst_3000_rep_0.vec


echo "REQUESTER_ID/"
scavetool x --type v --filter "module(**.**.relayUnit) AND \"RequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/REQUESTER_ID/sim_47f03c17_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_16_tbBurst_3000_rep_0.vec


echo "FLOW_STARTED/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"flowStartedRequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_STARTED/sim_47f03c17_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_16_tbBurst_3000_rep_0.vec


echo "FLOW_ENDED/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"flowEndedRequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_ENDED/sim_47f03c17_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_16_tbBurst_3000_rep_0.vec


echo "REQUEST_SENT/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"requestSentRequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/REQUEST_SENT/sim_47f03c17_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_16_tbBurst_3000_rep_0.vec


echo "REPLY_LENGTH_ASKED/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"replyLengthAsked:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/REPLY_LENGTH_ASKED/sim_47f03c17_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_16_tbBurst_3000_rep_0.vec


echo "FLOW_ID/"
scavetool x --type v --filter "module(**.**.relayUnit) AND \"FlowID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_ID/sim_47f03c17_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_16_tbBurst_3000_rep_0.vec


echo "PACKET_SIZE/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"PacketSize:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/PACKET_SIZE/sim_47f03c17_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_16_tbBurst_3000_rep_0.vec
echo "sim_97561c2d_eps0.1"
echo "QUEUE_LEN/"
scavetool x --type v --filter "(module(**.agg[*].relayUnit) OR module(**.spine[*].relayUnit)) AND name(QueueLen:vector)" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUE_LEN/sim_97561c2d_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_64_tbBurst_12000_rep_0.vec


echo "QUEUES_TOT_LEN/"
scavetool x --type v --filter "(module(**.agg[*].relayUnit) OR module(**.spine[*].relayUnit)) AND name(QueuesTotLen:vector)" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUES_TOT_LEN/sim_97561c2d_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_64_tbBurst_12000_rep_0.vec


echo "QUEUE_CAPACITY/"
scavetool x --type v --filter "(module(**.agg[*].relayUnit) OR module(**.spine[*].relayUnit)) AND name(QueueCapacity:vector)" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUE_CAPACITY/sim_97561c2d_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_64_tbBurst_12000_rep_0.vec


echo "QUEUES_TOT_CAPACITY/"
scavetool x --type v --filter "(module(**.agg[*].relayUnit) OR module(**.spine[*].relayUnit)) AND name(QueuesTotCapacity:vector)" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUES_TOT_CAPACITY/sim_97561c2d_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_64_tbBurst_12000_rep_0.vec


echo "PACKET_ACTION/"
scavetool x --type v --filter "(module(**.agg[*].relayUnit) OR module(**.spine[*].relayUnit)) AND name(PacketAction:vector)" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/PACKET_ACTION/sim_97561c2d_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_64_tbBurst_12000_rep_0.vec


echo "FLOW_ENDED_QUERY_ID/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"flowEndedQueryID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_ENDED_QUERY_ID/sim_97561c2d_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_64_tbBurst_12000_rep_0.vec


echo "RCV_TS_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"rcvSeq:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/RCV_TS_SEQ_NUM/sim_97561c2d_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_64_tbBurst_12000_rep_0.vec


echo "SND_TS_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"sndNxt:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SND_TS_SEQ_NUM/sim_97561c2d_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_64_tbBurst_12000_rep_0.vec


echo "OOO_SEG/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"rcvOooSeg:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/OOO_SEG/sim_97561c2d_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_64_tbBurst_12000_rep_0.vec


echo "SWITCH_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"switchSeqNum:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SWITCH_SEQ_NUM/sim_97561c2d_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_64_tbBurst_12000_rep_0.vec


echo "RETRANSMITTED/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"retransmitted:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/RETRANSMITTED/sim_97561c2d_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_64_tbBurst_12000_rep_0.vec


echo "TTL/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"switchTtl:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/TTL/sim_97561c2d_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_64_tbBurst_12000_rep_0.vec


echo "ACTION_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"actionSeqNum:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/ACTION_SEQ_NUM/sim_97561c2d_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_64_tbBurst_12000_rep_0.vec


echo "SWITCH_ID/"
scavetool x --type v --filter "module(**.relayUnit) AND \"switchId:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SWITCH_ID/sim_97561c2d_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_64_tbBurst_12000_rep_0.vec


echo "SWITCH_ID_ACTION/"
scavetool x --type v --filter "module(**.relayUnit) AND \"switchIdAction:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SWITCH_ID_ACTION/sim_97561c2d_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_64_tbBurst_12000_rep_0.vec


echo "INTERFACE_ID/"
scavetool x --type v --filter "module(**.relayUnit) AND \"interfaceId:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/INTERFACE_ID/sim_97561c2d_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_64_tbBurst_12000_rep_0.vec


echo "REQUESTER_ID/"
scavetool x --type v --filter "module(**.**.relayUnit) AND \"RequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/REQUESTER_ID/sim_97561c2d_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_64_tbBurst_12000_rep_0.vec


echo "FLOW_STARTED/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"flowStartedRequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_STARTED/sim_97561c2d_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_64_tbBurst_12000_rep_0.vec


echo "FLOW_ENDED/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"flowEndedRequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_ENDED/sim_97561c2d_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_64_tbBurst_12000_rep_0.vec


echo "REQUEST_SENT/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"requestSentRequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/REQUEST_SENT/sim_97561c2d_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_64_tbBurst_12000_rep_0.vec


echo "REPLY_LENGTH_ASKED/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"replyLengthAsked:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/REPLY_LENGTH_ASKED/sim_97561c2d_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_64_tbBurst_12000_rep_0.vec


echo "FLOW_ID/"
scavetool x --type v --filter "module(**.**.relayUnit) AND \"FlowID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_ID/sim_97561c2d_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_64_tbBurst_12000_rep_0.vec


echo "PACKET_SIZE/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"PacketSize:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/PACKET_SIZE/sim_97561c2d_eps0.1_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.1_tbRate_64_tbBurst_12000_rep_0.vec
echo "sim_de6b32e5_eps0.2"
echo "QUEUE_LEN/"
scavetool x --type v --filter "(module(**.agg[*].relayUnit) OR module(**.spine[*].relayUnit)) AND name(QueueLen:vector)" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUE_LEN/sim_de6b32e5_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_64_tbBurst_12000_rep_0.vec


echo "QUEUES_TOT_LEN/"
scavetool x --type v --filter "(module(**.agg[*].relayUnit) OR module(**.spine[*].relayUnit)) AND name(QueuesTotLen:vector)" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUES_TOT_LEN/sim_de6b32e5_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_64_tbBurst_12000_rep_0.vec


echo "QUEUE_CAPACITY/"
scavetool x --type v --filter "(module(**.agg[*].relayUnit) OR module(**.spine[*].relayUnit)) AND name(QueueCapacity:vector)" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUE_CAPACITY/sim_de6b32e5_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_64_tbBurst_12000_rep_0.vec


echo "QUEUES_TOT_CAPACITY/"
scavetool x --type v --filter "(module(**.agg[*].relayUnit) OR module(**.spine[*].relayUnit)) AND name(QueuesTotCapacity:vector)" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUES_TOT_CAPACITY/sim_de6b32e5_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_64_tbBurst_12000_rep_0.vec


echo "PACKET_ACTION/"
scavetool x --type v --filter "(module(**.agg[*].relayUnit) OR module(**.spine[*].relayUnit)) AND name(PacketAction:vector)" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/PACKET_ACTION/sim_de6b32e5_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_64_tbBurst_12000_rep_0.vec


echo "FLOW_ENDED_QUERY_ID/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"flowEndedQueryID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_ENDED_QUERY_ID/sim_de6b32e5_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_64_tbBurst_12000_rep_0.vec


echo "RCV_TS_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"rcvSeq:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/RCV_TS_SEQ_NUM/sim_de6b32e5_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_64_tbBurst_12000_rep_0.vec


echo "SND_TS_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"sndNxt:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SND_TS_SEQ_NUM/sim_de6b32e5_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_64_tbBurst_12000_rep_0.vec


echo "OOO_SEG/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"rcvOooSeg:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/OOO_SEG/sim_de6b32e5_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_64_tbBurst_12000_rep_0.vec


echo "SWITCH_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"switchSeqNum:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SWITCH_SEQ_NUM/sim_de6b32e5_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_64_tbBurst_12000_rep_0.vec


echo "RETRANSMITTED/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"retransmitted:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/RETRANSMITTED/sim_de6b32e5_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_64_tbBurst_12000_rep_0.vec


echo "TTL/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"switchTtl:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/TTL/sim_de6b32e5_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_64_tbBurst_12000_rep_0.vec


echo "ACTION_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"actionSeqNum:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/ACTION_SEQ_NUM/sim_de6b32e5_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_64_tbBurst_12000_rep_0.vec


echo "SWITCH_ID/"
scavetool x --type v --filter "module(**.relayUnit) AND \"switchId:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SWITCH_ID/sim_de6b32e5_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_64_tbBurst_12000_rep_0.vec


echo "SWITCH_ID_ACTION/"
scavetool x --type v --filter "module(**.relayUnit) AND \"switchIdAction:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SWITCH_ID_ACTION/sim_de6b32e5_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_64_tbBurst_12000_rep_0.vec


echo "INTERFACE_ID/"
scavetool x --type v --filter "module(**.relayUnit) AND \"interfaceId:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/INTERFACE_ID/sim_de6b32e5_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_64_tbBurst_12000_rep_0.vec


echo "REQUESTER_ID/"
scavetool x --type v --filter "module(**.**.relayUnit) AND \"RequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/REQUESTER_ID/sim_de6b32e5_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_64_tbBurst_12000_rep_0.vec


echo "FLOW_STARTED/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"flowStartedRequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_STARTED/sim_de6b32e5_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_64_tbBurst_12000_rep_0.vec


echo "FLOW_ENDED/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"flowEndedRequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_ENDED/sim_de6b32e5_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_64_tbBurst_12000_rep_0.vec


echo "REQUEST_SENT/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"requestSentRequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/REQUEST_SENT/sim_de6b32e5_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_64_tbBurst_12000_rep_0.vec


echo "REPLY_LENGTH_ASKED/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"replyLengthAsked:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/REPLY_LENGTH_ASKED/sim_de6b32e5_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_64_tbBurst_12000_rep_0.vec


echo "FLOW_ID/"
scavetool x --type v --filter "module(**.**.relayUnit) AND \"FlowID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_ID/sim_de6b32e5_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_64_tbBurst_12000_rep_0.vec


echo "PACKET_SIZE/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"PacketSize:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/PACKET_SIZE/sim_de6b32e5_eps0.2_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.2_tbRate_64_tbBurst_12000_rep_0.vec
echo "sim_05f8379a_eps0.05"
echo "QUEUE_LEN/"
scavetool x --type v --filter "(module(**.agg[*].relayUnit) OR module(**.spine[*].relayUnit)) AND name(QueueLen:vector)" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUE_LEN/sim_05f8379a_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_32_tbBurst_6000_rep_0.vec


echo "QUEUES_TOT_LEN/"
scavetool x --type v --filter "(module(**.agg[*].relayUnit) OR module(**.spine[*].relayUnit)) AND name(QueuesTotLen:vector)" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUES_TOT_LEN/sim_05f8379a_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_32_tbBurst_6000_rep_0.vec


echo "QUEUE_CAPACITY/"
scavetool x --type v --filter "(module(**.agg[*].relayUnit) OR module(**.spine[*].relayUnit)) AND name(QueueCapacity:vector)" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUE_CAPACITY/sim_05f8379a_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_32_tbBurst_6000_rep_0.vec


echo "QUEUES_TOT_CAPACITY/"
scavetool x --type v --filter "(module(**.agg[*].relayUnit) OR module(**.spine[*].relayUnit)) AND name(QueuesTotCapacity:vector)" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/QUEUES_TOT_CAPACITY/sim_05f8379a_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_32_tbBurst_6000_rep_0.vec


echo "PACKET_ACTION/"
scavetool x --type v --filter "(module(**.agg[*].relayUnit) OR module(**.spine[*].relayUnit)) AND name(PacketAction:vector)" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/PACKET_ACTION/sim_05f8379a_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_32_tbBurst_6000_rep_0.vec


echo "FLOW_ENDED_QUERY_ID/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"flowEndedQueryID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_ENDED_QUERY_ID/sim_05f8379a_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_32_tbBurst_6000_rep_0.vec


echo "RCV_TS_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"rcvSeq:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/RCV_TS_SEQ_NUM/sim_05f8379a_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_32_tbBurst_6000_rep_0.vec


echo "SND_TS_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"sndNxt:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SND_TS_SEQ_NUM/sim_05f8379a_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_32_tbBurst_6000_rep_0.vec


echo "OOO_SEG/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"rcvOooSeg:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/OOO_SEG/sim_05f8379a_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_32_tbBurst_6000_rep_0.vec


echo "SWITCH_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"switchSeqNum:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SWITCH_SEQ_NUM/sim_05f8379a_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_32_tbBurst_6000_rep_0.vec


echo "RETRANSMITTED/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"retransmitted:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/RETRANSMITTED/sim_05f8379a_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_32_tbBurst_6000_rep_0.vec


echo "TTL/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"switchTtl:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/TTL/sim_05f8379a_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_32_tbBurst_6000_rep_0.vec


echo "ACTION_SEQ_NUM/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"actionSeqNum:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/ACTION_SEQ_NUM/sim_05f8379a_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_32_tbBurst_6000_rep_0.vec


echo "SWITCH_ID/"
scavetool x --type v --filter "module(**.relayUnit) AND \"switchId:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SWITCH_ID/sim_05f8379a_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_32_tbBurst_6000_rep_0.vec


echo "SWITCH_ID_ACTION/"
scavetool x --type v --filter "module(**.relayUnit) AND \"switchIdAction:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/SWITCH_ID_ACTION/sim_05f8379a_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_32_tbBurst_6000_rep_0.vec


echo "INTERFACE_ID/"
scavetool x --type v --filter "module(**.relayUnit) AND \"interfaceId:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/INTERFACE_ID/sim_05f8379a_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_32_tbBurst_6000_rep_0.vec


echo "REQUESTER_ID/"
scavetool x --type v --filter "module(**.**.relayUnit) AND \"RequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/REQUESTER_ID/sim_05f8379a_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_32_tbBurst_6000_rep_0.vec


echo "FLOW_STARTED/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"flowStartedRequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_STARTED/sim_05f8379a_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_32_tbBurst_6000_rep_0.vec


echo "FLOW_ENDED/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"flowEndedRequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_ENDED/sim_05f8379a_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_32_tbBurst_6000_rep_0.vec


echo "REQUEST_SENT/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"requestSentRequesterID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/REQUEST_SENT/sim_05f8379a_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_32_tbBurst_6000_rep_0.vec


echo "REPLY_LENGTH_ASKED/"
scavetool x --type v --filter "module(**.server[*].app[*]) AND \"replyLengthAsked:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/REPLY_LENGTH_ASKED/sim_05f8379a_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_32_tbBurst_6000_rep_0.vec


echo "FLOW_ID/"
scavetool x --type v --filter "module(**.**.relayUnit) AND \"FlowID:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/FLOW_ID/sim_05f8379a_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_32_tbBurst_6000_rep_0.vec


echo "PACKET_SIZE/"
scavetool x --type v --filter "module(LeafSpine1G) AND \"PacketSize:vector\"" -o /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/extracted_results/PACKET_SIZE/sim_05f8379a_eps0.05_random_tb.csv -F CSV-R random_tb/uni_random_eps_0.05_tbRate_32_tbBurst_6000_rep_0.vec
