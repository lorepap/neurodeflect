// Copyright (C) 2013 OpenSim Ltd.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program.  If not, see http://www.gnu.org/licenses/.
//
// Author: Benjamin Martin Seregi

#ifndef BOUNCINGIEEE8021DRELAY_H
#define BOUNCINGIEEE8021DRELAY_H

#include "inet/common/INETDefs.h"
#include "inet/common/LayeredProtocolBase.h"
#include "inet/common/lifecycle/ModuleOperations.h"
#include "inet/common/packet/Packet.h"
#include "inet/networklayer/common/InterfaceTable.h"
#include "inet/linklayer/configurator/Ieee8021dInterfaceData.h"
#include "inet/linklayer/ethernet/EtherFrame_m.h"
#include "../LSSwitch/LSMACTable/LSIMacAddressTable.h"
#include "../Augmented_Mac/AugmentedEtherMac.h"
#include "../V2/V2PIFO.h"
#include "../pFabric/pFabric.h"
#include "../V2/buffer/V2PacketBuffer.h"
#include "../V2/V2PIFOBoltQueue.h"
#include <unordered_map>
#include <unordered_set>
#include <deque>
#include <array>
#include <cmath>
#include "omnetpp/simtime_t.h"

// PyTorch includes for RL model inference
#include <torch/torch.h>
#include <torch/script.h>
#include <vector>
#include <random>

using namespace inet;

//
// This module forward frames (~EtherFrame) based on their destination MAC addresses to appropriate ports.
// See the NED definition for details.
//
class BouncingIeee8021dRelay : public LayeredProtocolBase
{
  public:
    BouncingIeee8021dRelay();
    virtual ~BouncingIeee8021dRelay();

    /**
     * Register single MAC address that this switch supports.
     */

    void registerAddress(MacAddress mac);

    /**
     * Register range of MAC addresses that this switch supports.
     */
    void registerAddresses(MacAddress startMac, MacAddress endMac);

    // incremental deployment: public
    int deployed_with_deflection = -1;
    bool can_deflect = false;
    bool incremental_deployment = false;
    bool deflection_graph_partitioned = false;
    cModule *switch_module;
    void read_inc_deflection_properties(std::string incremental_deployment_identifier,
            std::string input_file_name);

  protected:
    MacAddress bridgeAddress;
    IInterfaceTable *ifTable = nullptr;
    LSIMacAddressTable *macTable = nullptr;
    InterfaceEntry *ie = nullptr;
    std::vector<AugmentedEtherMac*> macList;
    bool isStpAware = false;
    std::list<int> port_idx_connected_to_switch_neioghbors;
    int random_power_factor;
    int random_power_bounce_factor;
    // memory for power of N bouncing and forwarding
    bool use_memory;
    int random_power_memory_size;
    int random_power_bounce_memory_size;
    bool capacity_recorded = false; // todo
    /*
     * For forwarding we keep a prefix-based record.
     * Same memory for those prefixes with the same ECMP group
     * We use two unordered maps for this
     * 1) Maps groups (prefixes) to memory lists for that prefix : group_id --> list of previous choices
     * 2) maps IP addresses to groups based on their optional ports: IP_address --> group_id
     * 3) maps group_id_strings to groups: Group string --> group_id
     */
    std::unordered_map<unsigned int, std::list<int>> prefix_to_memory_map;
    std::unordered_map<uint64, unsigned int> mac_addr_to_prefix_map;
    std::unordered_map<std::string, unsigned int> group_to_prefix_map;
    unsigned int prefix_id_counter = 0;
    virtual unsigned int update_memory_hash_maps(MacAddress address, std::list<int> interface_ids);

    /*
     * Vertigo deflects packets to neighboring switches, so for any arbitrary packet
     * with any prefix, the memory is the same
     * so we just need a list that is the size of memory for deflection
     */
    std::list<int> deflection_memory;

    bool use_ecmp, use_power_of_n_lb;
    std::hash<std::string> header_hash;
    //Naive Deflection
    bool bounce_naively;
    int naive_deflection_idx;   // This indicates the index (ranging from 0 - number of neighboring switches) for packet deflection
    //DIBS
    bool bounce_randomly, filter_out_full_ports, approximate_random_deflection;
    // UNIFORM RANDOM (50% coin-flip) deflection mode: choose a neighbor uniformly at random
    // ignoring queue fullness and with a 50% probability of actually deflecting.
    bool bounce_uniform_random;
    // Per-switch RNG for independent randomness
    std::mt19937 rng;
    unsigned int uniform_random_seed_base = 0;
    // Probability to deflect in uniform-random mode (per-decision Bernoulli p)
    double uniform_random_deflect_prob = 0.05;
    //Vertigo
    bool bounce_on_same_path;
    //V2
    bool bounce_randomly_v2;
    bool use_v2_pifo; // V2PIFO is used <--> this should be true
    bool drop_bounced_in_relay; // if this is true, bounced packet get dropped in relay if the chosen queue is full. In other words, you don't do the last step of v2 which is pushing it and dropping the worst packet
    bool send_header_of_dropped_packet_to_receiver; // NDP
    // dctcp
    bool dctcp_mark_deflected_packets_only;
    // bolt
    bool use_bolt, use_bolt_queue, use_bolt_with_vertigo_queue, use_pifo_bolt_reaction;
    bool ignore_cc_thresh_for_deflected_packets;
    uint16 src_enabled_packet_type;
    //pFabric
    bool use_pfabric;
    //vertigo prio queue
    bool use_vertigo_prio_queue;
    //PABO
    bool bounce_probabilistically;
    double utilization_thresh;
    double bounce_probability_lambda;
    // Smooth probabilistic deflection (piecewise-linear or logistic alternative)
    bool bounce_probabilistic_smooth;
    // theta: midpoint queue utilization where p_deflect = 0.5
    double deflect_prob_theta;
    // beta: slope half-width (controls linear ramp width around theta)
    double deflect_prob_beta;
    // selective network feedback
    bool apply_selective_net_reaction;
    int selective_net_reaction_type;
    double sel_reaction_alpha;

    // RL-based deflection policy
    bool bounce_with_rl_policy;         // Enable RL-based deflection decisions
    std::string rl_model_path;          // Path to the trained PyTorch model
    torch::jit::script::Module rl_model;  // Loaded PyTorch model
    bool rl_model_loaded;               // Track if model is successfully loaded

    struct RunningStats {
        uint64_t count = 0;
        double mean = 0.0;
        double m2 = 0.0;
        void add(double x);
        double stddev() const;
    };

    struct RLFlowState {
        bool initialized = false;
        simtime_t firstSeen = SIMTIME_ZERO;
        simtime_t lastSeen = SIMTIME_ZERO;
        uint64_t packetCount = 0;
        uint64_t lastSeq = 0;
        double deflectEma = 0.0;
        double oooEma = 0.0;
        double lastAlpha = 0.0;
        std::deque<std::array<double, 4>> history;
    };

    // RL feature configuration and state
    int rl_history_length = 4;
    int rl_expected_flow_packets = 1000;
    double rl_flow_age_tau_us = 500.0;   // microseconds
    double rl_flow_age_tau_s = 5e-4;     // seconds (derived)
    double rl_ema_half_life_us = 80.0;
    double rl_default_dt_us = 80.0;

    std::unordered_map<size_t, RLFlowState> rl_flow_states;
    std::unordered_map<std::string, RunningStats> rl_queue_util_stats;
    std::unordered_map<std::string, RunningStats> rl_total_util_stats;

    // Deflection statistics
    unsigned long totalPackets = 0;
    unsigned long deflectedPackets = 0;

    // Token-bucket gating (per-egress deflection cap)
    struct TokenBucketState {
        double rate = 0;
        double burst = 0;
        double tokens = 0;
        omnetpp::simtime_t lastRefill = omnetpp::simtime_t(0);
    };
    bool use_token_bucket_gate = false;
    double token_bucket_rate = -1;
    double token_bucket_burst = -1;
    std::vector<TokenBucketState> token_buckets;

    typedef std::pair<MacAddress, MacAddress> MacAddressPair;

    static simsignal_t feedBackPacketDroppedSignal;
    static simsignal_t feedBackPacketDroppedPortSignal;
    static simsignal_t feedBackPacketGeneratedSignal;
    static simsignal_t feedBackPacketGeneratedReqIDSignal;
    static simsignal_t bounceLimitPassedSignal;
    static simsignal_t burstyPacketReceivedSignal;
    static simsignal_t requesterIDSignal; // Nuovo segnale per tracciare il requesterID
    static simsignal_t packetUniqueIDSignal; // Nuovo segnale per tracciare l'ID univoco del pacchetto
    static simsignal_t queueLenSignal;
    static simsignal_t queuesTotLenSignal;
    static simsignal_t queueCapacitySignal;
    static simsignal_t queuesTotCapacitySignal;
    static simsignal_t packetActionSignal;
    static simsignal_t switchSeqNumSignal; // Sequence number for the switch
    static simsignal_t switchTtlSignal; // Time to live for the switch
    static simsignal_t actionSeqNumSignal; // Sequence number for the action taken on the
    static simsignal_t switchIdSignal; // Switch ID for the packet
    static simsignal_t switchIdActionSignal; // Switch ID for the action taken on the packet
    static simsignal_t interfaceIdSignal; // Interface ID for the packet
    static simsignal_t flowIdSignal; // True flow ID (5-tuple without sequence number)
    static simsignal_t packetSizeSignal; // Packet size in bytes
    unsigned long long light_in_relay_packet_drop_counter = 0;

    // Definizione dei valori per relayAction
    enum RelayAction {
        DROP = 0,
        DEFLECT = 1,
        FORWARD = 2
    };

    struct Comp
    {
        bool operator() (const MacAddressPair& first, const MacAddressPair& second) const
        {
            return (first.first < second.first && first.second < second.first);
        }
    };

    bool in_range(const std::set<MacAddressPair, Comp>& ranges, MacAddress value)
    {
        return ranges.find(MacAddressPair(value, value)) != ranges.end();
    }


    std::set<MacAddressPair, Comp> registeredMacAddresses;

    // statistics: see finish() for details.
    int numReceivedNetworkFrames = 0;
    int numDroppedFrames = 0;
    int numReceivedBPDUsFromSTP = 0;
    int numDeliveredBDPUsToSTP = 0;
    int numDispatchedNonBPDUFrames = 0;
    int numDispatchedBDPUFrames = 0;
    bool learn_mac_addresses;

  protected:
    virtual void initialize(int stage) override;
    virtual int numInitStages() const override { return NUM_INIT_STAGES; }

    /**
     * Updates address table (if the port is in learning state)
     * with source address, determines output port
     * and sends out (or broadcasts) frame on ports
     * (if the ports are in forwarding state).
     * Includes calls to updateTableWithAddress() and getPortForAddress().
     *
     */
    void handleAndDispatchFrame(Packet *packet);

    void handleUpperPacket(Packet *packet) override;
    void handleLowerPacket(Packet *packet) override;

    void dispatch(Packet *packet, InterfaceEntry *ie);
    void learn(MacAddress srcAddr, int arrivalInterfaceId);
    void broadcast(Packet *packet, int arrivalInterfaceId);

    void sendUp(Packet *packet);

    //@{ For ECMP
    void chooseDispatchType(Packet *packet, InterfaceEntry *ie);
    unsigned int Compute_CRC16_Simple(std::list<int> bytes, int bytes_size);
    std::list<int> getInfoFlowByteArray(std::string srcAddr, std::string destAddr, int srcPort, int destPort);
    //@}

    //@{ For lifecycle
    virtual void start();
    virtual void stop();
    virtual void handleStartOperation(LifecycleOperation *operation) override { start(); }
    virtual void handleStopOperation(LifecycleOperation *operation) override { stop(); }
    virtual void handleCrashOperation(LifecycleOperation *operation) override { stop(); }
    virtual bool isUpperMessage(cMessage *message) override { return message->arrivedOn("upperLayerIn"); }
    virtual bool isLowerMessage(cMessage *message) override { return message->arrivedOn("ifIn"); }

    virtual bool isInitializeStage(int stage) override { return stage == INITSTAGE_LINK_LAYER; }
    virtual bool isModuleStartStage(int stage) override { return stage == ModuleStartOperation::STAGE_LINK_LAYER; }
    virtual bool isModuleStopStage(int stage) override { return stage == ModuleStopOperation::STAGE_LINK_LAYER; }
    //@}

    /*
     * Gets port data from the InterfaceTable
     */
    Ieee8021dInterfaceData *getPortInterfaceData(unsigned int portNum);

    bool isForwardingInterface(InterfaceEntry *ie);

    /*
     * Returns the first non-loopback interface.
     */
    virtual InterfaceEntry *chooseInterface();
    virtual void finish() override;

    InterfaceEntry* find_interface_to_bounce_naively();
    InterfaceEntry* find_interface_to_bounce_randomly(Packet *packet);
    InterfaceEntry* find_interface_to_bounce_uniform_random(Packet *packet);
    InterfaceEntry* find_interface_to_bounce_on_the_same_path(Packet *packet, InterfaceEntry *original_output_if);
    InterfaceEntry* find_interface_to_fw_randomly_power_of_n(Packet *packet, bool consider_servers);
    void find_interface_to_bounce_randomly_v2(Packet *packet, bool consider_servers, InterfaceEntry *ie2);
    double get_port_utilization(int port, Packet *packet);
    InterfaceEntry* find_a_port_for_packet_towards_source(Packet *packet);
    InterfaceEntry* find_interface_to_bounce_probabilistically(Packet *packet, InterfaceEntry *original_output_if);
    bool compute_smooth_deflection_decision(Packet *packet, InterfaceEntry *original_output_if, double &probability, double &dice);
    void apply_early_deflection(Packet *packet, bool consider_servers, InterfaceEntry *ie2);
    // dctcp
    void dctcp_mark_ecn_for_deflected_packets(Packet *packet, bool has_phy_header=true);
    // bolt
    void generate_and_send_bolt_src_packet(Packet *packet, int queue_occupancy_pkt_num, long link_util, int extraction_port_interface_id);
    void bolt_pifo_evaluate_if_src_packet_should_be_generated(Packet *packet, InterfaceEntry *, bool ignore_cc_thresh);
    void bolt_evaluate_if_src_packet_should_be_generated(Packet *packet, InterfaceEntry *, bool ignore_cc_thresh, bool has_phy_header=true, int extraction_port_interface_id=-1);
    void mark_packet_deflection_tag(Packet *packet, bool has_phy_header=true);
    bool bolt_is_packet_src(Packet *packet);
    
    // RL-based deflection methods
    void load_rl_model();
    std::vector<double> extract_rl_state_features(Packet *packet, InterfaceEntry *ie, size_t &flowKey);
    bool get_rl_deflection_decision(Packet *packet, InterfaceEntry *ie);
    double compute_alpha(double dt_us) const;
    std::string make_queue_key(const std::string& switchName, int interfaceIndex) const;
    RunningStats& access_queue_stats(const std::string& switchName, int interfaceIndex);
    RunningStats& access_total_stats(const std::string& switchName);
    RLFlowState& access_flow_state(size_t flowKey, simtime_t now, uint64_t sequenceNo);
    void update_flow_state_post_action(size_t flowKey, bool deflected);
    std::vector<double> build_history_vector(const RLFlowState& state) const;
    double compute_seq_norm(const RLFlowState& state) const;
    double compute_flow_age_norm(const RLFlowState& state, simtime_t now) const;
    double compute_queue_util_z(const std::string& switchName, int interfaceIndex, double value);
    double compute_total_util_z(const std::string& switchName, double value);
    int get_optional_int_param(const char *name, int defaultValue) const;
    double get_optional_double_param(const char *name, double defaultValue) const;
    
    void initialize_token_buckets();
    int resolve_token_bucket_index(InterfaceEntry *port) const;
    void refill_token_bucket(int bucket_index, omnetpp::simtime_t now);
    bool try_consume_token(int bucket_index, omnetpp::simtime_t now);
    bool acquire_token_for_deflection(InterfaceEntry *egress, InterfaceEntry *original, const char *context);
    
    unsigned long getRequesterIDFromPacket(Packet *packet, InterfaceEntry *ie = nullptr);
    unsigned long getSequenceNumberFromPacket(Packet *packet, InterfaceEntry *ie);
    unsigned long getPacketIDFromPacket(Packet *packet, InterfaceEntry *ie);

    void print_deflections_per_second();
};

#endif // ifndef __INET_BouncingIEEE8021DRELAY_H
