/* Copyright 2020 Stanford
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef _FLEXFLOW_SIMULATOR_H_
#define _FLEXFLOW_SIMULATOR_H_

#include "ffconst.h"
#include "config.h"
#include "blossom_match.h"
#include <memory>
#include <queue>
#include <fstream>
#include <unordered_map>
#include <unordered_set>

class Conv2DMeta;
class LinearMeta;
class Pool2DMeta;
class ElementUnaryMeta;
class ElementBinaryMeta;
//class SoftmaxMeta;
class BatchMatmulMeta;
//class BatchNormMeta;
class ConcatMeta;
//class DropoutMeta;
class TransposeMeta;
class Op;
class FFModel;

#define MOD(a, b) ((a) % (b)) < 0 ? ((a) % (b)) + (b) : ((a) % (b))
//#define DEBUG_PRINT

namespace flatbuffers {
  class FlatBufferBuilder;
}

struct CostMetrics {
  double forward_time, backward_time;
  size_t memory_requirement;
};

class Device {
public:
    enum DeviceType {
        DEVICE_COMP,
        DEVICE_MEM,
        DEVICE_COMM,
    };
    Device(std::string const &name, DeviceType type, int node_id, int socket_id, int device_id);
    std::string name;
    DeviceType type;
    int node_id;
    int socket_id;
    int device_id;
};

class CompDevice : public Device {
public:
    enum CompDevType {
        LOC_PROC,   //CPU
        TOC_PROC,   //GPU
    };
    CompDevType comp_type;
    size_t capacity;
    CompDevice(std::string const &name, CompDevType comp_type, int node_id, int socket_id, int device_id);
};

class MemDevice : public Device {
public:
    enum MemDevType {
        SYSTEM_MEM,     // DRAM on a single node
        Z_COPY_MEM,     // Zero-copy memory betweeen CPU DRAM and all GPUs on a single node
        GPU_FB_MEM,     // GPU framebuffer memory for a single GPU
    };
    MemDevType mem_type;
    size_t capacity;
    MemDevice(std::string const &name, MemDevType mem_type, int node_id, int socket_id, int device_id, size_t capacity);
};

class CommDevice : public Device {
public:
  enum CommDevType {
    MEMBUS_COMM,
    UPI_IN_COMM,
    UPI_OUT_COMM,
    NIC_IN_COMM,
    NIC_OUT_COMM,
    PCI_TO_HOST_COMM,
    PCI_TO_DEV_COMM,
    NVLINK_COMM,
    NW_COMM,
    NW_NOMINAL,
  };
  CommDevType comm_type;
  double latency;
  double bandwidth;
  CommDevice(std::string const &name, CommDevType comm_type, int node_id, int socket_id, int device_id, double latency, double bandwidth);
};

typedef std::vector<CommDevice *> Route;
/* first is an array of cumulative distribution */
typedef std::pair<std::vector<double>, std::vector<Route> > EcmpRoutes;
typedef std::vector<int> ConnectionMatrix;
class NetworkRoutingStrategy;
/**
 * Nomincal communication device. 
 * This is an communication device that allows "path expansion"
 * With this device, its possible to store a taskgraph in the "logical" 
 * view (p2p) while when doing the simulaion, expand to physical version
 */
class NominalCommDevice : public CommDevice {
public:
  // NominalCommDevice(std::string const &name, int device_id, const EcmpRoutes& routes);
  NominalCommDevice(std::string const &name, int device_id, int nnode, NetworkRoutingStrategy * routing);
  /* pick one of the weighted ECMP path */
  Route expand_to_physical() const;
  const EcmpRoutes & get_all_routes();
  void set_physical_paths(const EcmpRoutes& rs);
  void reset();
  // static inline int get_from_dev(int devid, int total) {return devid / total;}
  // static inline int get_to_dev(int devid, int total) {return devid % total;}
public:
  NetworkRoutingStrategy * routing_strategy;
  EcmpRoutes routes;
  int nnode;
  bool dirty = true;
};

class MachineModel {
public:
  virtual ~MachineModel() = default;
  virtual int get_version() const = 0;
  virtual CompDevice *get_gpu(int device_id) const = 0;
  virtual MemDevice *get_gpu_fb_mem(int devicd_id) const = 0;
  virtual int get_num_gpus() const = 0;
  virtual int get_total_devs() const {return get_num_gpus();}
  virtual int get_num_nodes() const = 0;
  virtual double get_intra_node_gpu_bandwidth() const = 0;
  virtual double get_inter_node_gpu_bandwidth() const = 0;
  virtual double get_intra_node_gpu_latency() const = 0;
  virtual double get_inter_node_gpu_latency() const = 0;
  virtual std::vector<CommDevice *> get_comm_path(MemDevice *src_mem, MemDevice *tar_mem) const = 0;
  virtual std::string to_string() const = 0;
  int version;
};

class SimpleMachineModel : public MachineModel {
public:
  SimpleMachineModel(int num_nodes, int num_gpus_per_node, size_t capacity);
  ~SimpleMachineModel();
  int get_version() const;
  CompDevice *get_gpu(int device_id) const;
  MemDevice *get_gpu_fb_mem(int devicd_id) const;
  int get_num_gpus() const;
  int get_num_nodes() const {return num_nodes;}
  double get_intra_node_gpu_bandwidth() const;
  double get_inter_node_gpu_bandwidth() const;
  double get_intra_node_gpu_latency() const {return 0;}
  double get_inter_node_gpu_latency() const {return 0;}
  std::vector<CommDevice *> get_comm_path(MemDevice *src_mem, MemDevice *tar_mem) const;
  std::string to_string() const;
public:
  int num_nodes;
  int num_gpus_per_node;
  int num_gpus;
  double inter_gpu_bandwidth;
  double inter_node_bandwidth;
  double gpu_dram_bandwidth;
  std::map<int, CompDevice*> id_to_gpu;
  std::map<int, MemDevice*> id_to_gpu_fb_mem;
  std::map<int, CommDevice*> id_to_gputodram_comm_device;
  std::map<int, CommDevice*> id_to_dramtogpu_comm_device;
  std::map<size_t, CommDevice*> ids_to_inter_gpu_comm_device;
  std::map<size_t, CommDevice*> ids_to_inter_node_comm_device;
};

/**
 * An enhanced machine model supports the following features:
 * 1. Customize the machine model with a configuration file.
 * 2. Support socket-level simulation.
 * 3. Simulate congestions on a communication device. In this machine model, some communication 
 *    devices, such as NIC_IN and NIC_OUT, represent the communication ports instead of the links 
 *    in the simple machine model. In this way, for example, concurrent inter-node communications 
 *    from node A to node B and from node A to node C share the same NIC_OUT device on node A, 
 *    which simulates the slowdown of concurrent communications when transferring big messages.
 * 4. When passing big messages, the messages usually are divided into segments and transferred 
 *    one-by-one to overlap the communications on different devices. This machine model can 
 *    simulate this kind of pipelining.
 */ 
class EnhancedMachineModel : public MachineModel {
public:
    enum NicDistribution {
      PER_NODE,
      PER_SOCKET,
    };
    EnhancedMachineModel(std::string file, size_t gpu_fb_mem_capacity);
    ~EnhancedMachineModel();
    int get_version() const;
    CompDevice *get_cpu(int device_id) const;
    CompDevice *get_cpu(int socket_id, int local_id) const;
    CompDevice *get_gpu(int device_id) const;
    CompDevice *get_gpu(int socket_id, int local_id) const;
    MemDevice *get_sys_mem(int socket_id) const;
    MemDevice *get_z_copy_mem(int socket_id) const;
    MemDevice *get_gpu_fb_mem(int device_id) const;
    MemDevice *get_gpu_fb_mem(int socket_id, int local_id) const;
    CommDevice *get_nvlink(MemDevice *src_mem, MemDevice *tar_mem) const;
    int get_num_gpus() const;
    int get_num_nodes() const {return num_nodes;}
    double get_intra_node_gpu_bandwidth() const;
    double get_inter_node_gpu_bandwidth() const;
    double get_intra_node_gpu_latency() const {return membus_latency;}
    double get_inter_node_gpu_latency() const {return nic_latency;}
    std::vector<CommDevice *> get_comm_path(MemDevice *src_mem, MemDevice *tar_mem) const;
    std::string to_string() const;
public:
    int num_nodes;
    int num_sockets_per_node;
    int num_cpus_per_socket;
    int num_gpus_per_socket;
    int num_sockets;
    int num_cpus;
    int num_gpus;
    int num_nvlinks_per_node;
    double membus_latency;
    double membus_bandwidth;
    double upi_latency;
    double upi_bandwidth;
    double nic_latency;
    double nic_bandwidth;
    NicDistribution nic_distribution;
    double pci_latency;
    double pci_bandwidth;
    double nvlink_latency;
    double nvlink_bandwidth;
    size_t gpu_fb_mem_capacity;
    std::vector<CommDevice::CommDevType> intra_socket_sys_mem_to_sys_mem;
    std::vector<CommDevice::CommDevType> inter_socket_sys_mem_to_sys_mem;
    std::vector<CommDevice::CommDevType> inter_node_sys_mem_to_sys_mem;
    std::vector<CommDevice::CommDevType> intra_socket_sys_mem_to_gpu_fb_mem;
    std::vector<CommDevice::CommDevType> inter_socket_sys_mem_to_gpu_fb_mem;
    std::vector<CommDevice::CommDevType> inter_node_sys_mem_to_gpu_fb_mem;
    std::vector<CommDevice::CommDevType> intra_socket_gpu_fb_mem_to_sys_mem;
    std::vector<CommDevice::CommDevType> inter_socket_gpu_fb_mem_to_sys_mem;
    std::vector<CommDevice::CommDevType> inter_node_gpu_fb_mem_to_sys_mem;
    std::vector<CommDevice::CommDevType> intra_socket_gpu_fb_mem_to_gpu_fb_mem;
    std::vector<CommDevice::CommDevType> inter_socket_gpu_fb_mem_to_gpu_fb_mem;
    std::vector<CommDevice::CommDevType> inter_node_gpu_fb_mem_to_gpu_fb_mem;
    std::vector<std::vector<CompDevice *>> cpus;   // socket_id, local_id
    std::vector<std::vector<CompDevice *>> gpus;   // socket_id, local_id
    std::vector<MemDevice *> sys_mems;             // socket_id
    std::vector<MemDevice *> z_copy_mems;          // socket_id
    std::vector<std::vector<MemDevice *>> gpu_fb_mems;     // socket_id, local_id
    std::vector<CommDevice *> membuses;            // socket_id
    std::vector<CommDevice *> upi_ins;             // socket_id
    std::vector<CommDevice *> upi_outs;            // socket_id
    std::vector<CommDevice *> nic_ins;             // socket_id
    std::vector<CommDevice *> nic_outs;            // socket_id
    std::vector<CommDevice *> pcis_to_host;             // from gpu to main memory, socket_id
    std::vector<CommDevice *> pcis_to_device;            // from main memory to gpu, socket_id
    std::vector<std::vector<CommDevice *>> nvlinks;    // node_id, local_id
    std::unordered_map<size_t, CommDevice *> mem_to_nvlink;
    // set up communication paths from a config file
    void set_comm_path(std::vector<CommDevice::CommDevType> &comm_path, std::string device_str);
    void add_cpus();
    void add_gpus();
    void add_membuses(double latency, double bandwidth);
    void add_upis(double latency, double bandwidth);
    void add_nics(double latency, double bandwidth, NicDistribution nic_distribution);
    void add_pcis(double latency, double bandwidth);
    void add_nvlinks(double latency, double bandwidth);
    // attach a nvlink communication device to a pair of GPU framebuffer memories
    void attach_nvlink(MemDevice *src_mem, MemDevice *tar_mem, CommDevice *comm);
    // return a list of specific communication devices based on the descriptions of a communication path
    void add_comm_path(std::vector<CommDevice::CommDevType> const &comm_device_list, MemDevice *src_mem, MemDevice *tar_mem, std::vector<CommDevice *> &ret) const;
};



/**
 * Base class that provides the network routing strategy
 */
class NetworkRoutingStrategy {
public:
    virtual ~NetworkRoutingStrategy() = default;
    /**
     * For weighted ecmp support: the return type is a vector of pair of 
     * <possible route, chance>
     */
    virtual EcmpRoutes get_routes(int src_node, int dst_node) = 0;
    virtual std::vector<EcmpRoutes> get_routes_from_src(int src_node) = 0;
};

/**
 * Single shortest path routing based on hop count
 */
class WeightedShortestPathRoutingStrategy : public NetworkRoutingStrategy {
public:
    WeightedShortestPathRoutingStrategy(const ConnectionMatrix & c, 
        const std::map<size_t, CommDevice*>& devmap, int total_devs);
    virtual EcmpRoutes get_routes(int src_node, int dst_node);
    virtual std::vector<EcmpRoutes> get_routes_from_src(int src_node);
    void hop_count(int src_node, int dst_node, int & hop, int & narrowest);
    std::vector<std::pair<int, int>> hop_count(int src_node);
    void clear();
public:
    const ConnectionMatrix& conn;
    const std::map<size_t, CommDevice*>& devmap;
    int total_devs;
};

class ShortestPathNetworkRoutingStrategy : public NetworkRoutingStrategy {
public:
    ShortestPathNetworkRoutingStrategy(const ConnectionMatrix & c, 
        const std::map<size_t, CommDevice*>& devmap, int total_devs);
    virtual EcmpRoutes get_routes(int src_node, int dst_node);
    virtual std::vector<EcmpRoutes> get_routes_from_src(int src_node);
    void hop_count(int src_node, int dst_node, int & hop, int & narrowest);
    std::vector<std::pair<int, int>> hop_count(int src_node);
    void clear();
public:
    const ConnectionMatrix& conn;
    const std::map<size_t, CommDevice*>& devmap;
    int total_devs;
};

/**
 * A model that is network topology-aware.
 * The network topology is represented as follows:
 *      An adjacency matrix is used to represnt the network connection
 *      The matrix has dimension (n+s)*(n+s) where n is the number of servers
 *      in the cluster, and s is the number of switches in the cluster.
 *      This implies that for a flat topology the matrix is n*n,
 *      while for a FatTree topology the network will have the upper n*n
 *      block to be 0. Switches has node_id starting from n.
 *      Note that the "big switch" model has the convinent representation of
 *      {{0, 1},{1, 0}} in block form.
 * As a first implementation this class is based on the existing SimpleMachine
 * model. We could use the enhanced version but it could be too much for the 
 * MCMC search to run for thousand of iterations...
 */
class NetworkedMachineModel : public MachineModel  {
  friend class DemandHeuristicNetworkOptimizer;
  friend class DemandHeuristicNetworkOptimizerPlus;
public:
  /**
   * Constructor. A network topology specified as above needs to be provided
   * in the form of a single vector.
   */
  NetworkedMachineModel(int num_nodes, 
      int num_gpus_per_node, 
      int num_switches, 
      double network_latency,
      const std::vector<int>& topology, 
      size_t capacity, 
      double link_bandwidth);
  ~NetworkedMachineModel();
  int get_version() const;
  CompDevice *get_gpu(int device_id) const;
  MemDevice *get_gpu_fb_mem(int devicd_id) const;
  int get_num_gpus() const;
  int get_num_nodes() const {return num_nodes;}
  int get_total_devs() const {return num_nodes + num_switches;}
  int get_num_switches() const {return num_switches;}
  double get_intra_node_gpu_bandwidth() const;
  double get_inter_node_gpu_bandwidth() const;
  double get_link_bandwidth() const;
  double get_link_bandwidth(int src, int dst) const;
  double get_intra_node_gpu_latency() const {return 0;}
  double get_inter_node_gpu_latency() const {return network_latency;}
  void set_routing_strategy(NetworkRoutingStrategy* rs);
  std::vector<CommDevice *> get_comm_path(MemDevice *src_mem, MemDevice *tar_mem) const;
  std::string to_string() const;
  /* return only the nominal device. For recording tg. */
  CommDevice* get_nominal_path(MemDevice* src_mem, MemDevice *tar_mem) const;
  /* stores the network topology as a json */
  void save_topology_json(const std::string& fname) const;
  void update_route();

  void set_topology(const std::vector<int>& topology);
  const ConnectionMatrix & get_conn_matrix();
  const std::map<size_t, NominalCommDevice*>& get_nomm_comm_devs();

  void set_pcie(bool state);
  void set_pipeline(bool state);
  
  int num_nodes;
  int num_gpus_per_node;
  int num_gpus;
  int num_switches;
  int total_devs;
  double inter_gpu_bandwidth;
  double link_bandwidth;
  double network_latency;
  double gpu_dram_bandwidth;

  bool pipelined;
  bool pcie_on;

  // double gpu_dram_bandwidth;
  /* Note that every non-zero entry corrsepond to a device in in_to_nw_comm_device */
  ConnectionMatrix conn_matrix;
  NetworkRoutingStrategy* routing_strategy;
  std::map<int, CompDevice*> id_to_gpu;
  std::map<int, MemDevice*> id_to_gpu_fb_mem;
  // don't model PCIE for speed
  std::map<int, CommDevice*> id_to_gputodram_comm_device;
  std::map<int, CommDevice*> id_to_dramtogpu_comm_device;
  std::map<size_t, CommDevice*> ids_to_inter_gpu_comm_device;
  
  /* this refers to the actual links in the system */
  std::map<size_t, CommDevice*> ids_to_nw_comm_device;
  /* on the other hand, this represents the "nomical" communication device
    * or the "logical connection" in side the system. Note that this is
    * keyed on GPUs only 
    */
  std::map<size_t, NominalCommDevice*> ids_to_nw_nominal_device;

public:
  std::map<size_t, uint64_t> logical_traffic_demand;
  std::map<size_t, uint64_t> physical_traffic_matrix;
};

/**
 * A (virtual base) class that generates network topology 
 * Maybe this should be moved out of simulator
 */
class NetworkTopologyGenerator {
public:
  virtual ConnectionMatrix generate_topology() const = 0;
  static void print_conn_matrix(const ConnectionMatrix &conn, int nnode, int nswitch) {
    int nnwdevs = nnode + nswitch;
    for (int i = 0; i < nnwdevs; i++) {
      if (i == nnode) std::cout << std::endl;
      for (int j = 0; j < nnwdevs; j++) {
        if (j == nnode) std::cout << "\t";
        std::cout << conn[i * nnwdevs + j] << "\t";
      }
      std::cout << std::endl;
    }
  }
};

/**
 * Generate a flat network topology that's degree constraint and guaranteed
 * to be connected
 */
class FlatDegConstraintNetworkTopologyGenerator : public NetworkTopologyGenerator {
public:
    FlatDegConstraintNetworkTopologyGenerator(int num_nodes, int degree);
    virtual ConnectionMatrix generate_topology() const;
public:
    inline int get_id(int i, int j) const;
    inline int get_if_in_use(int node, const ConnectionMatrix & conn) const;
    int num_nodes;
    int degree;
};

/**
 * Generate an abstract-switch network topology
 * good for simple simulation of a fattree
 */
class BigSwitchNetworkTopologyGenerator : public NetworkTopologyGenerator  {
public:
    BigSwitchNetworkTopologyGenerator(int num_nodes);
    virtual ConnectionMatrix generate_topology() const;
public: 
    int num_nodes;
};

/**
 * Generate a zero matrix
 */
class FlatEmptyNetworkTopologyGenerator : public NetworkTopologyGenerator  {
public:
    FlatEmptyNetworkTopologyGenerator(int num_nodes): num_nodes(num_nodes){}
    virtual ConnectionMatrix generate_topology() const {return ConnectionMatrix(num_nodes*num_nodes, 0);} 
public:
    int num_nodes;
};

class FCTopologyGenerator : public NetworkTopologyGenerator  {
public:
    FCTopologyGenerator(int num_nodes): num_nodes(num_nodes){}
    virtual ConnectionMatrix generate_topology() const {
      ConnectionMatrix result = ConnectionMatrix(num_nodes*num_nodes, 1);
      for (int i = 0; i < num_nodes; i++) result[i + i * num_nodes] = 0;
      return result;
    } 
public:
    int num_nodes;
};

class SimTask {
public:
  enum SimTaskType {
    TASK_FORWARD,
    TASK_BACKWARD,
    TASK_COMM,
    TASK_UPDATE,
    TASK_BARRIER,
    TASK_NOMINAL_COMM,
    TASK_ALLREDUCE
  };
  SimTask();
  void add_next_task(SimTask* task);
public:
  double ready_time, run_time;
  SimTaskType type;
  Device* device;
  MemDevice *mem;
  int counter;
  size_t xfer_size;
  size_t xfer_left;
  std::vector<SimTask*> next_tasks;
#if 0
  /* This stores only high-level information: computation and communication
   * from device to device 
   */
  std::vector<SimTask*> next_tasks_simplified;  
#endif 
  /*
   * in the case of logical task graph is used, some task may not need to be
   * storted. This flag if the task should be stored.
   */
  bool store;
  std::string name;
  std::string get_type_str() const;
};

/**
 * hack: since AllReduceTask only occurs in the logical task graph
 * and to keep TaskManager useable, use next_tasks to store the integer
 * of all reduce groupd node ids, and counter to store the leader.
 * Lets just say its an union...
 */
#if 0
/**
 * Special class of tasks for all reduce. This is more for exporting
 * task graph since recording individual transfers would take too much space
 */
class AllReduceTask : public SimTask {
public:
  std::vector<int> devicve_group; /* deviced involved in tha all reduce task */
  int leader;                     /* the "special" device (eg p server) */
};
#endif

struct L1OptimizerInformation {};

struct L1TopologyInformation: public L1OptimizerInformation {
  L1TopologyInformation(const ConnectionMatrix & conn): conn(conn) {};
  ConnectionMatrix conn;
};

/**
 * Interface for doing network topology optimization
 * define all your data structures... 
 * import_information uses the task graph to reconstruct useful information,
 * and export_information uses void* for best generallity
 */
class L1Optimizer {
public:
    L1Optimizer(MachineModel* machine)
        : machine(machine) {}
    virtual bool optimize(int mcmc_iter, double sim_iter_time, bool force_run = false) = 0;
    virtual void task_added(SimTask * task) { return; };
    virtual void reset() = 0;
    virtual std::unique_ptr<L1OptimizerInformation> export_information() = 0;
    virtual void import_information(const std::unique_ptr<L1OptimizerInformation>& information) = 0;
    virtual void delete_information(const std::unique_ptr<L1OptimizerInformation>& information) = 0;
    virtual void store_tm() const = 0;

protected:
    MachineModel *machine; // Would really like to do a T extends MachineModel...
};
 
// #if 0
/**
 * TODO: TopoOpt as of SIGCOMM 2021 submission
 */
class DemandHeuristicNetworkOptimizer : public L1Optimizer {
  friend class Simulator;
  friend class FFModel;
public:
  DemandHeuristicNetworkOptimizer(MachineModel* machine);
  ~DemandHeuristicNetworkOptimizer() = default;
  virtual bool optimize(int mcmc_iter, double sim_iter_time, bool force_run = false);
  virtual void task_added(SimTask * task);
  virtual void reset();
  virtual std::unique_ptr<L1OptimizerInformation> export_information();
  virtual void import_information(const std::unique_ptr<L1OptimizerInformation>& information);
  virtual void delete_information(const std::unique_ptr<L1OptimizerInformation>& information);
  size_t edge_id(int i, int j) const;
  size_t unordered_edge_id(int i, int j) const;
  void optimize_demand(
      ConnectionMatrix &conn,
      std::unordered_map<size_t, uint64_t> &max_of_bidir,
      std::unordered_map<size_t, size_t> &node_if_allocated); 
  void connect_unused_node(ConnectionMatrix &conn, std::unordered_map<size_t, size_t> &node_if_allocated);
  void connect_cc(std::unordered_map<uint64_t, uint64_t> &logical_id_to_demand, 
                  ConnectionMatrix &conn);

  size_t get_if_in_use(size_t node, const ConnectionMatrix & conn);
  bool add_link(size_t i, size_t j, ConnectionMatrix & conn);
  void remove_link(size_t i, size_t j, ConnectionMatrix & conn);

  virtual void store_tm() const;

  inline static bool has_endpoint(uint64_t e, size_t v, size_t n) {
    return e / n == v || e % n == v;
  }

  inline static bool maxed(const std::unordered_map<size_t, size_t> & node_if_allocated, 
                           size_t d, size_t n) 
  {
    size_t counter = 0;
    for (auto & item: node_if_allocated) {
      if (item.second == d) {
        counter++;
      }
    }
    return counter >= (n - 1);
  }

  inline static bool maxed(const std::unordered_map<size_t, size_t> & node_if_allocated, 
            const std::unordered_set<size_t> & nodes_to_care,
            size_t d) 
  {
    size_t counter = 0;
    for (auto & item: node_if_allocated) {
      if (nodes_to_care.find(item.first) != nodes_to_care.end() && item.second == d) {
        counter++;
      }
    }
    return counter >= (nodes_to_care.size() - 1);
  }


  std::unordered_map<uint64_t, uint64_t> dev_busy_time;
  // std::unordered_map<size_t, uint64_t> dev_busy_time_gpu;
  // std::unordered_map<size_t, uint64_t> dev_busy_time_dram_gpu;
  // std::unordered_map<size_t, uint64_t> dev_busy_time_gpu_dram;
  // std::unordered_map<size_t, uint64_t> dev_busy_time_gpu_gpu;
  std::unordered_map<size_t, uint64_t> physical_traffic_demand;
  std::unordered_map<size_t, uint64_t> logical_traffic_demand;

  size_t if_cnt;
  double best_sim_time, curr_sim_time;
  double alpha;
  int num_iter_nochange;
  int no_improvement_th;

};
// #endif

class DemandHeuristicNetworkOptimizerPlus : public DemandHeuristicNetworkOptimizer 
{
public:
  DemandHeuristicNetworkOptimizerPlus(MachineModel* machine);

  void connectivity_assign(ConnectionMatrix &conn,
    std::unordered_map<size_t, uint64_t> &max_of_bidir,
    std::unordered_map<size_t, size_t> &node_if_allocated);
  void connect_topology(
    // const std::unordered_map<uint64_t, uint64_t> & logical_id_to_demand, 
    ConnectionMatrix &conn, 
    std::unordered_map<size_t, size_t> & node_if_allocated);
  void utility_max_assign(
    ConnectionMatrix &conn,
    // const std::unordered_map<size_t, uint64_t> &max_of_bidir,
    std::unordered_map<size_t, size_t> &node_if_allocated);
  double compute_utility(
    const std::unordered_map<size_t, std::pair<uint64_t, double>> &indirect_traffic,
    const ConnectionMatrix & conn);
  double compute_utility(
    const std::unordered_map<size_t,uint64_t> &sum_of_bidir,
    const std::unordered_map<size_t,uint64_t> &indirect_traffic,
    const ConnectionMatrix & conn);
  std::unordered_map<size_t, std::pair<uint64_t, double>>
    construct_indir_traffic_list(const ConnectionMatrix &conn); 
  std::unordered_map<size_t, uint64_t> 
    construct_bidir_negative_util(const ConnectionMatrix &conn);
  virtual bool optimize(int mcmc_iter, double sim_iter_time, bool force_run = false);
};

template <typename T>
class DotFile {
public:
  size_t node_id;
  std::map<T,size_t> node_ids;
  std::unique_ptr<std::ostream> out;
  std::string get_node_name(size_t node_id) const {
    std::ostringstream s;
    s << "node" << node_id;
    return s.str();
  }
public:
  DotFile() : node_id(0) {}
  DotFile(std::string const &filename) : DotFile(std::unique_ptr<std::ostream>(new std::ofstream(filename))) {}
  DotFile(std::unique_ptr<std::ostream> s)
    : node_id(0), out(std::move(s))
  {
    *out << "digraph taskgraph {";
  }

  void set_filename(std::string filename) {
    this->out = std::unique_ptr<std::ostream>(new std::ofstream(filename));
    *out << "digraph taskgraph {";
  }
  void reserve_node(T const &t) {
    if (this->node_ids.find(t) == this->node_ids.end()) {
      this->node_ids[t] = this->node_id++;
    }
  }
  void add_node(T const &t, std::map<std::string, std::string> const &params) {
    this->reserve_node(t);
    *out << "  " << this->get_node_name(this->node_ids.at(t)) << " [";
    for (auto it = params.begin(); it != params.end(); ++it)  {
      *out << it->first << "=" << it->second;
      if (std::next(it) != params.end()) {
        *out << ",";
      }
    }
    *out << "];" << std::endl;
  }
  void add_edge(T const &src, T const &dst) {
    this->reserve_node(src);
    this->reserve_node(dst);
    auto src_name = this->get_node_name(this->node_ids.at(src));
    auto dst_name = this->get_node_name(this->node_ids.at(dst));
    *out << "  " << src_name << " -> " << dst_name << ";" << std::endl;
  }
  void close() {
    *out << "}";
    out->flush();
  }
};

class SimTaskCompare {
public:
  bool operator() (SimTask* lhs, SimTask* rhs) {
    return lhs->ready_time > rhs->ready_time;
  }
};

class TaskManager {
public:
  TaskManager(size_t max_num_tasks);
  void reset();
  SimTask* new_barrier_task();
  SimTask* new_update_task();
  SimTask* new_comm_task();
  SimTask* new_nominal_comm_task();
  SimTask* new_comm_task(std::string const &name, CommDevice *comm_device, size_t message_size);
  SimTask* new_nominal_comm_task(std::string const &name, CommDevice *comm_device, size_t message_size);
  SimTask* new_forward_task(Op* op, int idx);
  SimTask* new_allreduce_task(Op* op, const std::vector<int> &node_ids, size_t message_size); 
  SimTask* new_backward_task(Op* op, int idx);
  SimTask* get_forward_task(Op* op, int idx);
  SimTask* get_backward_task(Op* op, int idx);
  
  SimTask* new_task();
public:
  size_t global_task_id, max_num_tasks;
  SimTask** tasks;
  
  std::map<size_t, SimTask*> hash_to_forward_task, hash_to_backward_task;
};

size_t data_type_size(DataType);

class Simulator {
public:
  Simulator(const FFModel* model,
            FFHandler handler,
            Memory memory,
            MachineModel *machine);
  ~Simulator(void);
  virtual void free_all();
  virtual void* allocate(uint64_t num_elements, DataType type);
  virtual void add_task_dependencies_with_xfer(
      SimTask* src_task, SimTask* dst_task, size_t message_size);
  CostMetrics measure_operator_cost(Op* op, const ParallelConfig& config);
  virtual double simulate_runtime(const FFModel* model,
      const std::map<Op*, ParallelConfig>& global,
      CompMode comp_mode);
  virtual double simulate_runtime(const FFModel* model,
      const std::map<Op*, ParallelConfig>& global,
      CompMode comp_mode,
      std::string const &export_file_name);
  static void strategy_search_task(const Task *task,
                                   const std::vector<PhysicalRegion> &regions,
                                   Context ctx, Runtime *runtime);
  static void simulation_task(const Task *task,
                                   const std::vector<PhysicalRegion> &regions,
                                   Context ctx, Runtime *runtime);                                   
  static void measurement_task(const Task *task,
                                   const std::vector<PhysicalRegion> &regions,
                                   Context ctx, Runtime *runtime);                                   
public:
  Realm::RegionInstance simulatorInst;
  MachineModel *machine;
  Memory memory;
  FFHandler handler;
  char* base_ptr;
  uint64_t capacity;
  uint64_t offset;
  int warmup_times, repeat_times;
  TaskManager* task_manager;
  CompMode computationMode;
  cudaEvent_t start_event, end_event;
  std::map<size_t, CostMetrics> hash_to_operator_cost;
public:
  Conv2DMeta* conv2d_meta;
  LinearMeta* linear_meta;
  Pool2DMeta* pool2d_meta;
  ElementUnaryMeta* ele_unary_meta;
  ElementBinaryMeta* ele_binary_meta;
  //SoftmaxMeta *softmax_meta;
  BatchMatmulMeta *batch_matmul_meta;
  //BatchNormMeta *batch_norm_meta;
  ConcatMeta *concat_meta;
  //DropoutMeta *dropout_meta;
  TransposeMeta *transpose_meta;
  int segment_size;
  int max_num_segments; //simulation could be slow if the number of segments are too large

  std::unordered_map<std::string, CostMetrics>* measurements;

  /* extra optimizer that changes physical properties.
   * Each time a task is added, a callback would be called into this 
   * optimizer to provide information 
   */
  L1Optimizer* l1optimizer;
};

/**
 * An alternative implementation of the simulator which uses the "logical 
 * task graph", defined as a taskgraph that only records computation
 * and communication on a logical level.
 */
class LogicalTaskgraphBasedSimulator: public Simulator {
public: 
  LogicalTaskgraphBasedSimulator(const FFModel* model,
            FFHandler handler,
            Memory memory,
            MachineModel *machine);
  
  SimTask *new_comm_task_unrecorded();
  SimTask *new_update_task_unrecorded();
  virtual double simulate_runtime(const FFModel* model,
      const std::map<Op*, ParallelConfig>& global,
      CompMode comp_mode);
  virtual double simulate_runtime(const FFModel* model,
      const std::map<Op*, ParallelConfig>& global,
      CompMode comp_mode,
      std::string const &export_file_name);
  virtual double route_transfer(SimTask * transfer_task, 
                              double start_time,
                              std::map<Device*, double> &device_times);
  virtual double route_transfer_seg(SimTask * transfer_task, 
                            double start_time,
                            std::map<Device*, double> &device_times,
                            bool & finished);
  virtual void expand_allreduce(SimTask * allreduce_task, double start_time,std::priority_queue<SimTask*, std::vector<SimTask*>, SimTaskCompare>& ready_queue);
  void add_task_dependencies_with_xfer(
      SimTask* src_task, SimTask* dst_task, size_t message_size);
  bool searlize_logical_taskgraph(const FFModel* model, std::string const &export_file_name);
  static void simulation_task(const Task *task,
                                  const std::vector<PhysicalRegion> &regions,
                                  Context ctx, Runtime *runtime);
  virtual void get_taskgraph_flatbuf(const FFModel* model, flatbuffers::FlatBufferBuilder &builder);
  virtual double compute_internal_ar_time(const FFModel* model, SimTask * allreduce_task);

  bool segment_transfer;
  size_t segment_size;

  // flatbuffers::FlatBufferBuilder builder;
};

struct DPGroup {
  int starting_node;
  int group_size;
  size_t xfer_size;
  // std::set<int> jump_dists;
};

struct SpMulMatInformation : public L1OptimizerInformation {
  ConnectionMatrix conn;
  std::unordered_map<uint64_t, std::vector<std::vector<int>>> selected_jumps;
  // std::unordered_map<uint64_t, std::vector<NominalCommDevice*>> dp_ncomms;
};

// Space-multiplexed matching?...
class SpMulMat : public DemandHeuristicNetworkOptimizer {
public: 
  SpMulMat(MachineModel * machine, int degree, bool bidir);
  ~SpMulMat() = default;

  virtual bool optimize(int mcmc_iter, double sim_iter_time, bool force_run = false);
  virtual void task_added(SimTask * task);
  virtual void reset();
  virtual std::unique_ptr<L1OptimizerInformation> export_information();
  virtual void import_information(const std::unique_ptr<L1OptimizerInformation>& information);
  virtual void delete_information(const std::unique_ptr<L1OptimizerInformation>& information);
  virtual void store_tm() const; 

  std::vector<std::pair<uint64_t, int>> generate_dp_topology(ConnectionMatrix & conn, int dp_degree);
  void generate_mp_matching(ConnectionMatrix & conn, int dp_degree);
  void generate_one_match(ConnectionMatrix & conn, std::unordered_map<uint64_t, uint64_t> & mp_tm);
  ConnectionMatrix add_ring(const ConnectionMatrix & conn, int start, int dist);
  // ConnectionMatrix construct_hop_matrix(const ConnectionMatrix & conn);
  double compute_mp_satified(const ConnectionMatrix & hop_matrix);
  void construct_candidate_jumps();
  std::vector<int> all_coin_change(const std::set<int> & coins);
  std::vector<int> query_path(const std::vector<int>& candidates, int jump);
  std::pair<blossom_match::Graph, std::vector<double>>
    convert_to_blsm_match_graph(std::unordered_map<uint64_t, uint64_t> & mp_tm);
  // std::vector<std::vector<int>> get_selected_jumps(int group_sz); 
  ConnectionMatrix connect_topology(const ConnectionMatrix & conn, 
    ConnectionMatrix & mp_conn, ConnectionMatrix & dp_conn, 
    const std::vector<std::pair<uint64_t, int>> & dp_rings, int mp_degree);
  void construct_topology();
  const std::vector<NominalCommDevice*>& get_dp_ncomms(int src, int grp_sz);

  // uint64_t get_mp_bandwidth_tax(const ConnectionMatrix & conn);

  inline void get_dp_mp_degree(int & dp_degree, int & mp_degree);
  // inline size_t edge_id(int i, int j) const;
  // inline size_t unordered_edge_id(int i, int j) const;
  // inline int get_start_node(uint64_t id) const;
  // inline int get_group_size(uint64_t id) const;
  // inline uint64_t dpgrp_unique_key(const DPGroup & dpg) const;
  inline bool segment_overlap(const std::vector<int>& a, const std::vector<int>& b);
  inline std::vector<int> negative(const std::vector<int>& v);
  inline std::vector<int> choose_n(const std::vector<int>& cjs, int init_jmp, int n);
  inline std::vector<int> choose_n_geo(const std::vector<int>& cjs, int n);

  void print_all_rings() const;

  std::unordered_map<uint64_t, std::vector<int>> candidate_jumps;
  std::unordered_map<uint64_t, std::vector<std::vector<int>>> selected_jumps;
  std::unordered_map<uint64_t, uint64_t> dpgrpsz_xfersize;
  std::vector<DPGroup> dpgrps;
  std::unordered_map<uint64_t, uint64_t> mp_tm_logical;

  std::unordered_map<uint64_t, std::vector<NominalCommDevice*>> dp_ncomms;

  // int degree;
  bool bidir;
  bool constructed;
  // double best_sim_time;
  // double curr_sim_time;
  // double alpha;
  // int num_iter_nochange;
  // int no_improvement_th;
};

class SpMulMatSimulator: public LogicalTaskgraphBasedSimulator {
public: 
  SpMulMatSimulator(const FFModel* model,
            FFHandler handler,
            Memory memory,
            MachineModel *machine);
  virtual double simulate_runtime(const FFModel* model,
      const std::map<Op*, ParallelConfig>& global,
      CompMode comp_mode);
  virtual double simulate_runtime(const FFModel* model,
      const std::map<Op*, ParallelConfig>& global,
      CompMode comp_mode,
      std::string const &export_file_name);
  virtual void expand_allreduce(SimTask * allreduce_task, double start_time,std::priority_queue<SimTask*, std::vector<SimTask*>, SimTaskCompare>& ready_queue);
  virtual void get_taskgraph_flatbuf(const FFModel* model, flatbuffers::FlatBufferBuilder &builder);
  static void simulation_task(const Task *task,
                                  const std::vector<PhysicalRegion> &regions,
                                  Context ctx, Runtime *runtime);
};

#endif
