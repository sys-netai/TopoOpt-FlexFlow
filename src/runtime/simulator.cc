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

#include <random>
#include "simulator.h"
#include "model.h"
#include "queue"

// #include "flatbuffers/util.h"
#include "taskgraph_generated.h"

// #define DEBUG_PRINT
// #define WRITE_NETWORK_TRANSFER

int ParallelConfig::num_parts() const
{
  int nparts = 1;
  for (int i = 0; i < nDims; i++)
    nparts *= dim[i];
  return nparts;
}

bool ParallelConfig::is_data_parallel() const
{
  int nparts = 1;
  for (int i = 0; i < nDims; i++) {
    nparts *= dim[i];
    if ((i < nDims-1) && (dim[i] > 1))
      return false;
  }
  for (int i = 0; i < nparts; i++)
    if (device_ids[i] != i)
      return false;
  return true;
}

Device::Device(std::string const &name, DeviceType type, int node_id, int socket_id, int device_id)
: name(name), type(type), node_id(node_id), socket_id(socket_id), device_id(device_id)
{}

CompDevice::CompDevice(std::string const &name, CompDevType comp_type, int node_id, int socket_id, int device_id)
: Device(name, Device::DEVICE_COMP, node_id, socket_id, device_id), comp_type(comp_type)
{}

MemDevice::MemDevice(std::string const &name, MemDevType mem_type, int node_id, int socket_id, int device_id, size_t capacity)
: Device(name, Device::DEVICE_MEM, node_id, socket_id, device_id), mem_type(mem_type), capacity(capacity)
{}

CommDevice::CommDevice(std::string const &name, CommDevType comm_type, int node_id, int socket_id, int device_id, double latency, double bandwidth)
: Device(name, Device::DEVICE_COMM, node_id, socket_id, device_id), comm_type(comm_type), latency(latency), bandwidth(bandwidth)
{}

/* I hate this but this makes sense here... */
static std::random_device rd; 
static std::mt19937 gen = std::mt19937(rd()); 
static std::uniform_real_distribution<> std_uniform = std::uniform_real_distribution<>(0.0, 1.0); 

// NominalCommDevice::NominalCommDevice(std::string const &name, int device_id, const EcmpRoutes& routes) 
// : CommDevice(name, CommDevice::NW_NOMINAL, -1, -1, device_id, 0, 0), routes(routes)
// {}

NominalCommDevice::NominalCommDevice(std::string const &name, int device_id, int nnodes, NetworkRoutingStrategy * routing) 
: CommDevice(name, CommDevice::NW_NOMINAL, -1, -1, device_id, 0, 0), routing_strategy(routing), dirty(true), nnode(nnodes)
{}

void NominalCommDevice::reset() 
{
  dirty = true;
  routes = {};
}
    
Route NominalCommDevice::expand_to_physical() const 
{
  if (dirty) {
    if (routing_strategy == nullptr)
      assert("don't know how to route!" && false);
    // std::cerr << name << " dirty... " << std::endl;
    *const_cast<EcmpRoutes*>(&routes) = routing_strategy->get_routes(device_id / nnode, device_id % nnode);
    *const_cast<bool*>(&dirty) = false;
  }

  assert(routes.first.size() > 0 || device_id / nnode == device_id % nnode);
  int pick = 0;
  double choice = std_uniform(gen);
  for (int i = 0; i < routes.first.size(); i++) {
    if (choice > routes.first[i]) break;
    pick = i;
  }
  Route ret = Route(routes.second[pick].begin(), routes.second[pick].end());
  return ret;
}

void NominalCommDevice::set_physical_paths(const EcmpRoutes &rs) 
{
  routes = rs;
  dirty = false;
}

const EcmpRoutes & NominalCommDevice::get_all_routes() 
{
  if (dirty) {
    if (routing_strategy == nullptr)
      assert("don't know how to route!" && false);
    // std::cerr << name << " dirty... " << std::endl;
    *const_cast<EcmpRoutes*>(&routes) = routing_strategy->get_routes(device_id / nnode, device_id % nnode);
    *const_cast<bool*>(&dirty) = false;
  }
  return routes;
}

SimTask::SimTask()
{}

void SimTask::add_next_task(SimTask* task)
{
  next_tasks.push_back(task);
  task->counter ++;
}

std::string SimTask::get_type_str() const {
  switch (type) {
    case TASK_FORWARD:
      return "Forward";
    case TASK_BACKWARD:
      return "Backward";
    case TASK_COMM:
      return "Comm";
    case TASK_UPDATE:
      return "Update";
    case TASK_BARRIER:
      return "Barrier";
    default:
      assert(false && "Unknown task type");
  }
}

TaskManager::TaskManager(size_t _max_num_tasks)
: max_num_tasks(_max_num_tasks)
{
  tasks = (SimTask**) malloc(sizeof(SimTask*) * max_num_tasks);
  for (size_t i = 0; i < max_num_tasks; i++) {
    tasks[i] = new SimTask();
  }
}

void TaskManager::reset()
{
  global_task_id = 0;
  hash_to_forward_task.clear();
  hash_to_backward_task.clear();
}

SimTask* TaskManager::new_task()
{
  assert(global_task_id + 1 < max_num_tasks);
  SimTask* task = tasks[global_task_id++];
  task->ready_time = 0.0f;
  task->run_time = 0.0f;
  task->next_tasks.clear();
  task->counter = 0;
  task->device = NULL;
  task->mem = NULL;
  task->name.clear();

  // task->from_dev = -1;
  // task->to_dev = -1;
  task->xfer_size = 0;
  task->xfer_left = 0;
  task->store = true;
  
  return task;
}

SimTask* TaskManager::new_update_task()
{
  SimTask* task = new_task();
  task->type = SimTask::TASK_UPDATE;
  return task;
}

SimTask* TaskManager::new_barrier_task()
{
  SimTask* task = new_task();
  task->type = SimTask::TASK_BARRIER;
  return task;
}

SimTask* TaskManager::new_comm_task()
{
  SimTask* task = new_task();
  task->type = SimTask::TASK_COMM;
  return task;
}


SimTask* TaskManager::new_nominal_comm_task()
{
  SimTask* task = new_task();
  task->type = SimTask::TASK_NOMINAL_COMM;
  return task;
}

SimTask* TaskManager::new_comm_task(std::string const &name, CommDevice *comm_device, size_t message_size)
{
  SimTask* task = new_task();
  task->type = SimTask::TASK_COMM;
  task->name = name;
  task->device = comm_device;
  task->run_time = comm_device->latency + message_size / comm_device->bandwidth;
  return task;
}

SimTask* TaskManager::new_nominal_comm_task(std::string const &name, CommDevice *comm_device, size_t message_size)
{
  SimTask* task = new_task();
  task->type = SimTask::TASK_NOMINAL_COMM;
  task->name = name;
  task->device = comm_device;
  task->run_time = comm_device->latency + message_size / comm_device->bandwidth;
  return task;
}

SimTask* TaskManager::new_forward_task(Op* op, int idx)
{
  SimTask* task = new_task();
  task->type = SimTask::TASK_FORWARD;
  size_t hash = 17 * 31 + (size_t)(op);
  hash = hash * 31 + std::hash<int>()(idx);
  hash_to_forward_task[hash] = task;
  task->name = op->name;
  return task;
}

SimTask* TaskManager::new_backward_task(Op* op, int idx)
{
  SimTask* task = new_task();
  task->type = SimTask::TASK_BACKWARD;
  size_t hash = 17 * 31 + (size_t)(op);
  hash = hash * 31 + std::hash<int>()(idx);
  hash_to_backward_task[hash] = task;
  task->name = op->name;
  return task;
}

SimTask* TaskManager::new_allreduce_task(Op *op, const std::vector<int> &node_ids, size_t message_size) 
{
  SimTask* task = new_task();
  task->type = SimTask::TASK_ALLREDUCE;
  // task->counter = node_ids[0];
  for (int i = 0; i < node_ids.size(); i++) {
    task->next_tasks.push_back(reinterpret_cast<SimTask*>(node_ids[i]));
  } 
  task->xfer_size = message_size;
  return task;
}

SimTask* TaskManager::get_forward_task(Op* op, int idx)
{
  size_t hash = 17 * 31 + (size_t)(op);
  hash = hash * 31 + std::hash<int>()(idx);
  assert(hash_to_forward_task.find(hash) != hash_to_forward_task.end());
  return hash_to_forward_task[hash];
}

SimTask* TaskManager::get_backward_task(Op* op, int idx)
{
  size_t hash = 17 * 31 + (size_t)(op);
  hash = hash * 31 + std::hash<int>()(idx);
  assert(hash_to_backward_task.find(hash) != hash_to_backward_task.end());
  return hash_to_backward_task[hash];
}

void Simulator::free_all()
{
  offset = 0;
}

size_t data_type_size(DataType type) {
  switch (type) {
    case DT_FLOAT:
      return sizeof(float);
    case DT_DOUBLE:
      return sizeof(double);
    case DT_INT32:
      return sizeof(int32_t);
    case DT_INT64:
      return sizeof(int64_t);
    case DT_BOOLEAN:
      return sizeof(bool);
    default:
      assert(false);
  }
}

void* Simulator::allocate(uint64_t num_elements, DataType type)
{
  uint64_t element_size = data_type_size(type);
  void* ret_ptr = base_ptr + offset;
  offset += element_size * num_elements;
  if (offset > capacity) {
    fprintf(stderr, "Simulator cannot measure some operators' performance."
        " Increate --simulator-workspace-size to at least %lu. element_size: %lu, num_elements: %lu, capacity: %lu\n", offset, element_size, num_elements, capacity);
    return nullptr;
  }
  return ret_ptr;
}

void Simulator::add_task_dependencies_with_xfer(SimTask* src_task,
                                                SimTask* dst_task,
                                                size_t message_size)
{
  std::vector<CommDevice *> path = machine->get_comm_path(src_task->mem, dst_task->mem);
  // print the communication path
  // printf("Path from %s to %s is: ", src_task->mem->name.c_str(), dst_task->mem->name.c_str());
  // for (size_t i = 0; i < path.size(); i++) {
  //   printf("%s ", path[i]->name.c_str());
  // }
  // printf("\n");

  if (path.empty()) {
    src_task->add_next_task(dst_task);
    return;
  }
  assert(message_size > 0);
  std::vector<std::vector<SimTask *>> all_tasks;
  // Limit the max number of segments per message
  int seg_size = segment_size;
  int num_segment = message_size / seg_size;
  if (message_size % seg_size != 0) {
    num_segment += 1;
  }
  if (num_segment > max_num_segments) {
    num_segment = max_num_segments;
    seg_size = message_size / num_segment;
  }
  // Create all the comm tasks
  // Divide messages into segments
  for (size_t i = 0; i < path.size(); i++) {
    all_tasks.push_back({});
    for (int j = 0; j < num_segment; j++) {
      int cur_seg_size = seg_size;
      if (j == num_segment - 1) {
        cur_seg_size = message_size - (num_segment - 1) * seg_size;
      }
      std::string name = "seg " + std::to_string(j) + " from " + src_task->name + " to " + dst_task->name;
      SimTask *cur_task = task_manager->new_comm_task(name, path[i], cur_seg_size);
      all_tasks[i].push_back(cur_task);
    }
  }

  // Add dependencies among the comm tasks
  for (size_t i = 0; i < path.size(); i++) {
    for (int j = 0; j < num_segment; j++) {
      if (i == 0) {
        src_task->add_next_task(all_tasks[i][j]);
      }
      if (i == path.size() - 1) {
        all_tasks[i][j]->add_next_task(dst_task);
      }
      if (i > 0) {
        all_tasks[i-1][j]->add_next_task(all_tasks[i][j]);
      }
    }
  }

  // Add special dependencies for upi_ins, upi_outs, nic_ins, and nic_outs to prevent communication
  // overlap between upi_ins and upi_outs, and between nic_ins and nic_outs.
  if (num_segment > 1 and path.size() >= 2) {
    for (size_t i = 0; i < path.size(); i++) {
      for (int j = 1; j < num_segment; j++) {
        if (((CommDevice *)all_tasks[i][j]->device)->comm_type == CommDevice::NIC_OUT_COMM or
            ((CommDevice *)all_tasks[i][j]->device)->comm_type == CommDevice::UPI_OUT_COMM) {
          all_tasks[i+1][j-1]->add_next_task(all_tasks[i][j]);
        }
      }
    }
  }

  // call l1 optimizer's call back
  for (std::vector<SimTask *> & tv: all_tasks) {
    for (SimTask * t: tv) {
      if (l1optimizer) 
        l1optimizer->task_added(t);
    }
  }
}

void LogicalTaskgraphBasedSimulator::add_task_dependencies_with_xfer(
                                                SimTask* src_task,
                                                SimTask* dst_task,
                                                size_t message_size)
{
  std::vector<CommDevice *> path = machine->get_comm_path(src_task->mem, dst_task->mem);
#ifdef DEBUG_PRINT
  // print the communication path
  // printf("Path from %s to %s is: ", src_task->mem->name.c_str(), dst_task->mem->name.c_str());
  // for (size_t i = 0; i < path.size(); i++) {
  //   printf("%s ", path[i]->name.c_str());
  // }
  // printf("\n");
#endif

  if (path.empty()) {
    src_task->add_next_task(dst_task);
    return;
  }
  assert(message_size > 0);
  std::vector<SimTask*> final_tasks;
  for (CommDevice * d: path) {
    SimTask* task = task_manager->new_nominal_comm_task();
    // std::cerr << "add_task_dependency_with_xfer: " << task << std::endl;
    task->device = d;
    task->run_time = 0;
    task->xfer_size = message_size;
    task->xfer_left = message_size;
    if (!final_tasks.empty()) {
      final_tasks.back()->add_next_task(task);
    }
    final_tasks.push_back(task);
    if (l1optimizer) 
      l1optimizer->task_added(task);
  }
  src_task->add_next_task(final_tasks[0]);
  final_tasks.back()->add_next_task(dst_task);
}

[[noreturn]] void handle_measure_operator_cost_unimplemented(Op const *op) {
    std::cerr << "measure_operator_cost not implemented for op "
              << op->name
              << " (type " << op->op_type << ")"
              << ". Please report this issue to the FlexFlow developers."
              << std::endl;
    std::abort();
}

CostMetrics Simulator::measure_operator_cost(Op* op, const ParallelConfig& config)
{
  if (measurements != nullptr) {
    std::string key = op->get_name_structure() + ":" + config.get_pc_str();
    // std::cerr << "key: " << key << std::endl;
    return measurements->at(key);
  }

  size_t hash = 17 * 31 + (size_t)(op);
  hash = hash * 31 + std::hash<int>()(config.device_type);
  hash = hash * 31 + std::hash<int>()(config.nDims);
  for (int i = 0; i < config.nDims; i++)
    hash = hash * 31 + std::hash<int>()(config.dim[i]);
  std::map<size_t, CostMetrics>::const_iterator iter =
    hash_to_operator_cost.find(hash);
  if (iter == hash_to_operator_cost.end()) {
    CostMetrics cost_metrics;
    bool is_implemented = op->measure_operator_cost(this, config, cost_metrics);
    if (! is_implemented) {
      handle_measure_operator_cost_unimplemented(op);
    }
    hash_to_operator_cost[hash] = cost_metrics;
    return cost_metrics;
  } else {
    return iter->second;
  }
}

double Simulator::simulate_runtime(const FFModel* model,
                                  const std::map<Op*, ParallelConfig>& global,
                                  CompMode comp_mode)
{
  return this->simulate_runtime(model, global, comp_mode, "");
}

double Simulator::simulate_runtime(const FFModel* model,
                                  const std::map<Op*, ParallelConfig>& global,
                                  CompMode comp_mode,
                                  std::string const &export_file_name)
{
  // printf("%s\n", machine->to_string().c_str());
  task_manager->reset();
  if (l1optimizer)
    l1optimizer->reset();
  // Step 1: register forward and backward tasks
  for (size_t l = 0; l < model->layers.size(); l++) {
    Op* op = model->layers[l];
    ParallelConfig config = global.find(op)->second;
    CostMetrics cost_metrics = measure_operator_cost(op, config);
    double forward_time = cost_metrics.forward_time;
    double backward_time = cost_metrics.backward_time;
    for (int j = 0; j < config.num_parts(); j++) {
      SimTask* task1 = task_manager->new_forward_task(op, j);
      task1->device = machine->get_gpu(config.device_ids[j]);
      task1->mem = machine->get_gpu_fb_mem(config.device_ids[j]);
      task1->run_time = forward_time;
      if (l1optimizer) 
        l1optimizer->task_added(task1);
      if (comp_mode == COMP_MODE_TRAINING) {
        SimTask* task2 = task_manager->new_backward_task(op, j);
        task2->device = machine->get_gpu(config.device_ids[j]);
        task2->mem = machine->get_gpu_fb_mem(config.device_ids[j]);
        task2->run_time = backward_time;
        task1->add_next_task(task2);
        if (l1optimizer) 
          l1optimizer->task_added(task2);
      }
    }
  }
  // Step 2: insert dependencies and comm. tasks before compute tasks
  for (size_t l = 0; l < model->layers.size(); l++) {
    Op* op = model->layers[l];
    ParallelConfig config = global.find(op)->second;
    for (int j = 0; j < op->numInputs; j++) {
      Tensor t = op->inputs[j];
      Op* pre_op = t.owner_op;
      if (pre_op == NULL)
        continue;
      ParallelConfig pre_config = global.find(pre_op)->second;
      size_t element_size = data_type_size(t.data_type);
      for (int dstId = 0; dstId < config.num_parts(); dstId ++) {
        Domain dstR = op->get_input_tensor_shape(config, j, dstId);
        for (int srcId = 0; srcId < pre_config.num_parts(); srcId ++) {
          Domain srcR = pre_op->get_output_tensor_shape(pre_config, t.owner_idx, srcId);
          if (dstR.intersection(srcR).get_volume() > 0) {
            // Forward dependency
            {
              SimTask* dstT = task_manager->get_forward_task(op, dstId);
              SimTask* srcT = task_manager->get_forward_task(pre_op, srcId);
              add_task_dependencies_with_xfer(srcT, dstT, dstR.intersection(srcR).get_volume() * element_size);
            }
            // Backward dependency
            if (comp_mode == COMP_MODE_TRAINING) {
              SimTask* dstT = task_manager->get_backward_task(op, dstId);
              SimTask* srcT = task_manager->get_backward_task(pre_op, srcId);
              add_task_dependencies_with_xfer(dstT, srcT, dstR.intersection(srcR).get_volume() * element_size);
            }
          }
        }
      }
    }
  }
#ifdef FF_USE_NCCL
  // Do nothing since we will calculate NCCL cost at the end
  // fprintf(stderr, "USING NCCL\n");
#else
  // Step 2.5: add finals tasks for each compute device to capture the returning comm tasks
  // from parameter servers
  std::vector<SimTask*> finals;
  for (int d = 0; d < machine->get_num_gpus(); d++) {
    SimTask* t = task_manager->new_barrier_task();
    t->device = machine->get_gpu(d);
    t->mem = machine->get_gpu_fb_mem(d);
    t->run_time = 0;
    finals.push_back(t);
  }

  if (model->config.search_overlap_backward_update && comp_mode == COMP_MODE_TRAINING) {
    // Step 3a: consider backpropagation and weight update are overlapped
    for (int l = model->layers.size()-1; l >= 0; l--) {
      Op* op = model->layers[l];
      size_t element_size = data_type_size(DT_FLOAT); // assume all weights have double elements
      ParallelConfig pc = global.find(op)->second;
      for (int j = 0; j < op->numWeights; j++) {
        std::set<int> synched;
        for (int firstId = 0; firstId < pc.num_parts(); firstId++)
          if (synched.find(firstId) == synched.end()) {
            synched.insert(firstId);
            Domain firstR = op->get_weight_tensor_shape(pc, j, firstId);
            // Add a compute task for parameter update
            SimTask* updateT = task_manager->new_update_task();
            updateT->device = machine->get_gpu(pc.device_ids[firstId]);
            updateT->mem = machine->get_gpu_fb_mem(pc.device_ids[firstId]);
            // TODO add parameter synchronization time
            updateT->run_time = 0.0f; // Assume update task takes no time
            for (int nextId = firstId+1; nextId < pc.num_parts(); nextId++) {
              Domain nextR = op->get_weight_tensor_shape(pc, j, nextId);
              if (firstR.intersection(nextR).get_volume() > 0) {
                // Assert all or nothing:
                // The two weights must be fully overlapped or not at all
                assert(firstR == nextR);
                assert(synched.find(nextId) == synched.end());
                synched.insert(nextId);
                // Add comm. tasks from backT to updateT
                SimTask* backT = task_manager->get_backward_task(op, nextId);
                add_task_dependencies_with_xfer(backT, updateT, firstR.get_volume() * element_size);
                // Add comm. tasks from updateT to finalT
                SimTask* finalT = finals[backT->device->device_id];
                add_task_dependencies_with_xfer(updateT, finalT, firstR.get_volume() * element_size);
              }
            }
          }
      }
    }
  } else if (comp_mode == COMP_MODE_TRAINING) {
    // Step 3b: Bulk Synchronous Model
    // Add a per-device barrier before weight update
    std::vector<SimTask*> barriers;
    for (int d = 0; d < machine->get_num_gpus(); d++) {
      SimTask* t = task_manager->new_barrier_task();
      t->device = machine->get_gpu(d);
      t->mem = machine->get_gpu_fb_mem(d);
      t->run_time = 0;
      barriers.push_back(t);
    }
    for (size_t l = 0; l < model->layers.size(); l++) {
      Op* op = model->layers[l];
      ParallelConfig pc = global.find(op)->second;
      for (int j = 0; j < pc.num_parts(); j++) {
        SimTask* backT = task_manager->get_backward_task(op, j);
        backT->add_next_task(barriers[backT->device->device_id]);
      }
    }
    for (size_t l = 0; l < model->layers.size(); l++) {
      Op* op = model->layers[l];
      ParallelConfig pc = global.find(op)->second;
      size_t element_size = data_type_size(DT_FLOAT); // assume all weights have double elements
      for (int j = 0; j < op->numWeights; j++) {
        std::set<int> synched;
        for (int firstId = 0; firstId < pc.num_parts(); firstId++)
          if (synched.find(firstId) == synched.end()) {
            synched.insert(firstId);
            Domain firstR = op->get_weight_tensor_shape(pc, j, firstId);
            // Add a compute task for parameter update
            SimTask* updateT = task_manager->new_update_task();
            updateT->device = machine->get_gpu(pc.device_ids[firstId]);
            updateT->mem = machine->get_gpu_fb_mem(pc.device_ids[firstId]);
            updateT->run_time = 0.0f; // Assume update task takes no time
            barriers[updateT->device->device_id]->add_next_task(updateT);
            for (int nextId = firstId+1; nextId < pc.num_parts(); nextId++) {
              Domain nextR = op->get_weight_tensor_shape(pc, j, nextId);
              if (firstR.intersection(nextR).get_volume() > 0) {
                // Assert all or nothing:
                // The two weights must be fully overlapped or not at all
                assert(firstR == nextR);
                assert(synched.find(nextId) == synched.end());
                synched.insert(nextId);
                SimTask* backT = task_manager->get_backward_task(op, nextId);
                assert(backT->device->device_id == pc.device_ids[nextId]);
                SimTask* barrierT = barriers[backT->device->device_id];
                // Add comm. tasks from barrierT to updateT
                add_task_dependencies_with_xfer(barrierT, updateT, firstR.get_volume() * element_size);
                // Add comm. tasks from updateT to finalT
                SimTask* finalT = finals[backT->device->device_id];
                add_task_dependencies_with_xfer(updateT, finalT, firstR.get_volume() * element_size);
              }
            }
          }
      }
    }
  } else {
    assert(comp_mode == COMP_MODE_INFERENCE);
  }
#endif
  // Step 4: add ready tasks into ready_queue
  std::priority_queue<SimTask*, std::vector<SimTask*>, SimTaskCompare> ready_queue;
  for (size_t i = 0; i < task_manager->global_task_id; i++)
    if (task_manager->tasks[i]->counter == 0)
      ready_queue.push(task_manager->tasks[i]);
  // Step 5: perform simulation
  double sim_time = 0.0f;
  std::map<Device*, double> device_times;
  size_t idx = 0;
  DotFile<SimTask *> taskGraph;
  bool export_taskgraph = (export_file_name != "");
  if (export_taskgraph) {
    taskGraph.set_filename(export_file_name);
  }
  while (!ready_queue.empty()) {
    // Find the task with the earliest start time
    SimTask* cur_task = ready_queue.top();
    ready_queue.pop();
    double ready_time = 0;
    if (device_times.find(cur_task->device) != device_times.end()) {
      ready_time = device_times[cur_task->device];
    }
    double start_time = std::max(ready_time, cur_task->ready_time);
    double end_time = start_time + cur_task->run_time;
    device_times[cur_task->device] = end_time;
    if (export_taskgraph) {
      std::map<std::string, std::string> nodeAttrs;
      std::ostringstream label;
      label << "\"{ ";
      if (!(cur_task->name).empty()) {
        label << cur_task->name << " | ";
      }
      label << cur_task->get_type_str() << " | ";
      label << "{ " << start_time << " | " << end_time << " }";
      label << " }\"";
      nodeAttrs["label"] = label.str();
      nodeAttrs["shape"] = "record";
      taskGraph.add_node(cur_task, nodeAttrs);
    }
  #ifdef DEBUG_PRINT
    // printf("task[%lu] type(%d) run_time(%.4lf) ready_time(%.4lf) start_time(%.4lf) device(%s)\n",
    //       idx, cur_task->type, cur_task->run_time, ready_time, start_time, (cur_task->device->name).c_str());
  #endif
    if (end_time > sim_time)
      sim_time = end_time;
    for (size_t i = 0; i < cur_task->next_tasks.size(); i++) {
      SimTask* next = cur_task->next_tasks[i];
      if (export_taskgraph) {
        taskGraph.add_edge(cur_task, next);
      }
      next->ready_time = std::max(next->ready_time, end_time);
      next->counter --;
      if (next->counter == 0) {
        ready_queue.push(next);
      }
    }
    idx++;
  }
  if (export_taskgraph) {
    taskGraph.close();
  }
  // Assert all tasks were processed
  assert(idx == task_manager->global_task_id);
#ifdef FF_USE_NCCL
  if (comp_mode == COMP_MODE_TRAINING) {
    for (size_t l = 0; l < model->layers.size(); l++) {
      Op* op = model->layers[l];
      size_t element_size = data_type_size(DT_FLOAT); // assume all weights have double elements
      ParallelConfig pc = global.find(op)->second;
      // Since all NCCL calls are blocking, we can add the NCCL cost
      // sequentially 
      for (int j = 0; j < op->numWeights; j++) {
        std::set<int> synched;
        for (int firstId = 0; firstId < pc.num_parts(); firstId++)
          if (synched.find(firstId) == synched.end()) {
            synched.insert(firstId);
            Domain firstR = op->get_weight_tensor_shape(pc, j, firstId);
            Device* firstDevice = machine->get_gpu(pc.device_ids[firstId]);
            double nccl_time = 0.0f;
            for (int nextId = firstId+1; nextId < pc.num_parts(); nextId++) {
              Domain nextR = op->get_weight_tensor_shape(pc, j, nextId);
              if (firstR.intersection(nextR).get_volume() > 0) {
                // Assert all or nothing:
                // The two weights must be fully overlapped or not at all
                assert(firstR == nextR);
                assert(synched.find(nextId) == synched.end());
                synched.insert(nextId);
                Device* nextDevice = machine->get_gpu(pc.device_ids[nextId]);
                // Compute the bandwidth between firstDevice/nextDevice
                double bandwidth = 0.0f;
                if (firstDevice->node_id == nextDevice->node_id) {
                  bandwidth = machine->get_intra_node_gpu_bandwidth();
                } else {
                  bandwidth = machine->get_inter_node_gpu_bandwidth();
                }
                nccl_time = std::max(nccl_time, (double)firstR.get_volume() * element_size / bandwidth);
              }
            }
            // Add ncclTime to sim_time given nccl calls are blocking
            sim_time += nccl_time;
          }
      }
    }
  } else {
    assert(comp_mode == COMP_MODE_INFERENCE);
  }
#endif
  // Step 6: add penalty to strategies that exceed the memory limits on devices
  std::vector<size_t> gpu_mem_usage(machine->get_num_gpus(), 0);
  double memory_penalty = 0.0f;
  for (size_t l = 0; l < model->layers.size(); l++) {
    Op* op = model->layers[l];
    ParallelConfig config = global.find(op)->second;
    CostMetrics cost_metrics = measure_operator_cost(op, config);
    size_t memory_requirement = cost_metrics.memory_requirement;
    for (int j = 0; j < config.num_parts(); j++) {
      gpu_mem_usage[config.device_ids[j]] += memory_requirement;
    }
  }
  if (export_file_name != "") {  
    for (int i = 0; i < machine->get_num_gpus(); i++) {
        printf("Before penalty, dev id %d, usage %zu \n", i, gpu_mem_usage[i]); 
    }
  }
  // Penalize the total runtiem by 1ms if we exceed the memory budget by 1MB
  for (int i = 0; i < machine->get_num_gpus(); i++) {
    MemDevice* gpu_fb_mem = machine->get_gpu_fb_mem(i);
    if (gpu_mem_usage[i] > gpu_fb_mem->capacity and gpu_fb_mem->capacity >= 0)
      memory_penalty += (gpu_mem_usage[i] - gpu_fb_mem->capacity) * 1e-6;
  }
  //if (memory_penalty > 0.0f)
  //  printf("Memory penalty = %.4lf ms\n", memory_penalty);
  return sim_time + memory_penalty;
}

#ifdef WRITE_NETWORK_TRANSFER
static std::ofstream network_transfer_log;
#endif 

double LogicalTaskgraphBasedSimulator::simulate_runtime(
                                  const FFModel* model,
                                  const std::map<Op*, ParallelConfig>& global,
                                  CompMode comp_mode,
                                  std::string const &export_file_name) 
{
#ifdef WRITE_NETWORK_TRANSFER
  network_transfer_log.open("network.log");
#endif
  // printf("%s\n", machine->to_string().c_str());
  task_manager->reset();
  if (l1optimizer)
    l1optimizer->reset();
  std::unordered_map<SimTask*, Op*> task_to_op;
  // Step 1: register forward and backward tasks
  for (size_t l = 0; l < model->layers.size(); l++) {
    Op* op = model->layers[l];
    ParallelConfig config = global.find(op)->second;
    CostMetrics cost_metrics = measure_operator_cost(op, config);
    double forward_time = cost_metrics.forward_time;
    double backward_time = cost_metrics.backward_time;
    SimTask *ar_task = nullptr;
    for (int j = 0; j < config.num_parts(); j++) {
      SimTask* task1 = task_manager->new_forward_task(op, j);
      task_to_op[task1] = op;
      task1->device = machine->get_gpu(config.device_ids[j]);
      task1->mem = machine->get_gpu_fb_mem(config.device_ids[j]);
      task1->run_time = forward_time;
      if (l1optimizer) 
        l1optimizer->task_added(task1);
      if (comp_mode == COMP_MODE_TRAINING) {
        SimTask* task2 = task_manager->new_backward_task(op, j);
        task_to_op[task2] = op;
        task2->device = machine->get_gpu(config.device_ids[j]);
        task2->mem = machine->get_gpu_fb_mem(config.device_ids[j]);
        task2->run_time = backward_time;
        task1->add_next_task(task2);
        if (l1optimizer) 
          l1optimizer->task_added(task2);
        
      }
    }
  }

  std::set<SimTask*> ars;
  for (size_t l = 0; l < model->layers.size(); l++) {
    Op* op = model->layers[l];
    ParallelConfig config = global.find(op)->second;
    size_t element_size = data_type_size(DT_FLOAT);
    // NER step: add allreduce task after backward propogation
    for (int j = 0; j < op->numWeights; j++) {
      std::set<int> synched;
      std::vector<int> node_ids;
      for (int firstId = 0; firstId < config.num_parts(); firstId++) {
        if (synched.find(firstId) == synched.end()) {
          synched.insert(firstId);
          Domain firstR = op->get_weight_tensor_shape(config, j, firstId);
          size_t xfer_size = firstR.get_volume() * element_size;
          node_ids.push_back(config.device_ids[firstId]);
          for (int nextId = firstId+1; nextId < config.num_parts(); nextId++) {
            Domain nextR = op->get_weight_tensor_shape(config, j, nextId);
            if (firstR.intersection(nextR).get_volume() > 0) {
              // Assert all or nothing:
              // The two weights must be fully overlapped or not at all
              assert(firstR == nextR);
              assert(synched.find(nextId) == synched.end());
              synched.insert(nextId);
              node_ids.push_back(config.device_ids[nextId]);
            }
          }
          
          SimTask* ar_task = task_manager->new_allreduce_task(op, node_ids, xfer_size);
          task_to_op[ar_task] = op;
          ars.insert(ar_task);
          if (l1optimizer) 
            l1optimizer->task_added(ar_task);
          for (int dstId = 0; dstId < config.num_parts(); dstId ++) {
            task_manager->get_backward_task(op, dstId)->add_next_task(ar_task);
          }
        }
      
      }
    }
  }
          /*
  {
    for (size_t l = 0; l < model->layers.size(); l++) {
      Op* op0 = model->layers[l];
      ParallelConfig config0 = global.find(op0)->second;
      for (int dstId = 0; dstId < config0.num_parts(); dstId ++) {
        for (SimTask* t: ars) {
          task_manager->get_backward_task(op0, dstId)->add_next_task(t);
        }
      }
    }
  }
          */
        

  // Step 2: insert dependencies and comm. tasks before compute tasks
  for (size_t l = 0; l < model->layers.size(); l++) {
    Op* op = model->layers[l];
    ParallelConfig config = global.find(op)->second;
    for (int j = 0; j < op->numInputs; j++) {
      Tensor t = op->inputs[j];
      Op* pre_op = t.owner_op;
      if (pre_op == NULL)
        continue;
      ParallelConfig pre_config = global.find(pre_op)->second;
      size_t element_size = data_type_size(t.data_type);
      for (int dstId = 0; dstId < config.num_parts(); dstId ++) {
        Domain dstR = op->get_input_tensor_shape(config, j, dstId);
        for (int srcId = 0; srcId < pre_config.num_parts(); srcId ++) {
          Domain srcR = pre_op->get_output_tensor_shape(pre_config, t.owner_idx, srcId);
          if (dstR.intersection(srcR).get_volume() > 0) {
            // Forward dependency
            {
              SimTask* dstT = task_manager->get_forward_task(op, dstId);
              SimTask* srcT = task_manager->get_forward_task(pre_op, srcId);
              add_task_dependencies_with_xfer(srcT, dstT, dstR.intersection(srcR).get_volume() * element_size);
            }
            // Backward dependency
            if (comp_mode == COMP_MODE_TRAINING) {
              SimTask* dstT = task_manager->get_backward_task(op, dstId);
              SimTask* srcT = task_manager->get_backward_task(pre_op, srcId);
              add_task_dependencies_with_xfer(dstT, srcT, dstR.intersection(srcR).get_volume() * element_size);
            }
          }
        }
      }
    }
  }
  
  // Step 4: add ready tasks into ready_queue
  std::priority_queue<SimTask*, std::vector<SimTask*>, SimTaskCompare> ready_queue;
  for (size_t i = 0; i < task_manager->global_task_id; i++)
    if (task_manager->tasks[i]->counter == 0)
      ready_queue.push(task_manager->tasks[i]);

  // Step 5: perform simulation

  double sim_time = 0.0f;
  std::map<Device*, double> device_times;
  // map<Device*, SimTask*> device_schedule;
  size_t idx = 0;
  while (!ready_queue.empty()) {
    // Find the task with the earliest start time
    SimTask* cur_task = ready_queue.top();
    ready_queue.pop();
    double ready_time = 0;
    double end_time;
    if (device_times.find(cur_task->device) != device_times.end()) {
      ready_time = device_times[cur_task->device];
    }
    double start_time = std::max(ready_time, cur_task->ready_time);
    if (cur_task->type == SimTask::TASK_NOMINAL_COMM) {
      if (!segment_transfer)
        end_time = route_transfer(cur_task, start_time, device_times);
      else {
        bool finished;
        end_time = route_transfer_seg(cur_task, start_time, device_times, finished);
        if (!finished) {
          ready_queue.push(cur_task);
          continue;
        }
      }
    }
    else if (cur_task->type == SimTask::TASK_ALLREDUCE) {
      if (model->config.big_gpu > 0 && task_to_op[cur_task]->op_type != OperatorType::OP_EMBEDDING) {
        double internal_ar_time = compute_internal_ar_time(model, cur_task);
        cur_task->ready_time += internal_ar_time;
        cur_task->run_time = internal_ar_time;
        start_time += internal_ar_time;
      }
      expand_allreduce(cur_task, start_time, ready_queue);
      idx++;
      continue;
    }
    else {
      end_time = start_time + cur_task->run_time;
      device_times[cur_task->device] = end_time;
    }

#ifdef DEBUG_PRINT
    printf("task[%lu/%lu] type(%d) run_time(%.4lf) ready_time(%.4lf) start_time(%.4lf) device(%s)\n",
          idx, task_manager->global_task_id, cur_task->type, cur_task->run_time, ready_time, start_time, (cur_task->device->name).c_str());
#endif

    if (end_time > sim_time) {
      sim_time = end_time;
    }

    for (size_t i = 0; i < cur_task->next_tasks.size(); i++) {
      SimTask* next = cur_task->next_tasks[i];
      // next->ready_time = max(next->ready_time, end_time);
      if (end_time > next->ready_time) {
        next->ready_time = end_time;
        // next->prev = t;
      }
      next->counter--;
      if (next->counter == 0) {
        ready_queue.push(next);
      }
    }
    idx++;
  }
  assert(idx == task_manager->global_task_id);
  
  // Step 6: add penalty to strategies that exceed the memory limits on devices
  // std::vector<size_t> gpu_mem_usage(machine->get_num_gpus(), 0);
  // double memory_penalty = 0.0f;
  // for (size_t l = 0; l < model->layers.size(); l++) {
  //   Op* op = model->layers[l];
  //   ParallelConfig config = global.find(op)->second;
  //   CostMetrics cost_metrics = measure_operator_cost(op, config);
  //   size_t memory_requirement = cost_metrics.memory_requirement;
  //   for (int j = 0; j < config.num_parts(); j++) {
  //     gpu_mem_usage[config.device_ids[j]] += memory_requirement;
  //   }
  // }
  // if (export_file_name != "") {  
  //   for (int i = 0; i < machine->get_num_gpus(); i++) {
  //       printf("Before penalty, dev id %d, usage %zu \n", i, gpu_mem_usage[i]); 
  //   }
  // }
  // // Penalize the total runtiem by 1ms if we exceed the memory budget by 1MB
  // for (int i = 0; i < machine->get_num_gpus(); i++) {
  //   MemDevice* gpu_fb_mem = machine->get_gpu_fb_mem(i);
  //   if (gpu_mem_usage[i] > gpu_fb_mem->capacity and gpu_fb_mem->capacity >= 0)
  //     memory_penalty += (gpu_mem_usage[i] - gpu_fb_mem->capacity) * 1e-6;
  // }
  //if (memory_penalty > 0.0f)
  //  printf("Memory penalty = %.4lf ms\n", memory_penalty);
  if (export_file_name != "") {
    searlize_logical_taskgraph(model, export_file_name);
  }
#ifdef WRITE_NETWORK_TRANSFER
  network_transfer_log.close();
#endif

  return sim_time;//  + memory_penalty;
      
}

double LogicalTaskgraphBasedSimulator::simulate_runtime(const FFModel* model,
                                  const std::map<Op*, ParallelConfig>& global,
                                  CompMode comp_mode)
{
  return this->simulate_runtime(model, global, comp_mode, "");
}


double LogicalTaskgraphBasedSimulator::route_transfer(SimTask * transfer_task, 
                              double start_time,
                              std::map<Device*, double> &device_times) {
  std::vector<CommDevice *> route = 
    static_cast<NominalCommDevice*>(transfer_task->device)->expand_to_physical();

  double curr_task_start_time; 
  double curr_task_finish_time; 
  double curr_task_run_time = 0; 
  double curr_task_ready_time = transfer_task->ready_time; 
  double xfer_size = transfer_task->xfer_size;

  double final_start_time = 0;
  double final_finish_time = 0;

  SimTask * info_holder = new SimTask();
  info_holder->type = SimTask::TASK_COMM;

  for (unsigned int i = 0; i < route.size(); i++) {
    CommDevice * latency_task_device = route[i];
    if (device_times.find(latency_task_device) == device_times.end()) device_times[latency_task_device] = 0;
    double latency_task_run_time = machine->get_inter_node_gpu_latency();
    double latency_task_ready_time; 
    double latency_task_start_time; 
    if (i == 0) {
      latency_task_ready_time = curr_task_ready_time + curr_task_run_time;
      latency_task_start_time = std::max(device_times[latency_task_device], latency_task_ready_time);
      final_start_time = latency_task_start_time;
    }
    else {
      latency_task_ready_time = curr_task_finish_time;
      latency_task_start_time = std::max(device_times[latency_task_device], latency_task_ready_time);
    }
    double latency_task_finish_time = latency_task_start_time + latency_task_run_time;
    device_times[latency_task_device] = latency_task_finish_time;
    double dram_to_dram_run_time = xfer_size / latency_task_device->bandwidth;

    double dram_to_dram_start_time = latency_task_finish_time;
    double dram_to_dram_finish_time = dram_to_dram_start_time + dram_to_dram_run_time;
    device_times[latency_task_device] = dram_to_dram_finish_time;

    if (dram_to_dram_finish_time > final_finish_time) {
      final_finish_time = dram_to_dram_finish_time;
    }

    curr_task_ready_time = latency_task_ready_time;
    curr_task_start_time = latency_task_start_time;
    curr_task_finish_time = latency_task_finish_time;
    curr_task_run_time = latency_task_run_time;
    
#ifdef DEBUG_PRINT
    printf("\texpand: route[%u] run_time(%.4lf) ready_time(%.4lf) start_time(%.4lf) device(%s)\n",
          i, curr_task_run_time, curr_task_ready_time, curr_task_start_time, (latency_task_device->name).c_str());
    printf("\t\td2d: run_time(%.4lf) start_time(%.4lf) device(%s)\n",
          dram_to_dram_run_time, dram_to_dram_start_time, (latency_task_device->name).c_str());
#endif

    info_holder->device = latency_task_device;
    info_holder->run_time = dram_to_dram_run_time;
    info_holder->xfer_size = xfer_size;
    // info_holder->from_dev = CommDevice::get_from_dev(latency_task_device->device_id, mac);
    // info_holder->to_dev = 

    if (l1optimizer) 
      l1optimizer->task_added(info_holder);

  }
  delete info_holder;

#ifdef WRITE_NETWORK_TRANSFER
  auto * nw = static_cast<NominalCommDevice*>(transfer_task->device);
  network_transfer_log << nw->device_id / machine->get_total_devs() << ", "
                       << nw->device_id % machine->get_total_devs() << ", "
                       << xfer_size << ", "
                       << final_start_time << ", "
                       << final_finish_time << std::endl;
#endif
  

  transfer_task->run_time = final_finish_time - final_start_time;
  return final_finish_time;
}

double LogicalTaskgraphBasedSimulator::compute_internal_ar_time(const FFModel* model, SimTask * allreduce_task) 
{
  assert(model->config.big_gpu > 0);
  // std::cerr << "internal_ar_time: " << (2 * allreduce_task->xfer_size / model->config.big_gpu / model->config.inter_gpu_bandwidth) << " for size " << allreduce_task->xfer_size << " inter_gpu_bandwidth " << model->config.inter_gpu_bandwidth << std::endl;
  return (2 * allreduce_task->xfer_size / model->config.big_gpu / model->config.inter_gpu_bandwidth);
}

double LogicalTaskgraphBasedSimulator::route_transfer_seg(SimTask * transfer_task, 
                            double start_time,
                            std::map<Device*, double> &device_times,
                            bool & finished)
{
  std::vector<CommDevice *> route = 
    static_cast<NominalCommDevice*>(transfer_task->device)->expand_to_physical();

  double curr_task_start_time; 
  double curr_task_finish_time; 
  double curr_task_run_time = 0; 
  double curr_task_ready_time = transfer_task->ready_time; 
  double xfer_size = transfer_task->xfer_left > segment_size ? segment_size : transfer_task->xfer_left;
  // xfer_size /= route.size();
  transfer_task->xfer_left = transfer_task->xfer_left > segment_size ? transfer_task->xfer_left - segment_size : 0;
  finished = transfer_task->xfer_left == 0; 
// #ifdef DEBUG_PRINT
  // std::cerr << "xfer_total: " << transfer_task->xfer_size << ", xfer_left: " << transfer_task->xfer_left << " finished:" << finished << std::endl;
// #endif

  double final_start_time = 0;
  double final_finish_time = 0;
  double final_first_seg_finish_time = 0;

  SimTask * info_holder = new SimTask();
  info_holder->type = SimTask::TASK_COMM;

  for (unsigned int i = 0; i < route.size(); i++) {
    CommDevice * latency_task_device = route[i];
    if (device_times.find(latency_task_device) == device_times.end()) device_times[latency_task_device] = 0;
    double latency_task_run_time = machine->get_inter_node_gpu_latency();
    double latency_task_ready_time; 
    double latency_task_start_time; 
    if (i == 0) {
      latency_task_ready_time = curr_task_ready_time + curr_task_run_time;
      latency_task_start_time = std::max(device_times[latency_task_device], latency_task_ready_time);
      final_start_time = latency_task_start_time;
    }
    else {
      latency_task_ready_time = curr_task_finish_time;
      latency_task_start_time = std::max(device_times[latency_task_device], latency_task_ready_time);
    }
    double latency_task_finish_time = latency_task_start_time + latency_task_run_time;
    device_times[latency_task_device] = latency_task_finish_time;
    double dram_to_dram_run_time = xfer_size / latency_task_device->bandwidth;
    // std::cerr << "latency_task_device->bandwidth: " << latency_task_device->bandwidth << std::endl;
    // std::cerr << "d2drt: " << dram_to_dram_run_time << std::endl;

    double dram_to_dram_start_time = latency_task_finish_time;
    double dram_to_dram_finish_time = dram_to_dram_start_time + dram_to_dram_run_time;
    if (i == 0) {
      final_first_seg_finish_time = dram_to_dram_finish_time;
    }
    device_times[latency_task_device] = dram_to_dram_finish_time;

    if (dram_to_dram_finish_time > final_finish_time) {
      final_finish_time = dram_to_dram_finish_time;
    }

    curr_task_ready_time = latency_task_ready_time;
    curr_task_start_time = latency_task_start_time;
    curr_task_finish_time = latency_task_finish_time;
    curr_task_run_time = latency_task_run_time;
    
#ifdef DEBUG_PRINT
    printf("\texpand: route[%u] run_time(%.4lf) ready_time(%.4lf) start_time(%.4lf) device(%s)\n",
          i, curr_task_run_time, curr_task_ready_time, curr_task_start_time, (latency_task_device->name).c_str());
    printf("\t\td2d: run_time(%.4lf) start_time(%.4lf) device(%s)\n",
          dram_to_dram_run_time, dram_to_dram_start_time, (latency_task_device->name).c_str());
#endif
    info_holder->device = latency_task_device;
    info_holder->run_time = dram_to_dram_run_time;
    info_holder->xfer_size = xfer_size;
    if (l1optimizer) 
      l1optimizer->task_added(info_holder);
  }
  delete info_holder;

#ifdef WRITE_NETWORK_TRANSFER
  auto * nw = static_cast<NominalCommDevice*>(transfer_task->device);
  network_transfer_log << nw->device_id / machine->get_total_devs() << ", "
                       << nw->device_id % machine->get_total_devs() << ", "
                       << xfer_size << ", "
                       << final_start_time << ", "
                       << final_finish_time << std::endl;
#endif
  if (!finished) {
#ifdef DEBUG_PRINT
    std::cerr << "ready time: " << transfer_task->ready_time << " to " << final_first_seg_finish_time << std::endl;
#endif
    transfer_task->ready_time = final_first_seg_finish_time;

  }

  transfer_task->run_time = final_finish_time - final_start_time;
  return final_finish_time;
}

void LogicalTaskgraphBasedSimulator::expand_allreduce(SimTask * allreduce_task,
                                 double start_time,
                                 std::priority_queue<SimTask*, std::vector<SimTask*>, SimTaskCompare>& ready_queue) {

  int n_participants = allreduce_task->next_tasks.size();
  if (n_participants == 1) return;
  
  SimTask * final_task = new_update_task_unrecorded();

#ifdef FF_USE_NCCL
  // recall that next_task stores node group in this case
  final_task->device = machine->get_gpu(reinterpret_cast<uint64_t>(allreduce_task->next_tasks[0]));
  MemDevice * src_mem = machine->get_gpu_fb_mem(reinterpret_cast<uint64_t>(allreduce_task->next_tasks[0]));
  MemDevice * dst_mem;
  // std::cerr << "expand_ar size: " << allreduce_task->xfer_size << ", " << "grp size: " << n_participants << std::endl;

  int dir = std_uniform(gen) < 0.5 ? 1 : -1;
  // std::cerr << "dir: " << dir << std::endl;
  int round = 0, i = 0;
  // for (int i = 0; i < n_participants; i++) {
  while (round != n_participants) {
    dst_mem = machine->get_gpu_fb_mem(reinterpret_cast<uint64_t>(allreduce_task->next_tasks[MOD(i+dir,n_participants)]));
    // dst_mem = machine->get_gpu_fb_mem(reinterpret_cast<uint64_t>(allreduce_task->next_tasks[(i+1)%n_participants]));
    std::vector<CommDevice *> path = machine->get_comm_path(src_mem, dst_mem);
    // if (dir)
    // std::vector<CommDevice *> path = machine->get_comm_path(src_mem, dst_mem);
    // std::cerr << "\tDevices: ";
    for (CommDevice * d: path) {
      SimTask* task = new_comm_task_unrecorded();
      task->device = d;
      // std::cerr << "dir: " << dir << ", " << d->name << ", ";
      task->run_time = 0;
      task->ready_time = allreduce_task->ready_time;
      task->xfer_size = (2.0 * (n_participants-1)) * allreduce_task->xfer_size / n_participants;
      task->xfer_left = task->xfer_size;
      task->add_next_task(final_task);
      ready_queue.push(task);
      if (l1optimizer)
        l1optimizer->task_added(task);
    }
    // std::cerr << std::endl;
    src_mem = dst_mem;
    round++;
    i += dir;
  }
  if (final_task->counter == 0) {
    final_task->ready_time = allreduce_task->ready_time;
    ready_queue.push(final_task);
  }
#else
  // assume parameter server in this case
  MemDevice * leader_mem = machine->get_gpu_fb_mem(reinterpret_cast<uint64_t>(allreduce_task->next_tasks[0]));
  MemDevice * worker_mem;
  SimTask * ps_update_task = new_update_task_unrecorded();
  ps_update_task->device = machine->get_gpu(reinterpret_cast<uint64_t>(allreduce_task->next_tasks[0]));
  final_task->device = machine->get_gpu(reinterpret_cast<uint64_t>(allreduce_task->next_tasks[0]));
  ps_update_task->add_next_task(final_task);

  // ps gather
  for (int i = 0; i < n_participants; i++) {
    worker_mem = machine->get_gpu_fb_mem(reinterpret_cast<uint64_t>(allreduce_task->next_tasks[i]));
    std::vector<CommDevice *> path = machine->get_comm_path(worker_mem, leader_mem);
    for (CommDevice * d: path) {
      SimTask* task = new_comm_task_unrecorded();
      task->device = d;
      task->run_time = 0;
      task->ready_time = allreduce_task->ready_time;
      task->xfer_size = allreduce_task->xfer_size;
      task->xfer_left = task->xfer_size;
      task->add_next_task(ps_update_task);
      ready_queue.push(task);
      if (l1optimizer)
        l1optimizer->task_added(task);
    }
  }

  // scatter
  for (int i = 0; i < n_participants; i++) {
    worker_mem = machine->get_gpu_fb_mem(reinterpret_cast<uint64_t>(allreduce_task->next_tasks[i]));
    std::vector<CommDevice *> path = machine->get_comm_path(leader_mem, worker_mem);
    for (CommDevice * d: path) {
      SimTask* task = new_comm_task_unrecorded();
      task->device = d;
      task->run_time = 0;
      task->ready_time = allreduce_task->ready_time;
      task->xfer_size = allreduce_task->xfer_size;
      ps_update_task->add_next_task(task);
      task->add_next_task(final_task);
      if (l1optimizer)
        l1optimizer->task_added(task);
    }
  }

  if (ps_update_task->counter == 0) {
    assert(final_task->counter == 1);
    ps_update_task->ready_time = allreduce_task->ready_time;
    ready_queue.push(ps_update_task);
  }

#endif

}

SimTask* LogicalTaskgraphBasedSimulator::new_comm_task_unrecorded() {
  SimTask* task = task_manager->new_task();
  task->type = SimTask::TASK_NOMINAL_COMM;
  task->store = false;
  return task;
}

SimTask* LogicalTaskgraphBasedSimulator::new_update_task_unrecorded() {
  SimTask* task = task_manager->new_task();
  task->type = SimTask::TASK_UPDATE;
  task->store = false;
  return task;
}

bool LogicalTaskgraphBasedSimulator::searlize_logical_taskgraph(const FFModel* model, std::string const &export_file_name) {
  flatbuffers::FlatBufferBuilder builder(262144);
  get_taskgraph_flatbuf(model, builder);
  std::ofstream ofs(export_file_name, std::ofstream::binary);
  if (!ofs.is_open()) return false;
  ofs.write((const char *) builder.GetBufferPointer(), (size_t)builder.GetSize());
  return !ofs.bad();
  // flatbuffers::SaveFile(export_file_name.c_str(),
  //                       (const char *) builder.GetBufferPointer(),
  //                       (size_t) builder.GetSize(), true);
  // return;
}

void LogicalTaskgraphBasedSimulator::get_taskgraph_flatbuf(const FFModel* model, flatbuffers::FlatBufferBuilder &builder) 
{
  builder.Clear();

  // Store topology
  // flatbuffers::FlatBufferBuilder builder = flatbuffers::FlatBufferBuilder();
  NetworkedMachineModel *nm = static_cast<NetworkedMachineModel*>(machine);
  size_t total_devs = nm->get_total_devs();
  std::vector<flatbuffers::Offset<FlatBufTaskGraph::Connection>> conns_v = 
    std::vector<flatbuffers::Offset<FlatBufTaskGraph::Connection>>();
  for (size_t i = 0; i < nm->get_total_devs(); i++) {
    for (size_t j = 0; j < i; j++) {
      size_t nlink;
      if ((nlink = nm->get_conn_matrix()[i * total_devs + j]) > 0) {
        conns_v.emplace_back(FlatBufTaskGraph::CreateConnection(builder, i, j, nlink));
      }
    }
  }
  auto conns = builder.CreateVector(conns_v);

  // store operators
  // builder.Clear();
  std::vector<flatbuffers::Offset<FlatBufTaskGraph::Operator>> op_v = 
    std::vector<flatbuffers::Offset<FlatBufTaskGraph::Operator>>();
  for (size_t l = 0; l < model->layers.size(); l++) {
    Op* op = model->layers[l];
    auto opname = builder.CreateString(op->name);
    op_v.emplace_back(FlatBufTaskGraph::CreateOperator(builder, 
      reinterpret_cast<uint64_t>(op), (int)op->op_type, opname));
  }
  auto ops = builder.CreateVector(op_v);

  // store tasks
  // builder.Clear();
  std::vector<flatbuffers::Offset<FlatBufTaskGraph::Task>> task_v = 
    std::vector<flatbuffers::Offset<FlatBufTaskGraph::Task>>();
  // change: since there is no universal storage of device, creat a set of
  // all devices for the next entry
  std::unordered_set<Device *> devices;
  for (size_t i = 0; i < task_manager->global_task_id; i++) {
    SimTask * curr = task_manager->tasks[i];
    if (curr->store) {
      FlatBufTaskGraph::SimTaskType tasktype;
      uint64_t taskid = reinterpret_cast<uint64_t>(curr);
      std::vector<uint64_t> nexttasks = std::vector<uint64_t>();
      for (SimTask *t: curr->next_tasks) {
        nexttasks.push_back(reinterpret_cast<uint64_t>(t));
      }
      auto ntv = builder.CreateVector(nexttasks);
      switch (curr->type) {
      case SimTask::TASK_FORWARD:
        tasktype = FlatBufTaskGraph::SimTaskType_TASK_FORWARD;
      break;
      case SimTask::TASK_BACKWARD:
        tasktype = FlatBufTaskGraph::SimTaskType_TASK_BACKWARD;
      break;
      case SimTask::TASK_UPDATE:
        tasktype = FlatBufTaskGraph::SimTaskType_TASK_UPDATE;
      break;
      case SimTask::TASK_BARRIER:
        tasktype = FlatBufTaskGraph::SimTaskType_TASK_BARRIER;
      break;
      case SimTask::TASK_COMM:
        assert("Logical task graph shouldn't contain TASK_COMM!" && false);
      break;
      case SimTask::TASK_NOMINAL_COMM:
        tasktype = FlatBufTaskGraph::SimTaskType_TASK_NOMINAL_COMM;
      break;
      case SimTask::TASK_ALLREDUCE:
        tasktype = FlatBufTaskGraph::SimTaskType_TASK_ALLREDUCE;
      break;
      }
      task_v.emplace_back(FlatBufTaskGraph::CreateTask(
        builder,
        tasktype,
        taskid, 
        reinterpret_cast<uint64_t>(curr->device),
        curr->run_time,
        curr->xfer_size,
        ntv
      ));
    }
    if (curr->device)
      devices.insert(curr->device);
  }
  auto tasks = builder.CreateVector(task_v);

  // devices
  // builder.Clear();
  std::vector<flatbuffers::Offset<FlatBufTaskGraph::Device>> dev_v = 
    std::vector<flatbuffers::Offset<FlatBufTaskGraph::Device>>();
  for (Device *curr: devices) {
    FlatBufTaskGraph::DeviceType type;
    uint64_t deviceid = reinterpret_cast<uint64_t>(curr);
    CommDevice * comm_dev;
    switch (curr->type) {
    case Device::DEVICE_COMP: 
      dev_v.emplace_back(FlatBufTaskGraph::CreateDevice(
        builder, 
        reinterpret_cast<CompDevice*>(curr)->comp_type == CompDevice::LOC_PROC 
          ? FlatBufTaskGraph::DeviceType_DEVICE_COMP_CPU
          : FlatBufTaskGraph::DeviceType_DEVICE_COMP_GPU,
        deviceid, curr->node_id, curr->device_id, 0
      ));
    break;
    case Device::DEVICE_COMM: 
      comm_dev = reinterpret_cast<CommDevice*>(curr);
      switch (comm_dev->comm_type) {
      case CommDevice::MEMBUS_COMM:
        type = FlatBufTaskGraph::DeviceType_DEVICE_COMM_MEMBUS_COMM;
      break;
      case CommDevice::UPI_IN_COMM:
        type = FlatBufTaskGraph::DeviceType_DEVICE_COMM_UPI_IN_COMM;
      break;
      case CommDevice::UPI_OUT_COMM:
        type = FlatBufTaskGraph::DeviceType_DEVICE_COMM_UPI_OUT_COMM;
      break;
      case CommDevice::NIC_IN_COMM:
        type = FlatBufTaskGraph::DeviceType_DEVICE_COMM_NIC_IN_COMM;
      break;
      case CommDevice::NIC_OUT_COMM:
        type = FlatBufTaskGraph::DeviceType_DEVICE_COMM_NIC_OUT_COMM;
      break;
      case CommDevice::PCI_TO_HOST_COMM:
        type = FlatBufTaskGraph::DeviceType_DEVICE_COMM_PCI_TO_HOST_COMM;
      break;
      case CommDevice::PCI_TO_DEV_COMM:
        type = FlatBufTaskGraph::DeviceType_DEVICE_COMM_PCI_TO_DEV_COMM;
      break;
      case CommDevice::NVLINK_COMM:
        type = FlatBufTaskGraph::DeviceType_DEVICE_COMM_NVLINK_COMM;
      break;
      case CommDevice::NW_COMM:
        type = FlatBufTaskGraph::DeviceType_DEVICE_COMM_NW_COMM;
      break;
      case CommDevice::NW_NOMINAL:
        type = FlatBufTaskGraph::DeviceType_DEVICE_COMM_NW_NOMINAL;
      break;
      }
      dev_v.emplace_back(FlatBufTaskGraph::CreateDevice(
        builder, 
        type,
        deviceid, curr->node_id, curr->device_id, comm_dev->bandwidth
      ));
    break;
    case Device::DEVICE_MEM: 
      assert("Shouldn't store a memory device to taskgraph!" && false);
    }
  }
  auto devs = builder.CreateVector(dev_v);

  // routes
  // builder.Clear();
  std::vector<flatbuffers::Offset<FlatBufTaskGraph::Route>> route_v = 
    std::vector<flatbuffers::Offset<FlatBufTaskGraph::Route>>();
  for (auto ncd: nm->get_nomm_comm_devs()) {
    std::vector<flatbuffers::Offset<FlatBufTaskGraph::Path>> path_v = 
      std::vector<flatbuffers::Offset<FlatBufTaskGraph::Path>>();
    const EcmpRoutes& physical_routes = ncd.second->get_all_routes();
    for (size_t i = 0; i < physical_routes.first.size(); i++) {
      std::vector<uint32_t> hops_v = std::vector<uint32_t>();
      for (CommDevice * c: physical_routes.second[i]) {
        hops_v.push_back(c->device_id / nm->get_total_devs());
      }
      if (physical_routes.second[i].size() > 0) {
        hops_v.push_back(physical_routes.second[i].back()->device_id%nm->get_total_devs());
      }
      auto hops = builder.CreateVector(hops_v);
      auto path = FlatBufTaskGraph::CreatePath(builder, hops, physical_routes.first[i]);
      path_v.push_back(path);
    }
    auto paths = builder.CreateVector(path_v);
    route_v.push_back(FlatBufTaskGraph::CreateRoute(
      builder, 
      ncd.second->device_id / nm->get_total_devs(),
      ncd.second->device_id % nm->get_total_devs(),
      paths
    ));
  }
  auto routes = builder.CreateVector(route_v);

  FlatBufTaskGraph::TaskGraphBuilder tg_builder = FlatBufTaskGraph::TaskGraphBuilder(builder);

  tg_builder.add_ngpupernode(machine->get_num_gpus()/ machine->get_num_nodes());
  tg_builder.add_nnode(machine->get_num_nodes());
  tg_builder.add_nswitch(nm->get_num_switches());
  tg_builder.add_intergpubw(machine->get_intra_node_gpu_bandwidth());
  tg_builder.add_drambw(32 * 1024 * 1024.0f); // PCIE gen 4
  tg_builder.add_netbw(machine->get_inter_node_gpu_bandwidth());
  tg_builder.add_conn(conns);
  tg_builder.add_ops(ops);
  tg_builder.add_tasks(tasks);
  tg_builder.add_devices(devs);
  tg_builder.add_routes(routes);

  auto ftg = tg_builder.Finish();
  builder.Finish(ftg);
}

double SpMulMatSimulator::simulate_runtime(
                                  const FFModel* model,
                                  const std::map<Op*, ParallelConfig>& global,
                                  CompMode comp_mode,
                                  std::string const &export_file_name) 
{
#ifdef WRITE_NETWORK_TRANSFER
  network_transfer_log.open("network.log");
#endif
  // printf("%s\n", machine->to_string().c_str());
  task_manager->reset();
  if (l1optimizer)
    l1optimizer->reset();
  
  std::unordered_map<SimTask*, Op*> task_to_op;

  // Step 1: register forward and backward tasks
  for (size_t l = 0; l < model->layers.size(); l++) {
    Op* op = model->layers[l];
    ParallelConfig config = global.find(op)->second;
    CostMetrics cost_metrics = measure_operator_cost(op, config);
    double forward_time = cost_metrics.forward_time;
    double backward_time = cost_metrics.backward_time;
    SimTask *ar_task = nullptr;
    for (int j = 0; j < config.num_parts(); j++) {
      SimTask* task1 = task_manager->new_forward_task(op, j);
      task_to_op[task1] = op;
      task1->device = machine->get_gpu(config.device_ids[j]);
      task1->mem = machine->get_gpu_fb_mem(config.device_ids[j]);
      task1->run_time = forward_time;
      if (l1optimizer) 
        l1optimizer->task_added(task1);
      if (comp_mode == COMP_MODE_TRAINING) {
        SimTask* task2 = task_manager->new_backward_task(op, j);
        task_to_op[task2] = op;
        task2->device = machine->get_gpu(config.device_ids[j]);
        task2->mem = machine->get_gpu_fb_mem(config.device_ids[j]);
        task2->run_time = backward_time;
        task1->add_next_task(task2);
        if (l1optimizer) 
          l1optimizer->task_added(task2);
        
      }
    }
  }

  std::unordered_set<SimTask*> ars;

  for (size_t l = 0; l < model->layers.size(); l++) {
    Op* op = model->layers[l];
    ParallelConfig config = global.find(op)->second;
    size_t element_size = data_type_size(DT_FLOAT);
    // NER step: add allreduce task after backward propogation
    for (int j = 0; j < op->numWeights; j++) {
      std::set<int> synched;
      std::vector<int> node_ids;
      for (int firstId = 0; firstId < config.num_parts(); firstId++) {
        if (synched.find(firstId) == synched.end()) {
          synched.insert(firstId);
          Domain firstR = op->get_weight_tensor_shape(config, j, firstId);
          size_t xfer_size = firstR.get_volume() * element_size;
          node_ids.push_back(config.device_ids[firstId]);
          for (int nextId = firstId+1; nextId < config.num_parts(); nextId++) {
            Domain nextR = op->get_weight_tensor_shape(config, j, nextId);
            if (firstR.intersection(nextR).get_volume() > 0) {
              // Assert all or nothing:
              // The two weights must be fully overlapped or not at all
              assert(firstR == nextR);
              assert(synched.find(nextId) == synched.end());
              synched.insert(nextId);
              node_ids.push_back(config.device_ids[nextId]);
            }
          }
          SimTask* ar_task = task_manager->new_allreduce_task(op, node_ids, xfer_size);
          task_to_op[ar_task] = op;
          ars.insert(ar_task);
          
          if (l1optimizer) 
            l1optimizer->task_added(ar_task);
          for (int dstId = 0; dstId < config.num_parts(); dstId ++) {
            task_manager->get_backward_task(op, dstId)->add_next_task(ar_task);
          }
        }
      }
    }
  }
          /*
  {
    for (size_t l = 0; l < model->layers.size(); l++) {
      Op* op0 = model->layers[l];
      ParallelConfig config0 = global.find(op0)->second;
      for (int dstId = 0; dstId < config0.num_parts(); dstId ++) {
        for (SimTask* t: ars) {
          task_manager->get_backward_task(op0, dstId)->add_next_task(t);
        }
      }
    }
  }
          */

  // Step 2: insert dependencies and comm. tasks before compute tasks
  for (size_t l = 0; l < model->layers.size(); l++) {
    Op* op = model->layers[l];
    ParallelConfig config = global.find(op)->second;
    for (int j = 0; j < op->numInputs; j++) {
      Tensor t = op->inputs[j];
      Op* pre_op = t.owner_op;
      if (pre_op == NULL)
        continue;
      ParallelConfig pre_config = global.find(pre_op)->second;
      size_t element_size = data_type_size(t.data_type);
      for (int dstId = 0; dstId < config.num_parts(); dstId ++) {
        Domain dstR = op->get_input_tensor_shape(config, j, dstId);
        for (int srcId = 0; srcId < pre_config.num_parts(); srcId ++) {
          Domain srcR = pre_op->get_output_tensor_shape(pre_config, t.owner_idx, srcId);
          if (dstR.intersection(srcR).get_volume() > 0) {
            // Forward dependency
            {
              SimTask* dstT = task_manager->get_forward_task(op, dstId);
              SimTask* srcT = task_manager->get_forward_task(pre_op, srcId);
              add_task_dependencies_with_xfer(srcT, dstT, dstR.intersection(srcR).get_volume() * element_size);
            }
            // Backward dependency
            if (comp_mode == COMP_MODE_TRAINING) {
              SimTask* dstT = task_manager->get_backward_task(op, dstId);
              SimTask* srcT = task_manager->get_backward_task(pre_op, srcId);
              add_task_dependencies_with_xfer(dstT, srcT, dstR.intersection(srcR).get_volume() * element_size);
            }
          }
        }
      }
    }
  }

  SpMulMat * smmOpt = reinterpret_cast<SpMulMat*>(l1optimizer);
  if (!smmOpt->constructed) {
    reinterpret_cast<SpMulMat*>(l1optimizer)->construct_topology();
    smmOpt->constructed = true;
  }
  
  // Step 4: add ready tasks into ready_queue
  std::priority_queue<SimTask*, std::vector<SimTask*>, SimTaskCompare> ready_queue;
  for (size_t i = 0; i < task_manager->global_task_id; i++)
    if (task_manager->tasks[i]->counter == 0)
      ready_queue.push(task_manager->tasks[i]);

  // Step 5: perform simulation

  double sim_time = 0.0f;
  std::map<Device*, double> device_times;
  // map<Device*, SimTask*> device_schedule;
  size_t idx = 0;
  while (!ready_queue.empty()) {
    // Find the task with the earliest start time
    SimTask* cur_task = ready_queue.top();
    ready_queue.pop();
    double ready_time = 0;
    double end_time;
    if (device_times.find(cur_task->device) != device_times.end()) {
      ready_time = device_times[cur_task->device];
    }
    double start_time = std::max(ready_time, cur_task->ready_time);
    if (cur_task->type == SimTask::TASK_NOMINAL_COMM) {
      if (!segment_transfer)
        end_time = route_transfer(cur_task, start_time, device_times);
      else {
        bool finished;
        end_time = route_transfer_seg(cur_task, start_time, device_times, finished);
        if (!finished) {
          ready_queue.push(cur_task);
          continue;
        }
      }
    }
    else if (cur_task->type == SimTask::TASK_ALLREDUCE) {
      if (model->config.big_gpu > 0 && task_to_op[cur_task]->op_type != OperatorType::OP_EMBEDDING) {
        double internal_ar_time = compute_internal_ar_time(model, cur_task);
        
        cur_task->ready_time += internal_ar_time;
        cur_task->run_time = internal_ar_time;
        start_time += internal_ar_time;
      }
      expand_allreduce(cur_task, start_time, ready_queue);
      idx++;
      continue;
    }
    else {
      end_time = start_time + cur_task->run_time;
      device_times[cur_task->device] = end_time;
    }

#ifdef DEBUG_PRINT
    printf("task[%lu/%lu] type(%d) run_time(%.4lf) ready_time(%.4lf) start_time(%.4lf) device(%s)\n",
          idx, task_manager->global_task_id, cur_task->type, cur_task->run_time, ready_time, start_time, (cur_task->device->name).c_str());
#endif

    if (end_time > sim_time) {
      sim_time = end_time;
    }

    for (size_t i = 0; i < cur_task->next_tasks.size(); i++) {
      SimTask* next = cur_task->next_tasks[i];
      // next->ready_time = max(next->ready_time, end_time);
      if (end_time > next->ready_time) {
        next->ready_time = end_time;
        // next->prev = t;
      }
      next->counter--;
      if (next->counter == 0) {
        ready_queue.push(next);
      }
    }
    idx++;
  }
  assert(idx == task_manager->global_task_id);
  
  // Step 6: add penalty to strategies that exceed the memory limits on devices
  // std::vector<size_t> gpu_mem_usage(machine->get_num_gpus(), 0);
  // double memory_penalty = 0.0f;
  // for (size_t l = 0; l < model->layers.size(); l++) {
  //   Op* op = model->layers[l];
  //   ParallelConfig config = global.find(op)->second;
  //   CostMetrics cost_metrics = measure_operator_cost(op, config);
  //   size_t memory_requirement = cost_metrics.memory_requirement;
  //   for (int j = 0; j < config.num_parts(); j++) {
  //     gpu_mem_usage[config.device_ids[j]] += memory_requirement;
  //   }
  // }
  // if (export_file_name != "") {  
  //   for (int i = 0; i < machine->get_num_gpus(); i++) {
  //       printf("Before penalty, dev id %d, usage %zu \n", i, gpu_mem_usage[i]); 
  //   }
  // }
  // // Penalize the total runtiem by 1ms if we exceed the memory budget by 1MB
  // for (int i = 0; i < machine->get_num_gpus(); i++) {
  //   MemDevice* gpu_fb_mem = machine->get_gpu_fb_mem(i);
  //   if (gpu_mem_usage[i] > gpu_fb_mem->capacity and gpu_fb_mem->capacity >= 0)
  //     memory_penalty += (gpu_mem_usage[i] - gpu_fb_mem->capacity) * 1e-6;
  // }
  //if (memory_penalty > 0.0f)
  //  printf("Memory penalty = %.4lf ms\n", memory_penalty);
  if (export_file_name != "") {  
    searlize_logical_taskgraph(model, export_file_name);
  }
#ifdef WRITE_NETWORK_TRANSFER
  network_transfer_log.close();
#endif
  return sim_time;//  + memory_penalty;
}

double SpMulMatSimulator::simulate_runtime(const FFModel* model,
                                  const std::map<Op*, ParallelConfig>& global,
                                  CompMode comp_mode)
{
  return this->simulate_runtime(model, global, comp_mode, "");
}  

void SpMulMatSimulator::expand_allreduce(SimTask * allreduce_task, double start_time,std::priority_queue<SimTask*, std::vector<SimTask*>, SimTaskCompare>& ready_queue)
{
  int n_participants = allreduce_task->next_tasks.size();
  if (n_participants == 1) return;
  
  SimTask * final_task = new_update_task_unrecorded();
  SpMulMat * smmOpt = reinterpret_cast<SpMulMat*>(l1optimizer);
  size_t npath = smmOpt->get_dp_ncomms(0, n_participants).size();
  assert(npath > 0);

// #ifdef FF_USE_NCCL
  // recall that next_task stores node group in this case
  final_task->device = machine->get_gpu(reinterpret_cast<uint64_t>(allreduce_task->next_tasks[0]));
  double individual_xfer_size = std::ceil((2.0 * (n_participants-1)) * allreduce_task->xfer_size / n_participants / npath);
  int hops = machine->get_num_nodes() / n_participants;
  for (int i = 0; i < n_participants; i++) {
    uint64_t src = reinterpret_cast<uint64_t>(allreduce_task->next_tasks[i]);
    const std::vector<NominalCommDevice*>& ncomms = smmOpt->get_dp_ncomms(src, n_participants);
    for (NominalCommDevice * d: ncomms) {
#ifdef DEBUG_PRINT
      std::cerr << "expand_ar: adding " << d->name << " for size " << individual_xfer_size << std::endl;
#endif
      SimTask* task = new_comm_task_unrecorded();
      task->device = d;
      task->run_time = 0;
      task->ready_time = allreduce_task->ready_time;
      task->xfer_size = individual_xfer_size;
      task->xfer_left = task->xfer_size;
      task->add_next_task(final_task);
      ready_queue.push(task);
    }
  }
  if (final_task->counter == 0) {
    final_task->ready_time = allreduce_task->ready_time;
    ready_queue.push(final_task);
  }
// #else
//   assert("SpMulMat requires ring allreduce." && false);
// #endif

}

void SpMulMatSimulator::get_taskgraph_flatbuf(const FFModel* model, flatbuffers::FlatBufferBuilder &builder) 
{
  builder.Clear();

  // Store topology
  // flatbuffers::FlatBufferBuilder builder = flatbuffers::FlatBufferBuilder();
  NetworkedMachineModel *nm = static_cast<NetworkedMachineModel*>(machine);
  size_t total_devs = nm->get_total_devs();
  std::vector<flatbuffers::Offset<FlatBufTaskGraph::Connection>> conns_v = 
    std::vector<flatbuffers::Offset<FlatBufTaskGraph::Connection>>();
  for (size_t i = 0; i < nm->get_total_devs(); i++) {
    for (size_t j = 0; j < i; j++) {
      size_t nlink;
      if ((nlink = nm->get_conn_matrix()[i * total_devs + j]) > 0) {
        conns_v.emplace_back(FlatBufTaskGraph::CreateConnection(builder, i, j, nlink));
      }
    }
  }
  auto conns = builder.CreateVector(conns_v);

  // store operators
  // builder.Clear();
  std::vector<flatbuffers::Offset<FlatBufTaskGraph::Operator>> op_v = 
    std::vector<flatbuffers::Offset<FlatBufTaskGraph::Operator>>();
  for (size_t l = 0; l < model->layers.size(); l++) {
    Op* op = model->layers[l];
    auto opname = builder.CreateString(op->name);
    op_v.emplace_back(FlatBufTaskGraph::CreateOperator(builder, 
      reinterpret_cast<uint64_t>(op), (int)op->op_type, opname));
  }
  auto ops = builder.CreateVector(op_v);

  // store tasks
  // builder.Clear();
  std::vector<flatbuffers::Offset<FlatBufTaskGraph::Task>> task_v = 
    std::vector<flatbuffers::Offset<FlatBufTaskGraph::Task>>();
  // change: since there is no universal storage of device, creat a set of
  // all devices for the next entry
  std::unordered_set<Device *> devices;
  for (size_t i = 0; i < task_manager->global_task_id; i++) {
    SimTask * curr = task_manager->tasks[i];
    if (curr->store) {
      FlatBufTaskGraph::SimTaskType tasktype;
      uint64_t taskid = reinterpret_cast<uint64_t>(curr);
      std::vector<uint64_t> nexttasks = std::vector<uint64_t>();
      for (SimTask *t: curr->next_tasks) {
        nexttasks.push_back(reinterpret_cast<uint64_t>(t));
      }
      auto ntv = builder.CreateVector(nexttasks);
      switch (curr->type) {
      case SimTask::TASK_FORWARD:
        tasktype = FlatBufTaskGraph::SimTaskType_TASK_FORWARD;
      break;
      case SimTask::TASK_BACKWARD:
        tasktype = FlatBufTaskGraph::SimTaskType_TASK_BACKWARD;
      break;
      case SimTask::TASK_UPDATE:
        tasktype = FlatBufTaskGraph::SimTaskType_TASK_UPDATE;
      break;
      case SimTask::TASK_BARRIER:
        tasktype = FlatBufTaskGraph::SimTaskType_TASK_BARRIER;
      break;
      case SimTask::TASK_COMM:
        assert("Logical task graph shouldn't contain TASK_COMM!" && false);
      break;
      case SimTask::TASK_NOMINAL_COMM:
        tasktype = FlatBufTaskGraph::SimTaskType_TASK_NOMINAL_COMM;
      break;
      case SimTask::TASK_ALLREDUCE:
        tasktype = FlatBufTaskGraph::SimTaskType_TASK_ALLREDUCE;
      break;
      }
      task_v.emplace_back(FlatBufTaskGraph::CreateTask(
        builder,
        tasktype,
        taskid, 
        reinterpret_cast<uint64_t>(curr->device),
        curr->run_time,
        curr->xfer_size,
        ntv
      ));
    }
    if (curr->device)
      devices.insert(curr->device);
  }
  auto tasks = builder.CreateVector(task_v);

  // devices
  // builder.Clear();
  std::vector<flatbuffers::Offset<FlatBufTaskGraph::Device>> dev_v = 
    std::vector<flatbuffers::Offset<FlatBufTaskGraph::Device>>();
  for (Device *curr: devices) {
    FlatBufTaskGraph::DeviceType type;
    uint64_t deviceid = reinterpret_cast<uint64_t>(curr);
    CommDevice * comm_dev;
    switch (curr->type) {
    case Device::DEVICE_COMP: 
      dev_v.emplace_back(FlatBufTaskGraph::CreateDevice(
        builder, 
        reinterpret_cast<CompDevice*>(curr)->comp_type == CompDevice::LOC_PROC 
          ? FlatBufTaskGraph::DeviceType_DEVICE_COMP_CPU
          : FlatBufTaskGraph::DeviceType_DEVICE_COMP_GPU,
        deviceid, curr->node_id, curr->device_id, 0
      ));
    break;
    case Device::DEVICE_COMM: 
      comm_dev = reinterpret_cast<CommDevice*>(curr);
      switch (comm_dev->comm_type) {
      case CommDevice::MEMBUS_COMM:
        type = FlatBufTaskGraph::DeviceType_DEVICE_COMM_MEMBUS_COMM;
      break;
      case CommDevice::UPI_IN_COMM:
        type = FlatBufTaskGraph::DeviceType_DEVICE_COMM_UPI_IN_COMM;
      break;
      case CommDevice::UPI_OUT_COMM:
        type = FlatBufTaskGraph::DeviceType_DEVICE_COMM_UPI_OUT_COMM;
      break;
      case CommDevice::NIC_IN_COMM:
        type = FlatBufTaskGraph::DeviceType_DEVICE_COMM_NIC_IN_COMM;
      break;
      case CommDevice::NIC_OUT_COMM:
        type = FlatBufTaskGraph::DeviceType_DEVICE_COMM_NIC_OUT_COMM;
      break;
      case CommDevice::PCI_TO_HOST_COMM:
        type = FlatBufTaskGraph::DeviceType_DEVICE_COMM_PCI_TO_HOST_COMM;
      break;
      case CommDevice::PCI_TO_DEV_COMM:
        type = FlatBufTaskGraph::DeviceType_DEVICE_COMM_PCI_TO_DEV_COMM;
      break;
      case CommDevice::NVLINK_COMM:
        type = FlatBufTaskGraph::DeviceType_DEVICE_COMM_NVLINK_COMM;
      break;
      case CommDevice::NW_COMM:
        type = FlatBufTaskGraph::DeviceType_DEVICE_COMM_NW_COMM;
      break;
      case CommDevice::NW_NOMINAL:
        type = FlatBufTaskGraph::DeviceType_DEVICE_COMM_NW_NOMINAL;
      break;
      }
      dev_v.emplace_back(FlatBufTaskGraph::CreateDevice(
        builder, 
        type,
        deviceid, curr->node_id, curr->device_id, comm_dev->bandwidth
      ));
    break;
    case Device::DEVICE_MEM: 
      assert("Shouldn't store a memory device to taskgraph!" && false);
    }
  }
  auto devs = builder.CreateVector(dev_v);

  // routes
  // builder.Clear();
  std::vector<flatbuffers::Offset<FlatBufTaskGraph::Route>> route_v = 
    std::vector<flatbuffers::Offset<FlatBufTaskGraph::Route>>();
  for (auto ncd: nm->get_nomm_comm_devs()) {
    std::vector<flatbuffers::Offset<FlatBufTaskGraph::Path>> path_v = 
      std::vector<flatbuffers::Offset<FlatBufTaskGraph::Path>>();
    const EcmpRoutes& physical_routes = ncd.second->get_all_routes();
    for (size_t i = 0; i < physical_routes.first.size(); i++) {
      std::vector<uint32_t> hops_v = std::vector<uint32_t>();
      for (CommDevice * c: physical_routes.second[i]) {
        hops_v.push_back(c->device_id / nm->get_total_devs());
      }
      if (physical_routes.second[i].size() > 0) {
        hops_v.push_back(physical_routes.second[i].back()->device_id%nm->get_total_devs());
      }
      auto hops = builder.CreateVector(hops_v);
      auto path = FlatBufTaskGraph::CreatePath(builder, hops, physical_routes.first[i]);
      path_v.push_back(path);
    }
    auto paths = builder.CreateVector(path_v);
    route_v.push_back(FlatBufTaskGraph::CreateRoute(
      builder, 
      ncd.second->device_id / nm->get_total_devs(),
      ncd.second->device_id % nm->get_total_devs(),
      paths
    ));
  }
  auto routes = builder.CreateVector(route_v);

  SpMulMat * smmOpt = reinterpret_cast<SpMulMat*>(l1optimizer); 
  std::vector<flatbuffers::Offset<FlatBufTaskGraph::Rings>> ring_v = 
    std::vector<flatbuffers::Offset<FlatBufTaskGraph::Rings>>{};
  for (auto entry: smmOpt->selected_jumps) {
    std::vector<flatbuffers::Offset<FlatBufTaskGraph::RingDescriptor>> rd_v = 
      std::vector<flatbuffers::Offset<FlatBufTaskGraph::RingDescriptor>>{}; 
    // std::cerr << entry.first << ": " << entry.second.size() << std::endl;
    for (size_t i = 0; i < entry.second.size(); i++) {
      auto hops = builder.CreateVector(entry.second[i]);
      // for (size_t k = 0; k < entry.second[i].size(); k++) {
      //   std::cerr << entry.second[i][k] << ", ";
      // }
      // std::cerr << std::endl;
      auto rd = FlatBufTaskGraph::CreateRingDescriptor(builder, hops);
      rd_v.push_back(rd);
    }
    auto rds = builder.CreateVector(rd_v);
    auto ring = FlatBufTaskGraph::CreateRings(builder, entry.first, rds);
    ring_v.push_back(ring);
  }
  auto rings = builder.CreateVector(ring_v);
  
  FlatBufTaskGraph::TaskGraphBuilder tg_builder = FlatBufTaskGraph::TaskGraphBuilder(builder);

  tg_builder.add_ngpupernode(machine->get_num_gpus()/ machine->get_num_nodes());
  tg_builder.add_nnode(machine->get_num_nodes());
  tg_builder.add_nswitch(nm->get_num_switches());
  tg_builder.add_intergpubw(machine->get_intra_node_gpu_bandwidth());
  tg_builder.add_drambw(32 * 1024 * 1024.0f); // PCIE gen 4
  tg_builder.add_netbw(machine->get_inter_node_gpu_bandwidth());
  tg_builder.add_conn(conns);
  tg_builder.add_ops(ops);
  tg_builder.add_tasks(tasks);
  tg_builder.add_devices(devs);
  tg_builder.add_routes(routes);
  tg_builder.add_rings(rings);

  auto ftg = tg_builder.Finish();
  builder.Finish(ftg);
}
