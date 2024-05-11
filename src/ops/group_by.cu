/* Copyright 2019 Stanford
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

#include "model.h"
#include "cuda_helper.h"
#include <math.h>
#include <stdio.h>

#define MAX_K 4
#define MAX_BATCH_SIZE 32
#define MAX_N 12


void FFModel::group_by(const Tensor& input,
                        const Tensor& assign,
                        Tensor* outputs,
                        int n, float alpha,
                        const char* name)
{
  Group_by* group_by = new Group_by(*this, input, assign, n, alpha, name);
  layers.push_back(group_by);
  for (int i = 0; i < n; i++)
    outputs[i] = group_by->outputs[i];
}


Group_by::Group_by(FFModel& model,
                  const Tensor& _input,
                  const Tensor& _assign,
                  int _n, float _alpha,
                  const char* name)
: Op(model, OP_GROUP_BY, name, _input, _assign),
  n(_n),
  alpha(_alpha),
  profiling(model.config.profiling)
{
  // FIXME: For now, set upper limits Better: Do as follows, but memory is
  // assigned per block, so requires to check that
  // https://stackoverflow.com/questions/5531247/allocating-shared-memory/5531640#5531640
  assert(n <= MAX_N && "Increase MAX_N in #define");
  assert(inputs[1].adim[0] <= MAX_K && "Increase MAX_K in #define");
  assert(inputs[0].adim[1] <= MAX_BATCH_SIZE && "Increase MAX_BATCH_SIZE in #define");

  assert(_input.numDim == 2); // TODO: support dims > 2
  assert(_input.numDim == 2);
  assert(_input.adim[1] == _assign.adim[1]);
  assert(n > 0);

  // List of outputs
  int k = _assign.adim[0];
  printf("group_by: k:%d, inputs[0].adim_1:%d\n",k,inputs[0].adim[1]);
  for(int i = 0; i < n; i++) {
    outputs[i].numDim = 2;
    outputs[i].adim[0] = inputs[0].adim[0];
    outputs[i].adim[1] = (int)ceil(alpha*k/n*inputs[0].adim[1]);
  }

  numWeights = 0;
}


void Group_by::create_weights(FFModel& model)
{
  // Do nothing
}


void Group_by::create_output_and_partition(FFModel& model)
{
  // Retrieve the task indexspace for the op
  std::string pcname = name;
  task_is = IndexSpaceT<2>(model.get_or_create_task_is(2, pcname));
  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  Rect<2> part_rect = runtime->get_index_space_domain(ctx, task_is);

  // Can only partition over the sample dim
  assert(part_rect.hi[0] == part_rect.lo[0]);

  int k = inputs[1].adim[0];
  const int dims[2] = {(int)ceil(alpha*k/n*inputs[0].adim[1]), inputs[0].adim[0]};
  for(int i = 0; i < n; i++) {
    outputs[i] = model.create_tensor<2>(dims, DT_FLOAT, this);
    outputs[i].owner_op = this;
    outputs[i].owner_idx = i;
  }

  // Compute partition bound for input
  Rect<2> input_rect = runtime->get_index_partition_color_space(
      ctx, inputs[0].part.get_index_partition());
  if (input_rect == part_rect) {
    input_lps[0] = inputs[0].part;
    input_grad_lps[0] = inputs[0].part_grad;
  } else {
    model.create_disjoint_partition<2>(
      inputs[0], (IndexSpaceT<2>)task_is, input_lps[0], input_grad_lps[0]);
  }
  input_rect = runtime->get_index_partition_color_space(
      ctx, inputs[1].part.get_index_partition());
  if (input_rect == part_rect) {
    input_lps[1] = inputs[1].part;
    input_grad_lps[1] = inputs[1].part_grad;
  } else {
    model.create_disjoint_partition<2>(
      inputs[1], (IndexSpaceT<2>)task_is, input_lps[1], input_grad_lps[1]);
  }
}


OpMeta* Group_by::init_task(const Task* task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime* runtime)
{
  Group_by* gb = (Group_by*) task->args;
  FFHandler handle = *((FFHandler*)task->local_args);
  GroupByMeta* m = new GroupByMeta(handle, gb->n);
  m->profiling = gb->profiling;
  return m;
}


void Group_by::init(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  Domain domain = runtime->get_index_space_domain(ctx, task_is);
  switch (domain.get_dim()) {
#define DIMFUNC(DIM) \
    case DIM: \
    { \
      Rect<DIM> rect = domain; \
      ParallelConfig pc; \
      std::string pcname = name; \
      ff.config.find_parallel_config(DIM, pcname, pc); \
      int idx = 0; \
      for (PointInRectIterator<DIM> it(rect); it(); it++) { \
        FFHandler handle = ff.handlers[pc.device_ids[idx++]]; \
        argmap.set_point(*it, TaskArgument(&handle, sizeof(FFHandler))); \
      } \
      break; \
    }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
  IndexLauncher launcher(GROUP_BY_INIT_TASK_ID, task_is,
                         TaskArgument(this, sizeof(Group_by)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  switch (domain.get_dim()) {
#define DIMFUNC(DIM) \
    case DIM: \
    { \
      Rect<DIM> rect = domain; \
      int idx = 0; \
      for (PointInRectIterator<DIM> it(rect); it(); it++) { \
        meta[idx++] = fm.get_result<OpMeta*>(*it); \
      } \
      break; \
    }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
}


__global__
void gb_forward_kernel(const float* input,
        const int* exp_assign,
        float** outputs,
        int n, // num experts
        int k, // chosen experts
        float alpha, // factor additional memory assigned
        int batch_size,
        int data_dim)
{
  __shared__ float* chosen_exp_preds[MAX_K*MAX_BATCH_SIZE];//one pointer for each exp_assign (TopK_output[1]) element

  // Get pred pointers, single thread per block
  //printf("groupby: threadIdx.x:%d\n",threadIdx.x);
  if(threadIdx.x == 0) {
    int exp_tensor_rows = ceil(alpha*k/n*batch_size);//=This is the max expert capacity =26
    int expert_idx[MAX_N] = {0};// This is the number of tokens assigned to each expert
    for(int i = 0; i < k*batch_size; i++) {
      // Get pointer to chosen expert predictions
      int expert = exp_assign[i]; // index of the expert that is to receive the token i
      if(expert_idx[expert] >= exp_tensor_rows) { // check if the expert is already at capacity
        // dropped sample
        chosen_exp_preds[i] = 0;
        continue;
      }
      // chosen_exp_preds[i] is the pointer to the location in the outputs
      // tensor's memory where we should copy the i-th tensor. outputs[expert]
      // points us to the assigned expert's (DATA_DIM, expert capacity) tensor
      // block expert_idx[expert] * data_dim is the offset within the block
      chosen_exp_preds[i] = outputs[expert] + expert_idx[expert]*data_dim;
      expert_idx[expert]++;
    }
  }
  printf("groupby: before compute output\n");
  __syncthreads();

  // compute output
  CUDA_KERNEL_LOOP(i, k*batch_size*data_dim)
  {
    if(chosen_exp_preds[i/data_dim] != 0) {
      float a = input[(i/(k*data_dim))*data_dim + i%data_dim];
      chosen_exp_preds[i/data_dim][i%data_dim] = a;
    }
  }
  printf("groupby: gb_forward_kernel is done\n");
}


__global__
void gb_backward_kernel(float* input_grad,
        const int* exp_assign,
        float** output_grads,
        int n, // num experts
        int k, // chosen experts
        float alpha, // factor additional memory assigned
        int batch_size,
        int data_dim)
{
  __shared__ float* chosen_exp_grads[MAX_K*MAX_BATCH_SIZE];

  // Get pred pointers, single thread
  if(blockIdx.x * blockDim.x + threadIdx.x == 0) {
    int exp_tensor_rows = ceil(alpha*k/n*batch_size);
    int expert_idx[MAX_N] = {0};
    for(int i = 0; i < k*batch_size; i++) {
      // Get pointer to chosen expert predictions
      int expert = exp_assign[i];
      if(expert_idx[expert] >= exp_tensor_rows) {
        // dropped sample
        chosen_exp_grads[i] = 0;
        continue;
      }
      chosen_exp_grads[i] = output_grads[expert] + expert_idx[expert]*data_dim;
      expert_idx[expert]++;
    }
  }

  __syncthreads();

  // compute output
  CUDA_KERNEL_LOOP(i, k*batch_size*data_dim)
  {
    if(chosen_exp_grads[i/data_dim] != 0) {
      input_grad[(i/(k*data_dim))*data_dim + i%data_dim] = chosen_exp_grads[i/data_dim][i%data_dim];
    }
  }
}


void Group_by::forward_task(const Task *task,
                            const std::vector<PhysicalRegion>& regions,
                            Context ctx, Runtime* runtime)
{
  // Get n, alpha
  const Group_by* gb = (Group_by*) task->args;
  int n = gb->n;
  float alpha = gb->alpha;

  assert((int)regions.size() == n+2);
  assert((int)task->regions.size() == n+2);

  const GroupByMeta* m = *((GroupByMeta**)task->local_args);

  // get input and assign regions
  const AccessorRO<float, 2> acc_input(regions[0], FID_DATA);
  const AccessorRO<int, 2> acc_assign(regions[1], FID_DATA);

  Rect<2> rect_input = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Rect<2> rect_assign = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());

  coord_t input_rows = rect_input.hi[1] - rect_input.lo[1] + 1;
  coord_t input_cols = rect_input.hi[0] - rect_input.lo[0] + 1;
  assert(input_rows == rect_assign.hi[1] - rect_assign.lo[1] + 1);
  int k = rect_assign.hi[0] - rect_assign.lo[0] + 1;
  int batch_size = input_rows;
  int data_dim = input_cols;

  // get output
  float* outputs[n];
  //int exp_output_rows = (int)ceil(alpha*k/n*batch_size);
  for(int i = 0; i < n; i++) {
    Domain out_domain = runtime->get_index_space_domain(
      ctx, task->regions[i+2].region.get_index_space());
    outputs[i] = helperGetTensorPointerWO<float>(
      regions[i+2], task->regions[i+2], FID_DATA, ctx, runtime);

    //coord_t output_rows = out_domain.hi()[1] - out_domain.lo()[1] + 1;
    coord_t output_cols = out_domain.hi()[0] - out_domain.lo()[0] + 1;
    //assert((int)output_rows == exp_output_rows);
    assert(output_cols == input_cols);
  }

  // TODO: why cublas/cudnn stream is needed here?
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  // call forward kernel
  cudaMemcpy(m->dev_region_ptrs, outputs, n*sizeof(float*), cudaMemcpyHostToDevice);

  gb_forward_kernel<<<GET_BLOCKS(batch_size*k*data_dim), min(CUDA_NUM_THREADS,(int)(batch_size*k*data_dim)), 0, stream>>>(
    acc_input.ptr(rect_input), acc_assign.ptr(rect_assign), m->dev_region_ptrs, n, k,
    alpha, batch_size, data_dim);
}


void Group_by::backward_task(const Task *task,
                            const std::vector<PhysicalRegion>& regions,
                            Context ctx, Runtime* runtime)
{
  // Get n, alpha
  const GroupByMeta* m = *((GroupByMeta**)task->local_args);
  const Group_by* gb = (Group_by*) task->args;
  int n = gb->n;
  float alpha = gb->alpha;

  assert((int)regions.size() == n+2);
  assert((int)task->regions.size() == n+2);

  // get input and assign regions
  const AccessorWO<float, 2> acc_input_grad(regions[0], FID_DATA);
  const AccessorRO<int, 2> acc_assign(regions[1], FID_DATA);

  Rect<2> rect_input_grad = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Rect<2> rect_assign = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());

  coord_t input_rows = rect_input_grad.hi[1] - rect_input_grad.lo[1] + 1;
  coord_t input_cols = rect_input_grad.hi[0] - rect_input_grad.lo[0] + 1;
  assert(input_rows == rect_assign.hi[1] - rect_assign.lo[1] + 1);
  int k = rect_assign.hi[0] - rect_assign.lo[0] + 1;
  int batch_size = input_rows;
  int data_dim = input_cols;

  // get output
  float* output_grads[n];
  //int exp_output_rows = (int)ceil(alpha*k/n*batch_size);
  for(int i = 0; i < n; i++) {
    Domain out_domain = runtime->get_index_space_domain(
      ctx, task->regions[i+2].region.get_index_space());
    output_grads[i] = helperGetTensorPointerRW<float>(
      regions[i+2], task->regions[i+2], FID_DATA, ctx, runtime);

    //coord_t output_rows = out_domain.hi()[1] - out_domain.lo()[1] + 1;
    coord_t output_cols = out_domain.hi()[0] - out_domain.lo()[0] + 1;
    //assert((int)output_rows == exp_output_rows);
    assert(output_cols == input_cols);
  }

  // TODO: why cublas/cudnn stream is needed here
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  // call forward kernel
  cudaMemcpy(m->dev_region_ptrs, output_grads, n*sizeof(float*), cudaMemcpyHostToDevice);

  gb_backward_kernel<<<GET_BLOCKS(batch_size*k*data_dim), min(CUDA_NUM_THREADS,(int)(batch_size*k*data_dim)), 0, stream>>>(
    acc_input_grad.ptr(rect_input_grad), acc_assign.ptr(rect_assign), m->dev_region_ptrs,
    n, k, alpha, batch_size, data_dim);
}


void Group_by::forward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  Domain domain = runtime->get_index_space_domain(ctx, task_is);
  switch (domain.get_dim()) {
#define DIMFUNC(DIM) \
    case DIM: \
    { \
      Rect<DIM> rect = domain; \
      int idx = 0; \
      for (PointInRectIterator<DIM> it(rect); it(); it++) { \
        OpMeta* mp = meta[idx++]; \
        argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta*))); \
      } \
      break; \
    }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
  IndexLauncher launcher(GROUP_BY_FWD_TASK_ID, task_is,
                         TaskArgument(this, sizeof(Group_by)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));
  // data
  launcher.add_region_requirement(
    RegionRequirement(input_lps[0], 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, inputs[0].region));
  launcher.add_field(0, FID_DATA);

  // assign
  launcher.add_region_requirement(
    RegionRequirement(input_lps[1], 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, inputs[1].region));
  launcher.add_field(1, FID_DATA);

  // output
  for(int i = 0; i < n; i++) {
    launcher.add_region_requirement(
      RegionRequirement(outputs[i].part, 0/*projection id*/,
        WRITE_ONLY, EXCLUSIVE, outputs[i].region));
    launcher.add_field(i+2, FID_DATA);
  }

  runtime->execute_index_space(ctx, launcher);
}

void Group_by::backward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  Domain domain = runtime->get_index_space_domain(ctx, task_is);
  switch (domain.get_dim()) {
#define DIMFUNC(DIM) \
    case DIM: \
    { \
      Rect<DIM> rect = domain; \
      int idx = 0; \
      for (PointInRectIterator<DIM> it(rect); it(); it++) { \
        OpMeta* mp = meta[idx++]; \
        argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta*))); \
      } \
      break; \
    }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
  IndexLauncher launcher(GROUP_BY_BWD_TASK_ID, task_is,
                         TaskArgument(this, sizeof(Group_by)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));

  // input_grad
  launcher.add_region_requirement(
    RegionRequirement(input_grad_lps[0], 0/*projection id*/,
      WRITE_ONLY, EXCLUSIVE, inputs[0].region_grad));
  launcher.add_field(0, FID_DATA);

  // assign
  launcher.add_region_requirement(
    RegionRequirement(input_lps[1], 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, inputs[1].region));
  launcher.add_field(1, FID_DATA);

  // output grad
  for(int i = 0; i < n; i++) {
    launcher.add_region_requirement(
      RegionRequirement(outputs[i].part_grad, 0/*projection id*/,
        WRITE_ONLY, EXCLUSIVE, outputs[i].region_grad));
    launcher.add_field(i+2, FID_DATA);
  }

  runtime->execute_index_space(ctx, launcher);
}


GroupByMeta::GroupByMeta(FFHandler handler, int n)
: OpMeta(handler)
{
  checkCUDA(cudaMalloc(&dev_region_ptrs, n*sizeof(float*)));
}
GroupByMeta::~GroupByMeta(void)
{
  checkCUDA(cudaFree(&dev_region_ptrs));
}
void Group_by::forward_kernel_wrapper(GroupByMeta const *m,
                                      float const *input,
                                      int const *exp_assign,
                                      float **outputs,
                                      int n, // num experts
                                      int k, // chosen experts
                                      int batch_size,
                                      int data_dim) {
  // TODO: why cublas/cudnn stream is needed here?

  float alpha = m->alpha;

  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }
  // call forward kernel
  cudaMemcpyAsync(m->dev_region_ptrs,
                  outputs,
                  n * sizeof(float *),
                  cudaMemcpyHostToDevice,
                  stream);//new version
  //cudaMemcpy(m->dev_region_ptrs, outputs, n*sizeof(float*), cudaMemcpyHostToDevice);//from old
  printf("Groupby: finish cudaMemcpy\n");
  gb_forward_kernel<<<GET_BLOCKS(batch_size * k * data_dim),
                      min(CUDA_NUM_THREADS, (int)(batch_size * k * data_dim)),
                      0,
                      stream>>>(
      input, exp_assign, m->dev_region_ptrs, n, k, alpha, batch_size, data_dim);
  printf("Groupby: finish forward kernel\n");
  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("[GroupBy] forward time = %.2lfms\n", elapsed);
  }
}
void Group_by::inner_measure_operator_cost(Simulator *sim,
                                     std::function<void()> const &forward,
                                     std::function<void()> const &backward,
                                     CostMetrics& cost_metrics)
{
  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  // measure forward time
  printf("ready to measure forward time\n");
  checkCUDA(cudaDeviceSynchronize());
  for (int i = 0; i < sim->warmup_times + sim->repeat_times; i++) {
    if (i == sim->warmup_times) {
      checkCUDA(cudaEventRecord(sim->start_event, stream));
    }
    forward();
  }
  checkCUDA(cudaEventRecord(sim->end_event, stream));
  checkCUDA(cudaEventSynchronize(sim->end_event));
  float milliseconds;
  cudaEventElapsedTime(&milliseconds, sim->start_event, sim->end_event);
  cost_metrics.forward_time = milliseconds / sim->repeat_times;
  printf("finish measure forward time\n");
  // measure backward time
  if (false) {
    checkCUDA(cudaDeviceSynchronize());
    for (int i = 0; i < sim->warmup_times + sim->repeat_times; i++) {
      if (i == sim->warmup_times) {
        checkCUDA(cudaEventRecord(sim->start_event, stream));
      }
      backward();
    }
    checkCUDA(cudaEventRecord(sim->end_event, stream));
    checkCUDA(cudaEventSynchronize(sim->end_event));
    cudaEventElapsedTime(&milliseconds, sim->start_event, sim->end_event);
    cost_metrics.backward_time = milliseconds / sim->repeat_times;
  } else {
    cost_metrics.backward_time = 0.0f;
  }

  // compute memory usage
  // Assume:
  //   1. all memory allocations use Simulator::allocate
  //   2. we call Simulator::free_all before measure an operator
  // Therefore, the memory usage of an operator is sim->offset
  cost_metrics.memory_requirement = (size_t)sim->offset;
}
bool Group_by::measure_operator_cost(Simulator* sim,
                                 const ParallelConfig& pc,
                                 CostMetrics& cost_metrics)
{
  // //TODO: implement
  // cost_metrics.forward_time = 0.0f;
  // cost_metrics.backward_time = 0.0f;
  // cost_metrics.memory_requirement = 0;
  // return false;
  printf("Enter the Group_by measurement\n");
  assert(numOutputs <= MAX_NUM_OUTPUTS);
  Tensor sub_input, sub_assign;
  Tensor sub_outputs[MAX_NUM_OUTPUTS];
  if (!inputs[0].get_input_sub_tensor(pc, sub_input, op_type)) {
    return false;
  }
  if (!inputs[0].get_input_sub_tensor(pc, sub_assign, op_type)) {
    return false;
  }
  for (int i = 0; i < numOutputs; ++i) {
    if (!outputs[i].get_output_sub_tensor(pc, sub_outputs[i], op_type)) {
      return false;
    }
  }
  printf("Group_by: finish get sub tensor\n");

  GroupByMeta *m = new GroupByMeta(sim->handler, n);
  m->alpha = alpha; // also need to update alpha

  // allocate
  sim->free_all();
  float *input_ptr = (float *)sim->allocate(sub_input.get_volume(), DT_FLOAT);
  int *assign_ptr = (int *)sim->allocate(sub_assign.get_volume(), DT_INT32);
  //cost_metrics.inputs_memory += cost_metrics.total_mem_diff_from(sim->offset);

  float *output_ptrs[MAX_NUM_OUTPUTS];
  bool out_of_memory = false;
  for (int i = 0; i < numOutputs; i++) {
    output_ptrs[i] =
        (float *)sim->allocate(sub_outputs[i].get_volume(), DT_FLOAT);
    out_of_memory = out_of_memory || (output_ptrs[i] == NULL);
  }
  //cost_metrics.outputs_memory += cost_metrics.total_mem_diff_from(sim->offset);

  if (out_of_memory || !input_ptr || !assign_ptr) {
    cost_metrics.forward_time = 1e7;
    cost_metrics.backward_time = 1e7;
    return true;
  }

  assert(m->profiling == false);

  // compute
  std::function<void()> forward, backward;

  Domain in_domain = sub_input.get_domain();
  //int k = sub_assign.dims[0].size;
  Domain assign_domain = sub_assign.get_domain();
  int k = assign_domain.hi()[0] - assign_domain.lo()[0] + 1;//old version of calculating k

  int batch_size = in_domain.hi()[1] - in_domain.lo()[1] + 1;
  int data_dim = in_domain.hi()[0] - in_domain.lo()[0] + 1;
  printf("Group_by: ready to enter forward \n");
  forward = [&] {
    forward_kernel_wrapper(
        m, input_ptr, assign_ptr, output_ptrs, n, k, batch_size, data_dim);
  };

  inner_measure_operator_cost(sim, forward, backward, cost_metrics);
  printf("Group_by: finish inner measure operator\n");
  // log_measure.debug("[Measure GroupBy] name(%s) forward_time(%.4lf)\n",
  //                   name,
  //                   cost_metrics.forward_time);

  cost_metrics.backward_time = 0.0f; // not implemented for MOE
  delete m;
  
  return true;
}

std::string Group_by::get_name_structure() const {
  return "Group_by";
}