import pysnooper

from profile import subgraph
from profile import net_struct
from utils import utils

def update_children_start_point(op, ops_relation_dict, \
  device_start_point, latency):
  for child_name in op.children:
    if child_name in ops_relation_dict:
      op_child = ops_relation_dict[child_name]
      op_child.earlist_start_point = \
        max(op_child.earlist_start_point, device_start_point + latency)
    else:
      print("Can not find op %s in dict" % child_name)


# Place the op on CPU or GPU, return the updated device end point
def assign_op_to_device(op, ops_relation_dict, device_type, device_end_point, latency, op_execute_order_list):
  op.executed = True
  op.op_def.device_type = device_type
  op_execute_order_list.append((op.name, device_type, device_end_point + latency))
  update_children_start_point(op, ops_relation_dict, device_end_point, latency)
  return device_end_point + latency


def is_parents_executed(op, op_name_list, ops_relation_dict, is_subgraph=False):
  ready = True # When all his father has been executed, then the op can start executing
  for op_parent_name in op.parents:
    if not op_parent_name in ops_relation_dict.keys():
      raise KeyError()
    if op_parent_name not in op_name_list:
      continue
    op_parent = ops_relation_dict[op_parent_name]
    if is_subgraph:
      if not isinstance(op_parent, subgraph.Subgraph):
        continue
    if op_parent.executed == False:
      ready = False
      break
  return ready


def write_device_placement(filename, net_def):
  f = open(filename, 'w')
  for op in net_def.op:
    f.write("%s %d\n" % (op.name, op.op_def.device_type))
  f.flush()
  f.close()
  print("Write device placement done.")


# def get_ops_total_latency(op, ops_relation_dict):
#   to_CPU_transpose_latency = 0.0
#   to_GPU_transpose_latency = 0.0
  
#   for op_parent_name in op.parents:
#     op_parent = ops_relation_dict[op_parent_name]
#     if op_parent.op_def.device_type == net_struct.DeviceType.CPU:
#       to_GPU_transpose_latency = max(to_GPU_transpose_latency, op_parent.op_def.operator_latency.Transpose_latency_NCHW_to_NHWC)
#     elif op_parent.op_def.device_type == net_struct.DeviceType.GPU:
#       to_CPU_transpose_latency = max(to_GPU_transpose_latency, op_parent.op_def.operator_latency.Transpose_latency_NHWC_to_NCHW)
  
#   # print(op.op_def.operator_latency)
#   CPU_latency = op.op_def.operator_latency.CPU_latency + to_CPU_transpose_latency
#   GPU_latency = op.op_def.operator_latency.GPU_latency + to_GPU_transpose_latency
#   return CPU_latency, GPU_latency

# @pysnooper.snoop()
def get_ops_total_latency(op, name_op_dict):
  """Get the op's execution latency on CPU and GPU
  based on the device placement result of op's parents
  considering the communication latency
  """
  to_CPU_transpose_latency = 0.0
  to_GPU_transpose_latency = 0.0
  utils.get_logger().info("op name: {}, input tensors:{}, data trans dict {}".format(op.name, op.input_tensors, op.op_def.operator_latency.input_data_trans_latency))
  for op_parent_name in op.parents:
    op_parent = name_op_dict[op_parent_name]
    for child_tensor_addr, child_tensor_shape in op.input_tensors:
      for parent_tensor_addr, parent_tensor_shape in op_parent.output_tensors:
        if child_tensor_addr == parent_tensor_addr:
          utils.get_logger().info("{} {} {}".format(op.name, op_parent.name, parent_tensor_shape))
          if op_parent.op_def.device_type == net_struct.DeviceType.CPU:
            to_GPU_transpose_latency += op.op_def.operator_latency.input_data_trans_latency[child_tensor_addr][1]
          elif op_parent.op_def.device_type == net_struct.DeviceType.GPU:
            to_CPU_transpose_latency += op.op_def.operator_latency.input_data_trans_latency[child_tensor_addr][0]
  utils.get_logger().info("{} {} {} {} {}".format(op.name, \
    op.op_def.operator_latency.CPU_latency, op.op_def.operator_latency.GPU_latency,\
     to_CPU_transpose_latency, to_GPU_transpose_latency))
  CPU_latency = op.op_def.operator_latency.CPU_latency + to_CPU_transpose_latency
  GPU_latency = op.op_def.operator_latency.GPU_latency + to_GPU_transpose_latency
  return CPU_latency, GPU_latency

