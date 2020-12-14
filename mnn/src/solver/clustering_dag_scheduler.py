
import queue
from operator import itemgetter, attrgetter
import pysnooper

from utils import utils
from parser import dagp_clustering_parser as dagpp
from profile import read_profile_data
from profile import subgraph
from profile import net_struct
from solver import scheduler_utils

def get_op_bottom_level(op):
    return (op.op_def.operator_latency.CPU_latency + op.op_def.operator_latency.GPU_latency) / 2.0


def compute_bottom_level(op_name_list, name_op_dict):
    """Compute the bottom level of every operator in the DAG
    The bottom level bl(i) of a task vi ∈ V is defined 
    as the largest weight of a path from vi to a target node (vertex without successors), 
    including the weight wi of vi, and all communication costs.
    """
    output_node_list = [op_name for op_name in op_name_list if len(name_op_dict[op_name].children) == 0]
    print(output_node_list)
    ready = queue.Queue()
    visited = set()
    # Init output operators' bottom_level
    for output_node_name in output_node_list:
        output_op = name_op_dict[output_node_name]
        output_op.bottom_level = get_op_bottom_level(output_op)
        ready.put(output_node_name)
        visited.add(output_node_name)
    while not ready.empty():
        op_name = ready.get()
        op = name_op_dict[op_name]
        visited.add(op_name)
        for op_parent_name in op.parents:
            op_parent = name_op_dict[op_parent_name]
            op_parent.bottom_level = max(op_parent.bottom_level, \
                op.bottom_level + get_op_bottom_level(op_parent))
            if op_parent_name not in visited:
                ready.put(op_parent_name)
    for op_name in op_name_list:
        print(op_name, name_op_dict[op_name].bottom_level)


def bl_macro_scheduler(op_name_list, name_op_dict, model, mobile, thread):
    """ Scheduler based on bottom level priority and 
    """
    ops_not_support_by_GPU = []
    # Record the CPU queue and GPU queue finish timestamp
    CPU_end_point = 0.0
    GPU_end_point = 0.0
    # Need to record the execute order
    op_execution_order_list = []
    input_op_names = subgraph.find_input_nodes(op_name_list, name_op_dict)
    ops_queue = [name_op_dict[op_name] for op_name in input_op_names]
    print("input op name: ", input_op_names)
    assert(len(ops_queue) > 0)
    print("Start bottom level macro scheduler")
    compute_bottom_level(op_name_list, name_op_dict)
    ops_list = [name_op_dict[op_name] for op_name in op_name_list]
    ops_list = sorted(ops_list, key=attrgetter("bottom_level"))
    idx = 0
    for op_name in op_name_list:
        print(idx, op_name, name_op_dict[op_name].bottom_level)
        idx += 1
    
    ready_parts = []
    ready_parts.extend([name_op_dict[op_name] for op_name in input_op_names])
    while len(ready_parts) > 0:
        # Get the ready parts with the highest priority
        sorted(ready_parts, key=attrgetter("bottom_level"), reverse=True)
        part = ready_parts[0]
        ready_parts.remove(part)
        ops_in_part = [name_op_dict[op_name] for op_name in part.op_name_list]
        ready_ops = [op for op in ops_in_part if scheduler_utils.is_parents_executed(op, part.op_name_list, name_op_dict)]
        while len(ready_ops) > 0:
            ready_ops = sorted(ready_ops, key=attrgetter("bottom_level"), reverse=True)
            op = ready_ops[0]
            ready_ops.remove(op)

            if op.op_def.type in ops_not_support_by_GPU:
                CPU_end_point = scheduler_utils.assign_op_to_device(op, name_op_dict, net_struct.DeviceType.CPU, \
                    CPU_end_point, op.op_def.operator_latency.CPU_latency, op_execution_order_list)
                continue

            CPU_latency, GPU_latency = scheduler_utils.get_ops_total_latency(op, name_op_dict)
            
            # Greedy here: Earliest task finish time first
            if max(CPU_end_point, op.earlist_start_point) + CPU_latency \
                <= max(GPU_end_point, op.earlist_start_point) + GPU_latency:
                CPU_end_point = max(CPU_end_point, op.earlist_start_point)
                CPU_end_point = scheduler_utils.assign_op_to_device(op, name_op_dict, net_struct.DeviceType.CPU, CPU_end_point, CPU_latency, op_execution_order_list)
            else:
                GPU_end_point = max(GPU_end_point, op.earlist_start_point)
                GPU_end_point = scheduler_utils.assign_op_to_device(op, name_op_dict, net_struct.DeviceType.GPU, GPU_end_point, GPU_latency, op_execution_order_list)
            
            op.executed = True
            for child_name in op.children:
                if child_name not in part.op_name_list:
                    continue
                child_op = name_op_dict[child_name]
                if scheduler_utils.is_parents_executed(child_op, part.op_name_list, name_op_dict) and not child_op.executed:
                    ready_ops.append(child_op)
            # End of scheduling a subgraph/part
        part.executed = True
        for child_part_name in part.children:
            child_part = name_op_dict[child_part_name]
            if not isinstance(child_part, subgraph.Subgraph):
                continue
            if scheduler_utils.is_parents_executed(child_part, op_name_list, name_op_dict, is_subgraph=True) and not child_part.executed:
                ready_parts.append(child_part)
        # End of scheduling DAG
    print("CPU end point: %s ms." % CPU_end_point)
    print("GPU end point: %s ms." % GPU_end_point)
    print("Greedy Result %f" % (max(CPU_end_point, GPU_end_point)))
    subgraph.write_subgraph_device_placement_result(name_op_dict=name_op_dict, op_execution_order_list=op_execution_order_list)



def test_bottom_level():
    model, mobile, thread, _ = utils.parse_model_mobile()
    op_name_list, name_op_dict = read_profile_data.load_model_profile(model, mobile, thread)
    compute_bottom_level(op_name_list, name_op_dict)


def test_dagp_clusters():
    model, mobile, thread, _ = utils.parse_model_mobile()
    clusters = dagpp.dagp_clustering_parser(model, 8)
    op_name_list, name_op_dict = read_profile_data.load_model_profile(model, mobile, thread)
    
    subgraph_name_list, name_op_dict = subgraph.build_graphs_with_cluster_lists(clusters, op_name_list, name_op_dict)
    for subgraph_name in subgraph_name_list:
        sg = name_op_dict[subgraph_name]
        print(sg.name, [name for name in sg.parents], [name for name in sg.children])
    bl_macro_scheduler(subgraph_name_list, name_op_dict, model, mobile, thread)
    



if __name__=="__main__":
    # test_bottom_level()
    test_dagp_clusters()