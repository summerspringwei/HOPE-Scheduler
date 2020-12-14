import queue
import os
import pysnooper
import logging

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger()


from profile import read_profile_data
from profile import graph_partition
from profile import find_critical_node
from utils import utils
from visualization import *


# DO NOT MODIFY THE VALUE OF `CPU` AND `GPU`
CPU = 1
GPU = 2

# `M` and `K` can be changed based on the latency of the subgraph
M = 1000
K = 2000
# GPU has to do the data transformation for CPU
# there for the GPU execution time also increases
# we use a scale factor to simulate the GPU execution time increasing
GPU_TRANSFORM_SCALE_FACTOR = 1

class LPMode:
    Mode_Subgraph = 0
    Mode_Operator = 1
    Mode_AUTO_Subgraph = 2


# Read one module names and associate a name with one index
def associate_op_name_with_idx(file_path):
    f = open(file_path)
    one_module_names_idx_dict = {}
    idx = 1
    for line in f.readlines():
        one_module_names_idx_dict[line.strip()] = idx
        idx += 1
    return one_module_names_idx_dict


def associate_op_name_list_with_idx(op_name_list):
    idx = 1
    one_module_names_idx_dict = {}
    for op_name in op_name_list:
        one_module_names_idx_dict[op_name] = idx
        idx += 1
    return one_module_names_idx_dict


# Used to generate (s[y][i]*c[u][j][i] for all i belongs to j's parents)
def get_parent_idxes_and_data_trans(op_name, one_module_names_idx_dict, op_dict, device, mode=LPMode.Mode_Subgraph):
    op = op_dict[op_name]
    assert(device in [CPU, GPU])
    # For all 
    acc_data_trans_latency = 0.0
    parent_idx_data_trans = []
    for (addr, data_trans) in op.op_def.operator_latency.input_data_trans_latency.items():
        data_trans_latency = data_trans[device-1]
        for op_parent_name in op.parents:
            op_parent = op_dict[op_parent_name]
            if mode==LPMode.Mode_Subgraph and not isinstance(op_parent, subgraph.Subgraph):
                continue
            if op_parent_name not in one_module_names_idx_dict.keys():
                continue
            parent_idx = one_module_names_idx_dict[op_parent_name]
            parent_output_tensors_addr = [paddr for (paddr, _) in op_parent.output_tensors]
            if addr in parent_output_tensors_addr:
                acc_data_trans_latency += data_trans_latency
                parent_idx_data_trans.append((parent_idx, data_trans_latency))
    return (acc_data_trans_latency, parent_idx_data_trans)


def get_input_tensor_data_trans_latency(op_name, op_dict, device):
    op = op_dict[op_name]
    acc_data_trans_latency = 0.0
    assert(device in [CPU, GPU])
    for (addr, data_trans) in op.op_def.operator_latency.input_data_trans_latency.items():
        acc_data_trans_latency += data_trans[device]
    return acc_data_trans_latency


# Generate constraints for the "tt > node finish time"
def generate_final_latency_for_one_node(op_name, one_module_names_idx_dict,
                                        device_list, op_dict):
    constraints = []
    op = op_dict[op_name]
    idx = one_module_names_idx_dict[op_name]
    
    for device in device_list:
        assert(device in [CPU, GPU])
        (acc_data_trans_latency, parent_idx_data_trans) = get_parent_idxes_and_data_trans(op_name, one_module_names_idx_dict, op_dict, device)    
        c1 = "tt + %d s_%d_%d > 0.0\n" % (M, device, idx)
        c2 = ""

        if device == CPU:
            device_latency = op.op_def.operator_latency.CPU_latency
        elif device == GPU:
            device_latency = op.op_def.operator_latency.GPU_latency
        
        lp_data_trans = ""
        for (parent_idx, data_trans_latency) in parent_idx_data_trans:
            lp_data_trans += " + %f s_%d_%d " % (data_trans_latency, device, parent_idx)
        c2 = "tt - t_%d_%d - %f s_%d_%d %s > %f\n" \
            % (device, idx, M + acc_data_trans_latency, device, idx, lp_data_trans, (device_latency - M))
        constraints.extend([c1, c2])
    return constraints


# One node can only be executed once, so the sum of s_i_j is equal to 1
def generate_node_execute_once(one_module_names_idx_dict, device_list):
    constraints = []
    for _, idx in one_module_names_idx_dict.items():
        s = ""
        for device in device_list:
            if device == device_list[-1]:
                s += ("s_%d_%d = 1\n" % (device, idx))
            else:
                s += ("s_%d_%d + " % (device, idx))
        constraints.append(s)
    return constraints


# Find if op_name_b is op_name_a's ancestor
# if not ancestor, a and b can not execute on device at the same time
# else, use DAG constrains
def have_relative_relation(one_module_names_idx_dict, op_name_a, op_name_b,
                           op_dict):
    op_queue = queue.Queue()
    for parent in op_dict[op_name_a].parents:
        if parent in one_module_names_idx_dict.keys():
            op_queue.put(parent)
    while not op_queue.empty():
        op_parent_name = op_queue.get()
        if op_parent_name == op_name_b:
            return True
        op_parent = op_dict[op_parent_name]
        for parent in op_parent.parents:
            if parent in one_module_names_idx_dict.keys():
                op_queue.put(parent)
    return False


# Find if op_name_b is op_name_a's parent
def have_parent_relation(op_name_a, op_name_b, op_dict):
    op_a = op_dict[op_name_a]
    if op_name_b in op_a.parents:
        return True
    else:
        return False


# Generate constraints for parent and child operations.
# Parent and child constraints can both cover the DAG constrains and the device execute one device at a time constrains
# We have considered the format convert
def generate_parent_and_child_constraints(one_module_names_idx_dict,
                                          device_list, op_name_child,
                                          op_name_parent, op_dict):
    idx_parent = one_module_names_idx_dict[op_name_parent]
    idx_child = one_module_names_idx_dict[op_name_child]
    # idx_parent_parent = 0
    constraints = []
    for device1 in device_list:
        for device2 in device_list:
            c = ""
            op_parent_latency = op_dict[op_name_parent].op_def.operator_latency
            
            device_latency = 0.0
            if device2 == CPU:
                device_latency = op_parent_latency.CPU_latency
            elif device2 == GPU:
                device_latency = op_parent_latency.GPU_latency
            (acc_data_trans_latency, parent_idx_data_trans) = get_parent_idxes_and_data_trans(op_name_parent, one_module_names_idx_dict, op_dict, device2)
            lp_data_trans = ""
            for (parent_idx, data_trans_latency) in parent_idx_data_trans:
                lp_data_trans += " + %f s_%d_%d " % (data_trans_latency, device2, parent_idx)
            c = "t_%d_%d - t_%d_%d - %f s_%d_%d %s > %f\n" \
                % (device1, idx_child, device2, idx_parent, (M + acc_data_trans_latency), device2, idx_parent, \
                    lp_data_trans, device_latency - M)
            
            constraints.append(c)
    return constraints


def get_parent_idxes(one_module_names_idx_dict, op_name, op_dict):
    idx_parent = []
    for parent in op_dict[op_name].parents:
        if parent in one_module_names_idx_dict.keys():
            idx_parent.append(one_module_names_idx_dict[parent])
    return idx_parent


def get_parent_idx(one_module_names_idx_dict, op_name, op_dict):
    idx_parent = 0
    for parent in op_dict[op_name].parents:
        if parent in one_module_names_idx_dict.keys():
            idx_parent = one_module_names_idx_dict[parent]
    return idx_parent


def generate_one_node_at_a_device(one_module_names_idx_dict, op_name_a,
                                  op_name_b, device_list, op_dict):
    idx_a = one_module_names_idx_dict[op_name_a]
    idx_b = one_module_names_idx_dict[op_name_b]
    # idx_a_parent = get_parent_idx(one_module_names_idx_dict, op_name_a,
    #                               op_dict)
    idx_b_parent = get_parent_idx(one_module_names_idx_dict, op_name_b,
                                  op_dict)
    b_cpu_latency = op_dict[op_name_b].op_def.operator_latency.CPU_latency
    # a_cpu_latency = op_dict[op_name_a].op_def.operator_latency.CPU_latency
    b_gpu_latency = op_dict[op_name_b].op_def.operator_latency.GPU_latency
    # a_gpu_latency = op_dict[op_name_a].op_def.operator_latency.GPU_latency
    constraints = []
    u_variable = []

    for device in device_list:
        c = ""
        li = [idx_a, idx_b]
        li.sort()
        [u_idx_a, u_idx_b] = li
        u_val_str = "u_%d_%d_%d" % (device, u_idx_a, u_idx_b)
        cond_val_str = "r_%d_%d_%d" % (device, u_idx_a, u_idx_b)
        assert(idx_a != idx_b)
        if u_val_str not in u_variable:
            u_variable.append(u_val_str+"\n")
            u_variable.append(cond_val_str+"\n")
        # Judge whether the two op are executed on the same device
        # 0 <= s1 + s2 -2 + Kx <= K - 1
        # from: https://blog.adamfurmanek.pl/2015/09/12/ilp-part-4/
        judge_constraint1 = "s_%d_%d + s_%d_%d + %d %s >= 2\n" % (device, idx_a, device, idx_b, M, cond_val_str)
        judge_constraint2 = "- s_%d_%d - s_%d_%d - %d %s >= %d\n" % (device, idx_a, device, idx_b, M, cond_val_str, - M - 1)

        device_latency = 0.0
        if device == CPU:
            device_latency = b_cpu_latency
        elif device ==GPU:
            device_latency = b_gpu_latency
        
        (acc_data_trans_latency, parent_idx_data_trans) = get_parent_idxes_and_data_trans(op_name_b, one_module_names_idx_dict, op_dict, device)
        lp_data_trans = ""
        for (parent_idx, data_trans_latency) in parent_idx_data_trans:
            lp_data_trans += " + %f s_%d_%d " % (data_trans_latency, device, parent_idx)
        
        if idx_a > idx_b:
            c = "t_%d_%d - t_%d_%d + %f %s - %f s_%d_%d %s + %f %s > %f\n" \
                % (device, idx_a, device, idx_b, M, u_val_str, \
                acc_data_trans_latency, device, idx_b, lp_data_trans, K, cond_val_str, device_latency)
        else:
            c = "t_%d_%d - t_%d_%d - %f %s - %f s_%d_%d %s + %f %s > %f\n" \
                 % (device, idx_a, device, idx_b, M, u_val_str, \
                    acc_data_trans_latency, device, idx_b, lp_data_trans, K, cond_val_str, device_latency - M)
        constraints.extend([judge_constraint1, judge_constraint2, c])

    return constraints, u_variable


# One device can only execute one op at a time
# If op_a is op_b's child, then we can simply use the DAG constraints
def generate_device_execute_once_at_a_time(one_module_names_idx_dict,
                                           device_list, op_dict):
    constraints = []
    u_variables = []
    for op_name_a, idx_a in one_module_names_idx_dict.items():
        for op_name_b, idx_b in one_module_names_idx_dict.items():
            if op_name_a == op_name_b:
                continue
            # Generate DAG constraints
            if have_parent_relation(op_name_a, op_name_b, op_dict):
                constraints.extend( \
                    generate_parent_and_child_constraints(one_module_names_idx_dict, device_list, op_name_a, op_name_b, op_dict))
            elif have_parent_relation(op_name_b, op_name_a, op_dict):
                constraints.extend( \
                    generate_parent_and_child_constraints(one_module_names_idx_dict, device_list, op_name_b, op_name_a, op_dict))
            # Generate one device constraints
            if not have_relative_relation(one_module_names_idx_dict, op_name_a, op_name_b, op_dict) and \
                not have_relative_relation(one_module_names_idx_dict, op_name_b, op_name_a, op_dict):
                constraints_one_device, u_variable_one_device =  generate_one_node_at_a_device( \
                        one_module_names_idx_dict, op_name_a, op_name_b, device_list, op_dict)
                constraints.extend(constraints_one_device)
                u_variables.extend(u_variable_one_device)
    return constraints, u_variables


def generate_concat():
    pass


def generate_binary(one_module_names_idx_dict, device_list):
    binary_content = ["Binary\n"]
    for op_name, idx in one_module_names_idx_dict.items():
        for device in device_list:
            c = "s_%d_%d\n" % (device, idx)
            binary_content.append(c)
    return binary_content


def print_op_profile(one_module_names_idx_dict, op_dict):
    op_profile_list = []
    for op_name, idx in one_module_names_idx_dict.items():
        op = op_dict[op_name]
        op_profile_list.append((op_name, idx, op.op_def.operator_latency))
    op_profile_list = sorted(op_profile_list,
                             key=lambda op_profile: op_profile[1])

    for op_profile in op_profile_list:
        (op_name, idx, op.op_def.operator_latency) = op_profile
        print("%s %d %s" % (op_name, idx, op.op_def.operator_latency))


def generateLP(one_module_names_idx_dict, op_name_list, op_dict, net_def):

    # Print the relationship between op_name and index
    print_op_profile(one_module_names_idx_dict, op_dict)

    # Set s_cpu_%d of 'concat' to 1
    concat_constraint = ""
    for op_name, idx in one_module_names_idx_dict.items():
        if op_name.strip().split("/")[-1] == "concat":
            concat_constraint = ("s_%d_%d = 1\n" % (CPU, idx))
    LP_contents = []
    LP_objective = "Minimize\n\tvalue: tt\n\n"
    LP_constraints = ["Subject to\n"]
    LP_constraints.append(concat_constraint)
    # Generate for all op one all devices
    # 1 for CPU, 2 for GPU
    device_list = [CPU, GPU]
    for op_name, idx in one_module_names_idx_dict.items():
        LP_constraints.extend(
            generate_final_latency_for_one_node(op_name,
                                                one_module_names_idx_dict,
                                                device_list, op_dict))

    LP_constraints.extend(
        generate_node_execute_once(one_module_names_idx_dict, device_list))
    device_once_at_a_time, u_variable = generate_device_execute_once_at_a_time(
        one_module_names_idx_dict, device_list, op_dict)
    LP_constraints.extend(device_once_at_a_time)

    binary_content = generate_binary(one_module_names_idx_dict, device_list)
    binary_content.extend(u_variable)
    # Remove the dulplicate variables
    binary_content = sorted(set(binary_content), key=binary_content.index)
    LP_constraints = sorted(set(LP_constraints), key=LP_constraints.index)

    LP_contents.extend(LP_objective)
    LP_contents.extend(LP_constraints)
    LP_contents.extend(binary_content)
    LP_contents.append("\nEnd\n")
    
    return LP_contents



def run_glpsol(lp_file_path, result_file_path):
    glpsol_file_path = "glpsol"
    cmd_str = '%s --lp %s -o %s' % (glpsol_file_path, lp_file_path, result_file_path)
    logger.info("Execute %s" % (cmd_str))
    os.system(cmd_str)
    logger.info("Run solver done!")


# The result file are in follows:
# '4 s_1_6        *              1             0             1 '
def parse_glpk_result(one_module_names_idx_dict, result_file_path):
    f = open(result_file_path, 'r')
    name_device_tuple_list = []
    lines = f.readlines()
    for line in lines:
        # Find lines with s_
        line = line.strip()
        if line.find('s_') >= 0:
            com = line.split(' ')
            striped_com = []
            for c in com:
                if c != '':
                    striped_com.append(c)
            
            if len(striped_com) == 6 and striped_com[3] == '1':
                # print(striped_com)
                s = striped_com[1]
                sc = s.split('_')
                if int(sc[1]) == CPU:
                    device = 0
                elif int(sc[1]) == GPU:
                    device = 3
                op_idx = sc[2]
                for name, idx in one_module_names_idx_dict.items():
                    if int(op_idx) == idx:
                        name_device_tuple_list.append((name, device))
                        break
    for (name, device) in name_device_tuple_list:
        print("%s %d" %(name, device))
    return name_device_tuple_list

# @pysnooper.snoop()
def compute_data_trans_intersection(cpu_data, gpu_data, convert_data, convert_device):
    def add_latency(data):
        return [(start, start+latency) for (start, latency) in data]
    def max_time(data):
        if len(data) == 0:
            return 0
        else:
            return max([endtime for (_, endtime) in data])
    cpu_data = add_latency(cpu_data)
    gpu_data = add_latency(gpu_data)
    convert_data = add_latency(convert_data)
    print(cpu_data)
    print(gpu_data)
    print(convert_data)
    sum_of_intersection = 0
    for (cs, ce), device in list(zip(convert_data, convert_device)):
        for (gs, ge) in gpu_data:
            # if op execute on GPU, skip 
            if device == 2:
                continue
            if cs >= gs and cs <= ge:
                sum_of_intersection += (min(ce, ge) - cs)
            elif gs > cs and gs < ce:
                sum_of_intersection += (min(ce, ge) - gs)
    cpu_max, gpu_max, convert_max = max_time(cpu_data), max_time(gpu_data), max_time(convert_data)
    endpoint = max([cpu_max, gpu_max, convert_max])

    return endpoint, sum_of_intersection




def parse_glpk_timeline(one_module_names_idx_dict, result_file_path, op_dict, mode=LPMode.Mode_Subgraph):
    idx_name_dict = {v:k for k,v in one_module_names_idx_dict.items()}
    print(idx_name_dict)

    s_dict = {}
    t_dict = {}
    f = open(result_file_path, 'r')
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line.find('s_') >=0 or line.find('t_') >=0:
            com = line.split(' ')
            striped_com = []
            for c in com:
                if c != '':
                    striped_com.append(c)
            if striped_com[1].find('s_')>=0 and striped_com[3] == '1':
                sc = striped_com[1].split('_')
                device, idx = sc[1], sc[2]
                s_dict[int(device), int(idx)] = int(striped_com[3])
            elif striped_com[1].find('t_')>=0:
                sc = striped_com[1].split('_')
                device, idx = sc[1], sc[2]
                t_dict[(int(device), int(idx))] = float(striped_com[2])
    result = []
    op_execution_order_list = []
    for (device, idx), _ in s_dict.items():
        start_time = t_dict[(device, idx)]
        op_name = idx_name_dict[idx]
        op = op_dict[op_name]
        device_latency = 0.0
        if device == CPU:
            device_latency = op.op_def.operator_latency.CPU_latency
        elif device == GPU:
            device_latency = op.op_def.operator_latency.GPU_latency
        # Compute data trans latency
        acc_data_trans_latency = 0.0
        for (addr, data_trans) in op.op_def.operator_latency.input_data_trans_latency.items():
            data_trans_latency = data_trans[device-1]
            for op_parent_name in op.parents:
                op_parent = op_dict[op_parent_name]
                if mode==LPMode.Mode_Subgraph and not isinstance(op_parent, subgraph.Subgraph):
                    continue
                if op_parent_name not in one_module_names_idx_dict.keys():
                    continue
                parent_idx = one_module_names_idx_dict[op_parent_name]
                if device == CPU and (GPU, parent_idx) not in s_dict.keys():
                    continue
                elif device == GPU and (CPU, parent_idx) not in s_dict.keys():
                    continue
                parent_output_tensors_addr = [paddr for (paddr, _) in op_parent.output_tensors]
                if addr in parent_output_tensors_addr:
                    acc_data_trans_latency += data_trans_latency
                    break
        
        result.append((op_name, device, start_time, device_latency, acc_data_trans_latency))
        op_execution_order_list.append((op_name, device, start_time))
    # Get the schedule result from t_device_op
    result = sorted(result, key=lambda x: x[2])
    op_execution_order_list = sorted(op_execution_order_list, key=lambda x: x[2])
    for t in op_execution_order_list:
        print(t)
    cpu_data, gpu_data, convert_data, convert_device = [], [], [], []
    for (op_name, device, start_time, device_latency, acc_data_trans_latency) in result:
        print("name:{}, device:{}, start_time:{}, latency:{}, data_trans:{}"\
            .format(op_name, device, start_time, device_latency, acc_data_trans_latency))
        if device == 1:
            cpu_data.append((start_time, device_latency))
        elif device == 2:
            gpu_data.append((start_time, device_latency))
        if acc_data_trans_latency >0 :
            convert_data.append((start_time+device_latency, acc_data_trans_latency))
            convert_device.append(device)
    
    return cpu_data, gpu_data, convert_data, convert_device, op_execution_order_list




def solve_glpk(op_name_list, name_op_dict, net_def, module_name_list, folder_path, model_name, mode=LPMode.Mode_Subgraph):
    lines = []
    intersection_list = []
    name_weight_dict, result = find_critical_node.get_node_weight_wrapper(model)
    for module_name in module_name_list:
        one_module_names_idx_dict = {}
        if mode == LPMode.Mode_Subgraph:
            # For one module with multiple subgraphs, we need build subgraph and update the op_dict
            parent_subgraph = subgraph.Subgraph(module_name)
            if model_name != None and model_name.find("inception") >=0:
                one_module_name, branches = subgraph.get_inception_one_module_name(module_name, op_name_list)
                parent_subgraph.buildMultiSubgraph(op_name_list, name_op_dict, branches, pattern=module_name)
            elif model_name != None and (model_name.find("pnasnet") >=0 or model_name.find("nasnet") >=0):
                parent_subgraph.buildMultiSubgraph(op_name_list, name_op_dict, subgraph.pnasnet_mobile_subgraph_subprefix(), pattern=module_name)
            # one_module_names_idx_dict = associate_op_name_with_idx(
            #     "/mnt/d/home/Projects/DAG-scheduler/mnn/pnasnet-mobile/pnasnet-cell-0-subgraph-names.txt")
            print("aaa", parent_subgraph.op_name_list)
            for subgraph_name in parent_subgraph.op_name_list:
                print(name_op_dict[subgraph_name])
            one_module_names_idx_dict = associate_op_name_list_with_idx(parent_subgraph.op_name_list)
        elif mode == LPMode.Mode_Operator:
            one_module_names_idx_dict = associate_op_name_list_with_idx(subgraph.filter_op_name_list(op_name_list, module_name))
        elif mode == LPMode.Mode_AUTO_Subgraph:
            module_op_name_list = []
            for op_name in op_name_list:
                if op_name.find(module_name) == 0:
                    module_op_name_list.append(op_name)
            
            subgraph_list = graph_partition.auto_build_multi_subgrap_with_weight(module_op_name_list, name_op_dict, name_weight_dict, 972000)
            if len(subgraph_list) > 11:
                continue
            print([subgraph.name for subgraph in subgraph_list])
            one_module_names_idx_dict = associate_op_name_list_with_idx([subgraph.name for subgraph in subgraph_list])
            # exit(0)
        # Generate LP constraints and write them to a file
        LP_contents = generateLP(one_module_names_idx_dict, op_name_list, name_op_dict, net_def)

        module_name_striped = module_name.replace('/','-')
        if module_name_striped[-1] == '-':
            module_name_striped = module_name_striped[0:len(module_name_striped)-1]
            module_name_striped = module_name_striped.split('-')[-1]
        lp_file_path = os.path.join(folder_path, "subgraphs-" + module_name_striped + ".lp")
        result_file_path = os.path.join(folder_path, "lp-result-subgraphs-" + module_name_striped+ ".txt")
        logger.info("Write Integer Linear Programming models to {}".format(lp_file_path))
        utils.write_lines(lp_file_path, LP_contents)
        # Solve the LP
        # run_glpsol(lp_file_path, result_file_path)
        # Parse subgraph device placement result
        # name_device_tuple_list = parse_glpk_result(one_module_names_idx_dict, result_file_path)
        # print(name_device_tuple_list)
        cpu_data, gpu_data, convert_data, convert_device, op_execution_order_list = parse_glpk_timeline(one_module_names_idx_dict, result_file_path, name_op_dict, mode=mode)
        endpoint, sum_of_intersection = compute_data_trans_intersection(cpu_data, gpu_data, convert_data, convert_device)
        intersection_list.append((endpoint, sum_of_intersection))
        print("endpoint: %f , intersection: %f" % (endpoint, sum_of_intersection))
        print("module_name")
    
        tmp_module_name_list = []
        for c in module_name.split("/"):
            if len(c.strip()) > 0:
                tmp_module_name_list.append(c)
        
        draw_gantt(cpu_data, gpu_data, convert_data, os.path.join(folder_path, tmp_module_name_list[-1]))
        
        # device_placement_file_path = os.path.join(folder_path, "mDeviceMap-"+ "subgraphs-" + model_name + "-" + module_name_striped +".txt")
        results = subgraph.write_subgraph_device_placement_result([name for (name, device, start_time) in op_execution_order_list if device == CPU],\
            [name for (name, device, start_time) in op_execution_order_list if device == GPU], \
            name_op_dict, op_execution_order_list=op_execution_order_list)
        lines.extend(results)
        # print("Write result to %s" % (device_placement_file_path))
    return lines, intersection_list


def sum_lp_objectives(folder_path):
    grep_cmd = "cat %s | grep Objective | awk '{print $4}'" % os.path.join(folder_path, "lp-result-subgraphs-*")
    print("Execute "+grep_cmd)
    lp_result = os.popen(grep_cmd).read()
    print(lp_result)
    com = lp_result.split("\n")
    total = 0.
    for c in com:
        if c == '' or len(c) == 0:
            continue
        total += float(c)
    return total


def solve_pnasnet(model, mobile, thread, CPU_little_thread_index=None):
    # Read profile data
    model_dir = os.path.join("../models/", model)
    op_name_list, name_op_dict = read_profile_data.load_model_profile(model, mobile, thread, \
        SCALE=1.0, CPU_little_thread_index=CPU_little_thread_index)
    net_def = None
    
    # Using module prefix to form the subgraph
    pnasnet_module_list = ['cell_stem_0/', 'cell_stem_1/']
    if model == 'pnasnet-large':
        pnasnet_module_list.extend(['cell_'+str(i)+'/' for i in range(12)])
    elif model == 'pnasnet-mobile':
        pnasnet_module_list.extend(['cell_'+str(i)+'/' for i in range(9)])
    elif model == 'nasnet-large':
        pnasnet_module_list.extend(['cell_'+str(i)+'/' for i in range(18)])
        pnasnet_module_list.insert(8, 'reduction_cell_0/')
        pnasnet_module_list.insert(15, 'reduction_cell_1/')
        print('aaa', pnasnet_module_list)
    elif model == 'nasnet-mobile':
        pnasnet_module_list.extend(['cell_'+str(i)+'/' for i in range(12)])
        pnasnet_module_list.insert(6, 'reduction_cell_0/')
        pnasnet_module_list.insert(11, 'reduction_cell_1/')
    else:
        print("Model %s does not suport yet." % (model))
        return
    
    # Using GLPK solve device placement here
    folder_path = os.path.join(model_dir, mobile)
    lines, intersection_list = solve_glpk(op_name_list, name_op_dict, net_def, pnasnet_module_list, folder_path, model, mode=LPMode.Mode_Operator)
    unsupported_op_names = ["final_layer/Relu", "final_layer/Mean/reduction_indices", \
        "final_layer/Relu___tr4final_layer/Mean", "final_layer/Mean", \
        "final_layer/FC/weights", "final_layer/FC/MatMul", \
        "final_layer/FC/biases", "final_layer/FC/BiasAdd", "final_layer/predictions"]
    # Deal with ops that are not in the module prefix
    lines, untreated_op_latency = subgraph.insert_untreated_ops(lines, op_name_list, name_op_dict, unsupported_op_names=unsupported_op_names)
    lp_total = sum_lp_objectives(folder_path)
    
    # Write results
    device_map_file_path = os.path.join(model_dir, mobile, "mDeviceMap-{}-cpu-{}.txt".format(model, thread))
    new_lines = []
    if CPU_little_thread_index != None:
        device_map_file_path = os.path.join(model_dir, mobile, \
            "mDeviceMap-{}-cpu-big-{}-little-{}.txt".format(model, thread, CPU_little_thread_index))
        # Little core device type is 2
        for line in lines:
            line = line.replace(' 3', ' 2')
            new_lines.append(line)
        lines = new_lines
    
    utils.write_lines(device_map_file_path, lines)
    sh_cmd = "adb push {} /data/local/tmp/".format(device_map_file_path)
    print(sh_cmd)
    os.system(sh_cmd)
    print(untreated_op_latency)
    print("LP+serial total: {}".format(lp_total+untreated_op_latency))
    print("LP+serial total: {}".format(lp_total+untreated_op_latency+sum([intersection for _, intersection in intersection_list])))


def solve_inception(model, mobile, thread, CPU_little_thread_index=None):
    model_dir = os.path.join("../models/", model)
    op_name_list, name_op_dict = read_profile_data.load_model_profile(model, mobile, thread, \
        SCALE=1.0, CPU_little_thread_index=CPU_little_thread_index)
    net_def = None
    
    if model == "inception-v3":
        inception_prefix = "InceptionV3/InceptionV3/"
        inception_module_list = ["Mixed_5b/", "Mixed_5c/", "Mixed_5d/", \
            "Mixed_6a/", "Mixed_6b/", "Mixed_6c/", "Mixed_6d/", "Mixed_6e/", "Mixed_7a/", "Mixed_7b/", "Mixed_7c/"]
    elif model == "inception-v4":
        inception_prefix = "InceptionV4/InceptionV4/"
        inception_module_list = ["Mixed_4a/", "Mixed_5b/", "Mixed_5c/", "Mixed_5d/", "Mixed_5e/", \
            "Mixed_6a/", "Mixed_6b/", "Mixed_6c/", "Mixed_6d/", "Mixed_6e/","Mixed_6f/","Mixed_6g/","Mixed_6h/",\
            "Mixed_7a/","Mixed_7b/","Mixed_7c/","Mixed_7d/",]
    inception_module_list = [inception_prefix + module for module in inception_module_list]
    folder_path = os.path.join(model_dir, mobile)
    lines, intersection_list = solve_glpk(op_name_list, name_op_dict, net_def, inception_module_list, folder_path, model, mode=LPMode.Mode_Operator)
    lines, untreated_op_latency = subgraph.insert_untreated_ops(lines, op_name_list, name_op_dict)
    lp_total = sum_lp_objectives(folder_path)
    print("LP+serial total: {}".format(lp_total+untreated_op_latency))
    print("LP+serial+intersection total: {}".format(sum([(endpoint+intersection) for endpoint, intersection in intersection_list]) + untreated_op_latency))
    device_map_file_path = os.path.join(model_dir, mobile, "mDeviceMap-{}-cpu-{}.txt".format(model, thread))
    new_lines = []
    if CPU_little_thread_index != None:
        device_map_file_path = os.path.join(model_dir, mobile, \
            "mDeviceMap-{}-cpu-big-{}-little-{}.txt".format(model, thread, CPU_little_thread_index))
        # Little core device type is 2
        for line in lines:
            line = line.replace(' 3', ' 2')
            new_lines.append(line)
        lines = new_lines

    utils.write_lines(device_map_file_path, lines)

    sh_cmd = "adb push {} /data/local/tmp/".format(device_map_file_path)
    print(sh_cmd)
    os.system(sh_cmd)


if __name__ == "__main__":
    model, mobile, thread, CPU_little_thread_index = utils.parse_model_mobile()
    # model, mobile, thread = "inception-v3", "oneplus5t", 2
    if model in ['pnasnet-mobile', 'pnasnet-large', 'nasnet-large', 'nasnet-mobile']:
        solve_pnasnet(model, mobile, thread, CPU_little_thread_index=CPU_little_thread_index)
    elif model in ['inception-v3', 'inception-v4']:
        solve_inception(model, mobile, thread, CPU_little_thread_index=CPU_little_thread_index)
        # solve_inception(model, mobile, thread)
    