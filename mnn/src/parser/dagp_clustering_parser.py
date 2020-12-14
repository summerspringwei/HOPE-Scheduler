import os

# from profile import read_profile_data
from visualization import draw_dag
from utils import utils

def read_node_mappings(model):
    """Read the node mapping file genenrated by dagP
    Returns:
        The idx->op_name map
    """
    model_dir = os.path.join("/Users/xiachunwei/Projects/DAG-Scheduler/mnn/models/", model)
    node_mapping_file_path = os.path.join(model_dir, "{}.dot.nodemappings".format(model))
    f = open(node_mapping_file_path, 'r')
    lines = f.readlines()
    idx_name_dict = dict()
    for line in lines:
        name, idx = line.strip().split(" ")
        # Strip the quotation mark
        name = name[1:len(name)-1]
        idx_name_dict[int(idx)] = name
    f.close()
    return idx_name_dict


def read_part_mappings(model, parts):
    """Read the partsfile generated by dagP
    Returns:
        The idx->part_idx map
    """
    model_dir = os.path.join("/Users/xiachunwei/Projects/DAG-Scheduler/mnn/models/", model)
    part_file_path = os.path.join(model_dir, "{}.dot.partsfile.part_{}.seed_0.txt".format(model, parts))
    f = open(part_file_path, 'r')
    lines = f.readlines()
    idx = 1
    idx_part_dict = dict()
    for line in lines:
        idx_part_dict[idx] = int(line.strip())
        idx += 1
    f.close()
    return idx_part_dict



def dagp_clustering_parser(model, parts):
    """Combine the node mapping result and parts file
    Returns:
        The dict of clusters containing the operator names <cluster_id, {op names in a cluster}>
    """
    idx_name_dict = read_node_mappings(model)
    idx_part_dict = read_part_mappings(model, parts)
    # print(idx_name_dict)
    # print(idx_part_dict)
    parts_set = set()
    for idx, part in idx_part_dict.items():
        parts_set.add(part)
    
    clusters_list = dict()
    
    for idx, part in idx_part_dict.items():
        if part not in clusters_list.keys():
            clusters_list[part] = [idx_name_dict[idx]]
        else:
            clusters_list[part].append(idx_name_dict[idx])
    for k, cl in clusters_list.items():
        print(k, len(cl))
    return clusters_list


def dagp_clustering_model(model, mobile, thread, parts, set_weight=True):
    graphviz_file_path = draw_dag.generate_model_figures(model, mobile, thread, set_weight=set_weight)
    dagp_cmd = "cd ~/Projects/dagP && rMLGP {} {} --toggle 1 --debug 300 --write_parts 1 --print 3 --conpar 0".format(graphviz_file_path, parts)
    print(dagp_cmd)
    os.system(dagp_cmd)
    clusters_list = dagp_clustering_parser(model, parts)
    




if __name__=="__main__":
    # print(dagp_clustering_parser("inception-v4", 16))
    # clusters_list = dagp_clustering_parser("inception-v4", 16)
    # for k, names in clusters_list.items():
    #     print(k, names)
    model, mobile, thread, _ = utils.parse_model_mobile()
    dagp_clustering_model(model, mobile, thread, 16, set_weight=True)