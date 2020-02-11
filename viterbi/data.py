import os
import yaml

from edflow.main import get_obj_from_str


def get_input_data(eval_root):
    csv_path = os.path.join(eval_root, 'model_output.csv')
    meta_path = os.path.join(eval_root, 'meta.yaml')

    if os.path.exists(csv_path):
        with open(csv_path, 'r') as cf:
            yaml_string = ''
            for line in cf.readlines():
                if "# " in line:
                    yaml_string += line[2:] + "\n"
                else:
                    break
        config = yaml.full_load(yaml_string)

    elif os.path.exists(meta_path):
        config = yaml.full_load(meta_path)

    else:
        raise ValueError('eval_root must point to a folder containing a csv of meta')

    impl = get_obj_from_str(config["dataset"])
    in_data = impl(config)

    return in_data


def get_out_data(eval_root):
    from viterbi.transition_network.encodings_dataset import evaluation_encodings

    return evaluation_encodings({'data_folder': eval_root})


if __name__ == '__main__':

    from edflow.util import edprint

    eval_root = "/export/data/rmarwaha/projects/logs/2019-11-20T14-46-23_hg_disc/eval/2020-01-24T14-52-25_pose_enc"
    di = get_input_data(os.path.join(eval_root, '0'))

    edprint(di.labels)

    do = get_out_data(eval_root)

    edprint(do.labels)
