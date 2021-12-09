#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import pathlib
import sys


def parse_arguments(argv: list) -> dict:
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input-folder",
                        action="store",
                        default=None, type=str,
                        metavar=("PATH/TO/INPUT/"),
                        dest="input_folder",
                        required=True,
                        help="Path to the input folder.")

    parser.add_argument("-o", "--output-csv",
                        action="store",
                        default=None, type=str,
                        metavar=("PATH/TO/OUTPUT.CSV"),
                        dest="output_csv",
                        help="Path to save the evaluation csv to.")

    parser.add_argument("-a", "--output-images",
                        action="store",
                        default=None, type=str,
                        metavar=("PATH/TO/OUTPUT/"),
                        dest="output_images",
                        help="Path to save the annotated images to.")

    parser.add_argument("-t", "--data-type",
                        action="store",
                        default="structured", type=str,
                        metavar=("structured|unstructured"),
                        dest="data_type",
                        help="Type of data in the input folder.")

    parser.add_argument("-p", "--prediction-models",
                        action="store",
                        default=[ "image", "feature" ], type=lambda x: x.split(","),
                        metavar=("image|feature"),
                        dest="prediction_models",
                        help="Models to make the predictions with. Multiple may be "
                             "specified in a comma separated list.")

    parser.add_argument("--python-path",
                        action="store",
                        default="/opt/conda/envs/pt/bin/python3", type=str,
                        metavar=("PATH/TO/PYTHON"),
                        dest="python_path",
                        help="Path to python version to run the prediction with.")

    parser.add_argument("--python-envs",
                        action="store",
                        default="PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 DISPLAY=False", type=str,
                        metavar=("ENV1=VAL1..."),
                        dest="python_envs",
                        help="Environmental variables used when running the prediction under python.")

    def parse_model_path(val: str) -> (str, str):
        splits = val.split(":")
        return splits[0], splits[1]

    default_model_locations = [
        parse_model_path("feature:/projects/data/feature.tpm"),
        parse_model_path("image:/projects/data/image.tpm"),
    ]
    parser.add_argument("--prediction-model-path",
                        action="append",
                        default=None, type=parse_model_path,
                        metavar=("[image|feature]:/PATH/TO/MODEL.TPM"),
                        dest="prediction_model_paths",
                        help="Specification of which pre-trained models to use.")

    parsed, extra_args = parser.parse_known_args(args=argv[1:])

    arguments = { }

    arguments["input_folder"] = parsed.input_folder
    arguments["output_csv"] = parsed.output_csv
    arguments["output_images"] = parsed.output_images
    arguments["data_type"] = parsed.data_type
    arguments["prediction_models"] = parsed.prediction_models
    arguments["python_path"] = parsed.python_path
    arguments["python_envs"] = parsed.python_envs

    prediction_model_paths = parsed.prediction_model_paths or default_model_locations
    prediction_model_paths = {
        model_type: model_path
        for model_type, model_path in prediction_model_paths
    }
    arguments["prediction_model_paths"] = prediction_model_paths

    for prediction_model in arguments["prediction_models"]:
        if prediction_model not in arguments["prediction_model_paths"]:
            raise RuntimeError(f"Prediction model \"{prediction_model}\" is requested, "
                               f"but its location is not specified!")
        model_path = pathlib.Path(arguments["prediction_model_paths"][prediction_model])
        if not model_path.exists():
            raise RuntimeError(f"Prediction model \"{prediction_model}\" is requested, "
                               f"but the file \"{model_path}\" does not exist!")

    extra_arg_dict = { }
    for arg in extra_args:
        split = arg.split(":")
        extra_arg_dict[split[0]] = extra_arg_dict.get(split[0], list()) + [ " ".join(split[1:]) ]

    arguments["extra_args"] = {
        name: " " + " ".join(args)
        for name, args in extra_arg_dict.items()
    }

    return arguments


def build_command(arguments: dict) -> str:
    base_path = pathlib.Path(__file__).parent.parent.absolute()

    command_base = "Logging -v --graphs-to-file-and-show True --logging-directory-use-timestamp False " \
                   "--logging-directory-use-model False --logging-name PerceptreePredict Data " \
                   "--load-node-data False Featurize --cross-validation-split 1/20/42"

    if "feature" in arguments["prediction_models"]:
        model_path = pathlib.Path(arguments["prediction_model_paths"]["feature"]).absolute()
        command_feature_predict = f"FeaturePredictor --load {model_path} --do-fit True " \
                                  f"--feature-normalize False --feature-standardize True --use-view-scores " \
                                  f"True --use-tree-variants True --use-view-variants True " \
                                  f"--pre-generate-variants False --binary-pretraining True " \
                                  f"--binary-pretraining-ptg 0.2 --differential-pretraining True " \
                                  f"--differential-pretraining-ptg 0.2 --feature-types stat,image,other " \
                                  f"--model-type dnn --feature-buckets 8 --feature-resolution 32"
        command_feature_predict += arguments["extra_args"].get("FeaturePredictor", "")
    else:
        command_feature_predict = ""

    if "image" in arguments["prediction_models"]:
        model_path = pathlib.Path(arguments["prediction_model_paths"]["image"]).absolute()
        command_image_predict = f"ImagePredictor --load {model_path} --do-fit True " \
                                f"--view-resolution 256 --use-view-scores True --use-tree-variants True " \
                                f"--use-view-variants True --pre-generate-variants False " \
                                f"--binary-pretraining True --differential-pretraining True " \
                                f"--differential-pretraining-ptg 0.5 --network-style res2net50_v1b " \
                                f"--network-pretrained False"
        command_image_predict += arguments["extra_args"].get("ImagePredictor", "")
    else:
        command_image_predict = ""

    if arguments["data_type"] == "structured":
        command_predict = f"Predict --predict-tree-folder {arguments['input_folder']} " \
                          f"--predict-views-folder {arguments['input_folder']}"
    elif arguments["data_type"] == "unstructured":
        command_predict = f"Predict --predict-views-folder-unstructured {arguments['input_folder']}"
    else:
        raise RuntimeError(f"Unknown data type \"{arguments['data_type']}\"!")
    command_predict += arguments["extra_args"].get("Predict", "")

    if arguments["output_csv"] is not None:
        command_evaluate = f"Evaluate --evaluate-current --export-evaluations {arguments['output_csv']}"
    else:
        command_evaluate = f"Evaluate --evaluate-current"
    command_evaluate += arguments["extra_args"].get("Evaluate", "")

    if arguments["output_images"] is not None:
        command_evaluate += f" --export-images {arguments['output_images']} --export-images-text True " \
                            f"--export-images-side l --export-images-transparent False " \
                            f"--export-images-show False"

    command_python = f"PYTHONPATH=. {arguments['python_envs']} {arguments['python_path']} -u "
    command_work_dir = f"{base_path}/psrc/"
    command_script = f"{base_path}/psrc/perceptree/perceptree_main.py"

    command = f"( cd {command_work_dir}; {command_python} {command_script} {command_base} " \
              f"{command_feature_predict} {command_image_predict} {command_predict} " \
              f"{command_evaluate}; )"

    return command


def main(argv: list) -> int:
    arguments = parse_arguments(argv=argv)

    command = build_command(arguments=arguments)
    print(f"Running command: \n{command}")

    result = os.system(command=command)

    return result


if __name__ == "__main__":
    exit(main(list(sys.argv)))
