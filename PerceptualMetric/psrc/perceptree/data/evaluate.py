# -*- coding: utf-8 -*-

"""
Evaluation of trained models.
"""

import io
import pathlib
from typing import Dict, List, Optional, Union, Tuple

from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import seaborn as sns
import scipy.stats as scs
import sklearn.metrics as sklm

from perceptree.common.cache import update_dict_recursively
from perceptree.common.configuration import Config
from perceptree.common.configuration import Configurable
from perceptree.common.file_saver import FileSaver
from perceptree.common.graph_saver import GraphSaver
from perceptree.common.logger import Logger
from perceptree.common.util import parse_bool_string
from perceptree.common.util import reshape_scalar
from perceptree.data.predict import Prediction
from perceptree.model.base import BaseModel


class EvaluationProcessor(Logger, Configurable):
    """
    Evaluation of trained models.
    """

    COMMAND_NAME = "Evaluate"
    """ Name of this command, used for configuration. """

    def __init__(self, config: Config):
        super().__init__(config=config)
        self._set_instance()

        self.__l.info("Initializing evaluation system...")

    @classmethod
    def register_options(cls, parser: Config.Parser):
        """ Register configuration options for this class. """

        parser.add_argument("--evaluate-current",
                            action="store_true",
                            default=False,
                            dest=cls._add_config_parameter("evaluate_current"),
                            help="Perform evaluation of all current model predictions?")

        parser.add_argument("--export-evaluations",
                            action="store",
                            default=None, type=str,
                            metavar=("PATH/TO/EVALUATIONS.csv"),
                            dest=cls._add_config_parameter("export_evaluations"),
                            help="Export evaluations data-frame to given csv.")

        parser.add_argument("--export-images",
                            action="store",
                            default=None, type=str,
                            metavar=("PATH/TO/OUTPUT/"),
                            dest=cls._add_config_parameter("export_images"),
                            help="Export annotated images with score bars to given location.")

        parser.add_argument("--export-images-text",
                            action="store",
                            default=True, type=parse_bool_string,
                            metavar=("True/False"),
                            dest=cls._add_config_parameter("export_images_text"),
                            help="Draw text in annotated images?")

        parser.add_argument("--export-images-side",
                            action="store",
                            default="l", type=str,
                            metavar=("r|l"),
                            dest=cls._add_config_parameter("export_images_side"),
                            help="Which side to display the bar on annotated images")

        parser.add_argument("--export-images-transparent",
                            action="store",
                            default=False, type=parse_bool_string,
                            metavar=("True/False"),
                            dest=cls._add_config_parameter("export_images_transparent"),
                            help="Should bars on annotated images be transparent?")

        parser.add_argument("--export-images-show",
                            action="store",
                            default=False, type=parse_bool_string,
                            metavar=("True/False"),
                            dest=cls._add_config_parameter("export_images_show"),
                            help="Display the annotated images through Matplotlib?")

    def _evaluate_prediction_score(self, prediction: Prediction) -> [ dict ]:
        """ Perform evaluation of tree scoring. """

        if prediction.score_prediction is None:
            self.__l.error("Failed to evaluate tree score, no prediction available!")
            return [ ]

        if prediction.score_expected is not None:
            ground_truth_score = prediction.score_expected
            ground_truth_available = True
        else:
            ground_truth_score = prediction.score_prediction
            ground_truth_available = False
        predicted_score = prediction.score_prediction

        if len(ground_truth_score) != len(predicted_score) and len(ground_truth_score) == 1:
            ground_truth_score = np.repeat(ground_truth_score, len(predicted_score))

        if len(ground_truth_score) != len(predicted_score):
            self.__l.error(f"Failed to evaluate tree score, GT and predicted don't match up "
                           f"({ground_truth_score} vs {predicted_score})!")

        return [
            {
                "tree_id": prediction.tree_id,
                "view_id": view_id,
                "score_gt": reshape_scalar(gt["jod"])[0],
                "score_p": reshape_scalar(pred["jod"])[0],
                "mse": sklm.mean_squared_error([ gt["jod"] ], [ pred["jod"] ]),
                "source": prediction.data_source,
                "true_gt": ground_truth_available,
                "tree_file": prediction.tree_file,
                "view_files": prediction.tree_views[view_id] if view_id in prediction.tree_views else { },
            }
            for view_id in ground_truth_score.keys()
            for gt, pred in [ ( ground_truth_score[view_id], predicted_score[view_id] ) ]
        ]

    def _process_evaluations(self, evaluations: pd.DataFrame):
        """ Process evaluation results. """

        evaluations = evaluations.drop([ "tree_file", "view_files" ], axis=1)

        if self.c.export_evaluations:
            evaluations.to_csv(self.c.export_evaluations, sep=";")
        evaluations_test = evaluations[evaluations["source"] != "train"]

        evaluations_desc = { }
        evaluation_stats = { }
        evaluation_text = { }

        for model_name in evaluations.model_name.unique():
            model_eval = evaluations.loc[evaluations.model_name == model_name]
            model_eval_test = evaluations_test.loc[evaluations_test.model_name == model_name]

            evaluations_desc[model_name] = f"Description: \n" + str(
                model_eval[["score_gt", "score_p", "mse"]].describe().to_string()
            )

            mse_a = evaluations["mse"].mean()
            mse_t = evaluations_test["mse"].mean()
            if len(model_eval) > 1:
                pear_a = tuple(scs.pearsonr(model_eval["score_p"], model_eval["score_gt"]))
                spea_a = tuple(scs.spearmanr(model_eval["score_p"], model_eval["score_gt"]))
            else:
                pear_a = tuple(( 1.0, 0.0 ))
                spea_a = tuple(( 1.0, 0.0 ))

            if len(model_eval_test) > 1:
                pear_t = tuple(scs.pearsonr(model_eval_test["score_p"], model_eval_test["score_gt"]))
                spea_t = tuple(scs.spearmanr(model_eval_test["score_p"], model_eval_test["score_gt"]))
            else:
                pear_t = tuple(( 1.0, 0.0 ))
                spea_t = tuple(( 1.0, 0.0 ))

            evaluation_stats[model_name] = \
                f"Statistics: \n" \
                f"\tMSE_a: {mse_a}\n" \
                f"\tMSE_t: {mse_t}\n" \
                f"\tPear_a: {pear_a}\n" \
                f"\tPear_t: {pear_t}\n" \
                f"\tSpea_a: {spea_a}\n" \
                f"\tSpea_t: {spea_t}"

            evaluation_text[model_name] = f"{evaluations_desc[model_name]}\n" \
                                          f"{evaluation_stats[model_name]}"

        model_evaluation_text = "\n".join([
            f"Evaluation for model \"{model_name}\": \n" + model_text
            for model_name, model_text in evaluation_text.items()
        ])
        all_evaluation_text = f"{str(evaluations.to_string())}\n{model_evaluation_text}"

        self.__l.info(f"Evaluation Results: \n{all_evaluation_text}")
        FileSaver.save_string("EvaluationResults", all_evaluation_text)
        FileSaver.save_csv("EvaluationResults", evaluations)

        fig, ax = plt.subplots()
        g = sns.lineplot(ax=ax, data=evaluations, x="score_gt", y="score_gt", color="red")
        plt.setp(g.lines, alpha=0.6)
        g = sns.lineplot(ax=ax, data=evaluations, x="score_gt", y="score_p")
        plt.setp(g.lines, alpha=0.3)
        all_sources = evaluations.source.unique()
        source_order = [
            source
            for source in [ "train", "valid", "test", "external", "unknown" ]
            if source in all_sources
        ]
        g = sns.scatterplot(ax=ax, data=evaluations, x="score_gt", y="score_p", hue="source", hue_order=source_order)
        logging_name = self.config["logging.logging_name"] or "Model"
        g.set_title(
            f"{logging_name} | MSE: {mse_a:.2} / {mse_t:.2} | "
            f"PE: {pear_a[0]:.2} / {pear_t[0]:.2} | "
            f"SP: {spea_a[0]:.2} / {spea_t[0]:.2}",
            fontsize=8
        )
        GraphSaver.save_graph("EvaluationResults")

    def _prepare_annotated_image(self, tree_id: tuple, view_id: tuple,
                                 annotation: dict, do_text: bool, bar_side: str,
                                 transparent: bool, show: bool) -> (Image, str):
        """ Prepare annotated image from given specification. """

        views = annotation["views"]
        if "base" in views:
            view_data = views["base"]
        elif len(views) > 0:
            view_data = views[list(views.keys())[0]]
        else:
            return None, None

        full_path = view_data.description or None
        if full_path is not None and pathlib.Path(full_path).exists():
            view_image = Image.open(full_path)
            name = pathlib.Path(full_path).with_suffix(".png").name
        else:
            full_path = None
            view_image = Image.fromarray(view_data.data)
            name = f"tree{tree_id[0]}_{tree_id[1]}_view_{view_id[0]}_{view_id[1]}.png"

        data = [ ]
        colors = [ ]
        if "score_gt" in annotation:
            data.append({
                "name": "t",
                "value": annotation["score_gt"]
            })
            colors.append("#e03a3a")
        if "score_feature" in annotation:
            data.append({
                "name": "f",
                "value": annotation["score_feature"]
            })
            colors.append("#88d962")
        if "score_image" in annotation:
            data.append({
                "name": "i",
                "value": annotation["score_image"]
            })
            colors.append("#127331")

        if len(data) > 0:
            data = pd.DataFrame(data=data)
            colors = sns.set_palette(sns.color_palette(colors))

            fig, ax = plt.subplots(figsize=(2, 16))
            plt.ylim(-0.01, 6.0)

            g = sns.barplot(ax=ax, x="name", y="value", data=data, orient="v", color=colors)
            ax.tick_params(axis="both", which="both", length=0)
            g.set(xticklabels=[ ])
            g.set(xlabel=None)
            g.set(yticklabels=[ ])
            g.set(ylabel=None)
            max_value = np.max([ row[ "value" ] for index, row in data.iterrows() ])
            label_height = max_value + 0.01
            if do_text:
                for index, row in data.iterrows():
                    g.text(row.name, label_height, round(row[ "value" ], 2), color='black', ha="center", fontsize=30)

            # Blue, red, green
            bar_data = io.BytesIO()
            fig.savefig(bar_data, format="png", transparent=transparent)
            plt.close("all")

            bar_image = Image.open(bar_data)
            view_image = view_image.convert("RGBA")

            bar_image_size = bar_image.size
            tree_image_size = view_image.size
            size_mult = tree_image_size[1] / bar_image_size[1]
            new_bar_image_size = (int(bar_image_size[0] * size_mult), int(bar_image_size[1] * size_mult))

            annotated_image = Image.new("RGB", tree_image_size, "WHITE")
            bar_image = bar_image.resize(size=new_bar_image_size)

            if bar_side == "r":
                annotated_image.paste(view_image, (0, 0), view_image)
                annotated_image.paste(bar_image, (view_image.size[0] - bar_image.size[0], 0), bar_image)
            else:
                annotated_image.paste(view_image, (0, 0), view_image)
                annotated_image.paste(bar_image, (0, 0), bar_image)

            if show:
                plt.imshow(np.asarray(annotated_image))
                plt.axis("off")
                plt.show()
        else:
            annotated_image = view_image

        return annotated_image, name

    def _export_images(self, evaluations: pd.DataFrame,
                       output_path: Optional[str],
                       do_text: bool, bar_side: str,
                       transparent: bool, show: bool):
        """ Export images with score bars if requested. """

        if output_path is None:
            return

        output_path = pathlib.Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        annotations = { }
        for idx, evaluation in evaluations.iterrows():
            if evaluation.model_name.lower().startswith("image"):
                model_type = "image"
            elif evaluation.model_name.lower().startswith("feature"):
                model_type = "feature"
            else:
                continue

            new_data = { }

            new_data[f"score_{model_type}"] = evaluation.score_p
            if evaluation.true_gt:
                new_data["score_gt"] = evaluation.score_gt
            new_data["views"] = evaluation.view_files

            update_dict_recursively(
                annotations, {
                    ( evaluation.tree_id, evaluation.view_id ): new_data
                },
                create_keys=True
            )

        for ( tree_id, view_id ), annotation in annotations.items():
            if len(annotation["views"]) == 0:
                continue

            annotated_image, name = self._prepare_annotated_image(
                tree_id=tree_id, view_id=view_id,
                annotation=annotation,
                do_text=do_text, bar_side=bar_side,
                transparent=transparent,
                show=show,
            )
            if annotated_image is None:
                continue

            annotated_path = output_path / name
            annotated_image.save(annotated_path, "PNG")


    def _evaluate_scores(self, predictions: Dict[str, Tuple[BaseModel, List[Prediction]]]) -> pd.DataFrame:
        """ Perform evaluation of score precision on given predictions. """

        evaluations = [ ]

        for model_idx, (model_name, (model, model_predictions)) in enumerate(predictions.items()):
            self.__l.info(f"Evaluating mode \"{model_name}\" ({model_idx + 1} / {len(predictions)}")
            for prediction_idx, prediction in enumerate(model_predictions):
                self.__l.info(f"\tEvaluation {prediction_idx + 1}/{len(model_predictions)}")

                prediction_evaluations = self._evaluate_prediction_score(prediction=prediction)

                prediction_evaluations = [
                    update_dict_recursively(evaluation, {
                        "model_idx": model_idx,
                        "model_name": model_name,
                        "eval_idx": idx,
                    }, create_keys=True)
                    for idx, evaluation in enumerate(prediction_evaluations)
                ]

                evaluations += prediction_evaluations

                self.__l.info(f"\t\tEvaluation {prediction_idx + 1}/{len(model_predictions)} Completed!")

        evaluations = pd.DataFrame(data=evaluations)
        self._process_evaluations(evaluations=evaluations)
        self._export_images(
            evaluations=evaluations,
            output_path=self.c.export_images,
            do_text=self.c.export_images_text,
            bar_side=self.c.export_images_side,
            transparent=self.c.export_images_transparent,
            show=self.c.export_images_show,
        )

        return evaluations


    def process(self, predictions: Dict[str, Tuple[BaseModel, List[Prediction]]]):
        """ Perform evaluation processing operations. """

        if self.c.evaluate_current and len(predictions) > 0:
            self._evaluate_scores(predictions)

