# -*- coding: utf-8 -*-

"""
Growth file utilities.
"""

import io
import pathlib
import xml.dom.minidom
from xml.etree import ElementTree
from typing import Dict, List, Optional, Tuple, Union

from perceptree.common.cache import Cache


class XmlAttributes(dict): pass


def xml_try_parse_text(text: str) -> any:
    """ Attempt to parse the text as numeric values, falling back to text. """

    try: return int(text)
    except: pass

    try: return float(text)
    except: pass

    return text

def xml_to_dict(root: ElementTree.Element, process_attributes: bool = True) -> dict:
    """ Recursively convert XML document starting with root into a dictionary. """

    def recursive_parser(parent_element: ElementTree.Element):
        result = dict({ })

        for element in parent_element:
            obj = recursive_parser(element) if len(element) else xml_try_parse_text(element.text)
            val = ( XmlAttributes(element.attrib), obj ) if process_attributes else obj

            if element.tag in result:
                if hasattr(result[element.tag], "append"):
                    result[element.tag].append(val)
                else:
                    result[element.tag] = [ result[element.tag], val ]
            else:
                result[element.tag] = val

        return result

    return { root.tag: recursive_parser(root) }


def dict_to_xml(data: dict) -> Union[ElementTree.Element, List[ElementTree.Element]]:
    """ Recursively convert dictionary into an XML document. """

    def recursive_constructor(document: ElementTree.Element, key: str, val: any) -> ElementTree.Element:
        if isinstance(val, list):
            for item in val:
                recursive_constructor(document, key, item)
            node = document
        elif isinstance(val, tuple) and len(val) == 2 and \
                isinstance(val[0], XmlAttributes):
            node = recursive_constructor(document, key, val[1])
            node.attrib = dict(val[ 0 ])
        elif isinstance(val, dict):
            node = ElementTree.Element(key)
            for name, item in val.items():
                recursive_constructor(node, name, item)
            document.append(node)
        elif val is None:
            node = ElementTree.Element(key)
            document.append(node)
        else:
            node = ElementTree.Element(key)
            node.text = str(val)
            document.append(node)

        return node

    results = [ ]
    for root_key, root_value in data.items():
        root_node = ElementTree.Element("root")
        recursive_constructor(root_node, root_key, root_value)
        results.append(root_node[0])

    return results[0] if len(results) == 1 else results


class GrowthParameters(object):
    """
    Wrapper around growth parameter files.

    :param file_path: Provide a path or buffer to load the parameters from in XML format.
    :param parameters: Provide a dictionary containing parameter values.
    """

    PARAMETER_FORMATS = {
        "XML": {
            "parameter_path": "TreeSettings.Instance.TreeParameters",
            "default_content": """
<TreeSettings amount="1">
    <Instance index="0">
        <Position>
            <x>0.000000</x>
            <y>0.000000</y>
            <z>0.000000</z>
        </Position>
        <TreeParameters>
            <Seed>1</Seed>
            <Age>10</Age>
            <LateralBudPerNode>3</LateralBudPerNode>
            <VarianceApicalAngle>20.000000</VarianceApicalAngle>
            <BranchingAngleMean>29.520000</BranchingAngleMean>
            <BranchingAngleVariance>2.000000</BranchingAngleVariance>
            <RollAngleMean>91.000000</RollAngleMean>
            <RollAngleVariance>1.000000</RollAngleVariance>
            <ApicalBudKillProbability>0.000000</ApicalBudKillProbability>
            <LateralBudKillProbability>0.210000</LateralBudKillProbability>
            <ApicalDominanceBase>3.130000</ApicalDominanceBase>
            <ApicalDominanceDistanceFactor>0.130000</ApicalDominanceDistanceFactor>
            <ApicalDominanceAgeFactor>0.820000</ApicalDominanceAgeFactor>
            <GrowthRate>3.000000</GrowthRate>
            <InternodeLengthBase>1.000000</InternodeLengthBase>
            <InternodeLengthAgeFactor>0.930000</InternodeLengthAgeFactor>
            <ApicalControlBase>2.200000</ApicalControlBase>
            <ApicalControlAgeFactor>0.500000</ApicalControlAgeFactor>
            <ApicalControlLevelFactor>1.000000</ApicalControlLevelFactor>
            <ApicalControlDistanceFactor>0.000000</ApicalControlDistanceFactor>
            <MaxBudAge>20</MaxBudAge>
            <InternodeSize>0.500000</InternodeSize>
            <Phototropism>0.320000</Phototropism>
            <GravitropismBase>0.080000</GravitropismBase>
            <GravitropismLevelFactor>-0.030000</GravitropismLevelFactor>
            <PruningFactor>0.700000</PruningFactor>
            <LowBranchPruningFactor>1.300000</LowBranchPruningFactor>
            <ThicknessRemovalFactor>99999.000000</ThicknessRemovalFactor>
            <GravityBendingStrength>0.720000</GravityBendingStrength>
            <GravityBendingAngleFactor>0.830000</GravityBendingAngleFactor>
            <ApicalBudLightingFactor>0.390000</ApicalBudLightingFactor>
            <LateralBudLightingFactor>1.000000</LateralBudLightingFactor>
            <EndNodeThickness>0.010000</EndNodeThickness>
            <ThicknessControlFactor>0.650000</ThicknessControlFactor>
            <CrownShynessBase>1.000000</CrownShynessBase>
            <CrownShynessFactor>1.000000</CrownShynessFactor>
            <FoliageType>6</FoliageType>
        </TreeParameters>
    </Instance>
</TreeSettings>
            """
        }
    }
    """ Information about each of the supported parameter formats. """

    def __init__(self, file_path: Optional[Union[str, io.IOBase]] = None,
                 parameters: Optional[dict] = None, format: str = "XML"):

        self._name = "Model"
        self._data = Cache()
        self._format = format
        self.initialize_default()

        if file_path:
            self.load_from(file_path=file_path)
        if parameters:
            self.load_parameters(parameters)

    def initialize_default(self):
        """ Load default parameters file. """

        default_xml = io.StringIO(GrowthParameters.PARAMETER_FORMATS[self._format]["default_content"])
        return self.load_from(file_path=default_xml)

    def load_from(self, file_path: Union[str, io.IOBase]):
        """ Load growth parameters from given path. """

        self._name = pathlib.Path(file_path).with_suffix("").name if isinstance(file_path, str) else "model"
        self._data = Cache(xml_to_dict(ElementTree.parse(file_path).getroot(), process_attributes=False))

    def load_parameters(self, parameters: dict):
        """ Replace current parameter set with provided parameters. """

        self._data[GrowthParameters.PARAMETER_FORMATS[self._format]["parameter_path"]] = parameters

    def save_as(self, path: str):
        """ Save the current growth parameters into given path. """

        with open(path, "w") as f:
            xml.dom.minidom.parseString(
                ElementTree.tostring(dict_to_xml(self._data), encoding="unicode", method="xml")
            ).writexml(f, "", "\t", "\n", encoding="unicode")

    @property
    def format(self) -> str:
        """ Get format currently used for this parameter set. """

        return self._format

    @property
    def name(self) -> str:
        """ Get name of this model, deduced from filename when loaded. """

        return self._name

    @property
    def data(self) -> dict:
        """ Get data dictionary containing all of the parameters. """

        return dict(self._data.cache)

    @property
    def parameters(self) -> { }:
        """ Get list of all parameters within this parameters wrapper. """

        return self._data.get_path(GrowthParameters.PARAMETER_FORMATS[self._format]["parameter_path"], default={ })

    @property
    def parameter_names(self) -> List[str]:
        """ Get list of all parameter types within this parameters wrapper. """

        return list(self.parameters.keys())

