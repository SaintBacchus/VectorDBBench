from typing import Annotated, TypedDict, Unpack

import click
from pydantic import SecretStr

from .config import LanceHNSWConfig


from ....cli.cli import (
    CommonTypedDict,
    HNSWFlavor2,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)
from .. import DB


class LanceTypedDict(TypedDict):
    url: Annotated[
        str, click.option("--url", type=str, help="url path", required=True)
    ]


class LanceHNSWTypedDict(CommonTypedDict, LanceTypedDict, HNSWFlavor2):
    ...


@cli.command()
@click_parameter_decorators_from_typed_dict(LanceHNSWTypedDict)
def Lance(**parameters: Unpack[LanceHNSWTypedDict]):
    from .config import LanceConfig
    run(
        db=DB.Lance,
        db_config=LanceConfig(
            url=SecretStr(parameters["url"]),
        ),
        db_case_config=LanceHNSWConfig(
            M=parameters["m"],
            efConstruction=parameters["ef_construction"],
            ef=parameters["ef_runtime"],
        ),
        **parameters,
    )
