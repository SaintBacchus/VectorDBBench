from typing import Annotated, TypedDict, Unpack

import click
from pydantic import SecretStr

from .config import LancedbHNSWConfig


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


class LancedbHNSWTypedDict(CommonTypedDict, LanceTypedDict, HNSWFlavor2):
    ...


@cli.command()
@click_parameter_decorators_from_typed_dict(LancedbHNSWTypedDict)
def Lancedb(**parameters: Unpack[LancedbHNSWTypedDict]):
    from .config import LancedbConfig
    run(
        db=DB.Lancedb,
        db_config=LancedbConfig(
            url=SecretStr(parameters["url"]),
        ),
        db_case_config=LancedbHNSWConfig(
            M=parameters["m"],
            efConstruction=parameters["ef_construction"],
            ef=parameters["ef_runtime"],
        ),
        **parameters,
    )
