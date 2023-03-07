import logging
import os


# setup logger
def get_logger(f_handler_fp: str = None, f_handler_level=logging.WARNING, c_handler_level=logging.INFO):
    logger_ = logging.getLogger('sfeprapy')

    c_handler = logging.StreamHandler()
    c_handler.setLevel(c_handler_level)
    c_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'))
    logger_.addHandler(c_handler)

    logger_.setLevel(logging.DEBUG)

    return logger_


logger = get_logger()

# make root directory of this app which will be used 1. when running the app; 2. pyinstaller at compiling the app.
if os.path.exists(os.path.dirname(__file__)):
    # this path should be used when running the app as a Python package (non compiled) and/or pyinstaller at compiling
    # stage.
    __root_dir__ = os.path.realpath(os.path.dirname(__file__))
elif os.path.exists(os.path.dirname(os.path.dirname(__file__))):
    # the path will become invalid when the app run after compiled as the dirname `fsetoolsGUI` will disappear.
    # instead, the parent folder of the project dir will be used.
    __root_dir__ = os.path.realpath(os.path.dirname(os.path.dirname(__file__)))
else:
    raise IsADirectoryError(
        f'Project root directory undefined: '
        f'{os.path.dirname(__file__)} nor '
        f'{os.path.dirname(os.path.dirname(__file__))}'
    )

"""
VERSION IDENTIFICATION RULES DOCUMENTED IN PEP 440 ARE FOLLOWED.

Version scheme
==============

Distributions are identified by a public version identifier which supports all defined version comparison operations

The version scheme is used both to describe the distribution version provided by a particular distribution archive, as
well as to place constraints on the version of dependencies needed in order to build or run the software.

Public version identifiers
--------------------------

The canonical public version identifiers MUST comply with the following scheme:

`[N!]N(.N)*[{a|b|rc}N][.postN][.devN]`

Public version identifiers MUST NOT include leading or trailing whitespace.

Public version identifiers MUST be unique within a given distribution.

See also Appendix B : Parsing version strings with regular expressions which provides a regular expression to check
strict conformance with the canonical format, as well as a more permissive regular expression accepting inputs that may
require subsequent normalization.

Public version identifiers are separated into up to five segments:

    - Epoch segment: N!
    - Release segment: N(.N)*
    - Pre-release segment: {a|b|rc}N
    - Post-release segment: .postN
    - Development release segment: .devN

"""

__version__ = "0.8.1"


def _test_version_canonical(version=__version__):
    import re
    check = re.match(
        r"^([1-9][0-9]*!)?"
        r"(0|[1-9][0-9]*)"
        r"(\.(0|[1-9][0-9]*))*"
        r"((a|b|rc)(0|[1-9][0-9]*))?"
        r"(\.post(0|[1-9][0-9]*))?"
        r"(\.dev(0|[1-9][0-9]*))?$",
        version,
    )
    assert check is not None
