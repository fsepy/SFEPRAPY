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

__version__ = "0.7"


def check_pip_upgrade():

    # Parse the latest version string
    import subprocess
    from subprocess import STDOUT, check_output

    try:
        output = check_output("pip search --version sfeprapy", stderr=STDOUT, timeout=5)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
        return

    # extract the version number string
    import re

    v = re.findall(r"sfeprapy[\s]*\([\d.]+\)", str(output))[0]
    v = re.findall(r"[\d.]+", str(v))[0]

    # check if upgrade required
    from packaging import version

    is_new_version_available = version.parse(v) > version.parse(__version__)

    # raise message if upgrade is needed
    if is_new_version_available:
        print(
            "New SfePrapy version is available, use `pip install sfeprapy --upgrade` to install the latest version."
        )
        print(f"Current: {__version__}\nLatest: {v}\n\n")


if __name__ == "__main__":
    import re

    def is_canonical(version):
        return (
            re.match(
                r"^([1-9][0-9]*!)?(0|[1-9][0-9]*)(\.(0|[1-9][0-9]*))*((a|b|rc)(0|[1-9][0-9]*))?(\.post(0|[1-9][0-9]*))?(\.dev(0|[1-9][0-9]*))?$",
                version,
            )
            is not None
        )

    assert is_canonical(__version__)
