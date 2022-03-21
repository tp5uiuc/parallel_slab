"""Nox sessions."""
import tempfile
from typing import Any

import nox
from nox.sessions import Session
from pathlib import PurePath


example_files = "run_parameter_sweep.py", "run_single.py"

locations = (
    "parallel_slab",
    "tests",
    "noxfile.py",
    *map(lambda x: str(PurePath("examples").joinpath(x)), example_files),
)


def install_with_constraints(session: Session, *args: str, **kwargs: Any) -> None:
    """Install packages constrained by Poetry's lock file.

    This function is a wrapper for nox.sessions.Session.install. It
    invokes pip to install packages inside of the session's virtualenv.
    Additionally, pip is passed a constraints file generated from
    Poetry's lock file, to ensure that the packages are pinned to the
    versions specified in poetry.lock. This allows you to manage the
    packages as Poetry development dependencies.

    :param session: The Session object.
    :param args: Command-line arguments for pip.
    :param kwargs: Additional keyword arguments for Session.install.
    """
    with tempfile.NamedTemporaryFile() as requirements:
        session.run(
            "poetry",
            "export",
            "--dev",
            "--format=requirements.txt",
            # Workaround as in https://github.com/cjolowicz/hypermodern-python/issues/174
            "--without-hashes",
            f"--output={requirements.name}",
            external=True,
        )
        session.install(f"--constraint={requirements.name}", *args, **kwargs)


@nox.session(python=["3.9", "3.10"])
def tests(session: Session) -> None:
    """Run the test suite."""
    args = session.posargs or ["--cov"]
    session.run("poetry", "install", "--no-dev", external=True)
    install_with_constraints(
        session, "coverage[toml]", "pytest", "pytest-cov", "pytest-mpl"
    )
    session.run("pytest", *args)


@nox.session(python=["3.9", "3.10"])
def lint(session) -> None:
    """Lint using flake8."""
    args = session.posargs or locations
    install_with_constraints(
        session,
        "flake8",
        "flake8-black",
        # "flake8-bugbear",
        # "flake8-import-order",
        # "flake8-bandit",
        # "flake8-comprehensions",
        # "flake8-docstrings",
        # "flake8-annotations-complexity",
        # "pep8-naming",
    )
    session.run("flake8", *args)


@nox.session(python="3.10")
def black(session: Session) -> None:
    """Run black code formatter."""
    args = session.posargs or locations
    install_with_constraints(session, "black")
    session.run("black", *args)
