set export := true
set positional-arguments := true
set unstable
set script-interpreter := ['uv', 'run', '--script']
# set shell := ['bash', '-e']

# Define variables and functions, these are set as environment variables during each invocation of the justfile
user := file_stem(home_directory())
curr_time := datetime("%Y%m%dT%H%M%S")
NIXPKGS_ALLOW_UNFREE := "1"

# Open the devenv shell
shell:
    {{ if env("DEVENV_RUNTIME", "") == "" { "sudo -E devenv shell --impure --verbose" } else { "echo 'Already in devenv shell!'" } }}

# Own the current directory as the logged in user
own:
    sudo chown -R {{ user }} .

# Bump the version of the package
bump type="pre_n" dry="dry" force="":
    just gadd
    uv run bump-version.py bump {{ type }} {{ if dry != "foreal" { "--dry-run" } else { "" } }} {{ if force == "force" { "--force" } else { "" } }}

show-bump from-version="":
    uv tool run bump-my-version show-bump {{ from-version }}

get-version:
    uv tool run bump-my-version show current_version

# Initialize the python project environment   
ready-py:
    uv lock
    uv sync --locked
    mkdir -p tmp
    uv export --no-dev --no-emit-project --no-hashes --no-annotate --no-header --output-file ./tmp/prod_requirements.txt
    unset VIRTUAL_ENV

# Get the versions of different packages in the python uv environment
[script]
get-package-versions:
    import tensorflow as tf
    import keras
    import numpy

    print(f"Tensorflow version: {tf.__version__}")
    print(f"Keras version: {keras.__version__}")
    print(f"NumPy version: {numpy.__version__}")

run image-filepath="./image/image/t.png":
    uv run 

# Git add all files
gadd:
    git add .

# Garbage collect
gc:
    sudo devenv gc
    sudo nix-collect-garbage -d

# Run pre-commit hooks
pre-commit args="": gadd
    pre-commit run {{ args }}

# Run ruff checks on all of the git staged files
check: gadd
    ruff check --unsafe-fixes $(git --no-pager diff --cached --name-only --diff-filter=d '*.py')

# Run ruff formatting on all of the git staged files
format dry="": gadd
    ruff format {{ if dry == "foreal" { "" } else { "--check" } }} $(git --no-pager diff --cached --name-only --diff-filter=d '*.py')