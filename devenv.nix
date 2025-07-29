# Enabling cuda support in devenv.nix: https://github.com/clementpoiret/nix-python-devenv/blob/cuda/devenv.nix
{ pkgs, lib, config, inputs, ... }:

let
  system = pkgs.stdenv.system;
  git-hooks = inputs.git-hooks.packages.${system}.git-hooks;
  buildInputs = with pkgs; [
    # Cuda 12.4
    cudaPackages.cuda_cudart
    cudaPackages.cudatoolkit
    cudaPackages.cudnn
    cudaPackages.cuda_nvcc
    stdenv.cc.cc
    libuv
    libGL
    zlib
  ];
in {
  name = "pgi-pipeline";

  cachix.enable = true;

  # Languages to include in the environment
  languages = {
    python = {
      enable = true;
      version = "3.7";
      uv = {
        enable = true;
        sync.enable = false;
      };
    };
  };

  # Define global environment variables. Private environment variables should reside in '.envrc.private'
  env = {
    UV_LINK_MODE = "copy";
    UV_PYTHON_PREFERENCE = "only-system";
    UV_PYTHON_DOWNLOADS = "never";

    COMPOSE_BAKE = "true";

    LD_LIBRARY_PATH = "${
        with pkgs;
        lib.makeLibraryPath buildInputs
      }:${pkgs.libGL}/lib/:${pkgs.stdenv.cc.cc.lib}/lib/:/run/opengl-driver/lib:/run/opengl-driver-32/lib";
    XLA_FLAGS =
      "--xla_gpu_cuda_data_dir=${pkgs.cudaPackages.cudatoolkit}"; # For tensorflow with GPU support
    CUDA_PATH = pkgs.cudaPackages.cudatoolkit;
  };

  # https://devenv.sh/packages/
  packages = with pkgs;
    [
      bashInteractive
      just
      uv
      ffmpeg_7
      libGL
      glibc
      util-linux
      xclip
      gcc.cc
      fzf
      jq
      curl
      zip
      git
      cudaPackages.cuda_nvcc
    ]
    # Packages to be only in native devenv shell and not containerized
    # LINK: https://devenv.sh/containers/#changing-the-environment-based-on-the-build-type
    ++ lib.optionals (!config.container.isBuilding) [
      awscli2
      mpv
      dive
      gdb
      gdbgui
      docker
      docker-buildx
      nvidia-docker
      circleci-cli
      pre-commit
    ];

  # Commands which run when the shell is started
  enterShell = ''
    export UV_PROJECT_ENVIRONMENT=$(pwd)/.venv
    export IO_DIR=$(pwd)/IO
    nvcc -V

    # Set the SSH agent if not already set
    if [ -z "$SSH_AUTH_SOCK" ] ; then
      eval `ssh-agent -s`
      ssh-add $PRIVATE_SSH_PATH
    fi

    uv tool install bump-my-version
    uv tool update-shell

    # Python environment setup
    just ready-py

    # Own the local directory
    just own

    # Run the fish shell instead of bash
    fish --init-command="source .venv/bin/activate.fish"

    # When the command 'exit' is run to exit the fish shell, then the bash shell is run, so exit that
    exit
  '';

  # Git pre-commit hooks, defined here. LINK: https://github.com/cachix/git-hooks.nix/tree/master
  git-hooks = {
    excludes = [ ".xml" ".dll" ".exe" ".pdb" ".flake.nix" ];
    #enabledPackages = [ pkgs.python312Packages.ruff ];
    hooks = {
      # Lint and format YAML files
      yamllint = {
        enable = false;
        #excludes = [ "*/.circleci/config.yml" ];
        settings = { preset = "relaxed"; };
      };
      yamlfmt.enable = false;

      # Lint and format shell scripts
      shellcheck.enable = false;
      shfmt.enable = false;

      # Lint and format python using ruff
      ruff = {
        enable = true;
        excludes = [ ];
      };
      ruff-format = {
        enable = true;
        excludes = [ ];
      };

      # Lint and format for nix files
      nixfmt-classic.enable = true;
      #   statix.enable = true;
      #   statix.settings.ignore = [ ".devenv*" ];

      # No-commit-to-branch
      no-commit-to-branch = {
        enable = true;
        settings.pattern = [
          "v[0-9]+.[0-9]+.[0-9]+" # Matches vX.Y.Z, where X, Y, Z are numbers
          "ma.*" # Matches "main", "master", etc
        ];
      };

      # Spell-checking hook
      typos = {
        excludes = [ "terraform/" "src/core/networks/" "devenv.nix" ".ipynb" ];
        enable = true;
        verbose = true;
        settings = {
          ignored-words = [ "datas" "Thr" "gather_and" ];
          write = false;
          verbose = true;
        };
      };

      # Prevent secrets from being committed
      ripsecrets.enable = true;
      # ignore devenv.nix file
      ripsecrets.excludes = [ "devenv.nix" ];
    };
  };

  # See full reference at https://devenv.sh/reference/options/
}
