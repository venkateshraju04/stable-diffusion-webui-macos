#!/usr/bin/env bash
#################################################
# Stable Diffusion WebUI — macOS Launch Script  #
# Change your variables in webui-user.sh        #
#################################################

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Load macOS defaults
if [[ -f "$SCRIPT_DIR"/webui-macos-env.sh ]]; then
    source "$SCRIPT_DIR"/webui-macos-env.sh
fi

# Read user overrides
# shellcheck source=/dev/null
if [[ -f "$SCRIPT_DIR"/webui-user.sh ]]; then
    source "$SCRIPT_DIR"/webui-user.sh
fi

# If $venv_dir is "-", then disable venv support
use_venv=1
if [[ $venv_dir == "-" ]]; then
  use_venv=0
fi

# Set defaults
if [[ -z "${install_dir}" ]]; then
    install_dir="$SCRIPT_DIR"
fi

if [[ -z "${clone_dir}" ]]; then
    clone_dir="stable-diffusion-webui"
fi

# python3 executable — prefer python3.11
if [[ -z "${python_cmd}" ]]; then
  python_cmd="python3.11"
fi
if [[ ! -x "$(command -v "${python_cmd}")" ]]; then
  python_cmd="python3"
fi

# git executable
if [[ -z "${GIT}" ]]; then
    export GIT="git"
else
    export GIT_PYTHON_GIT_EXECUTABLE="${GIT}"
fi

# python3 venv (defaults to venv/)
if [[ -z "${venv_dir}" ]] && [[ $use_venv -eq 1 ]]; then
    venv_dir="venv"
fi

if [[ -z "${LAUNCH_SCRIPT}" ]]; then
    LAUNCH_SCRIPT="launch.py"
fi

# this script cannot be run as root by default
can_run_as_root=0

while getopts "f" flag > /dev/null 2>&1
do
    case ${flag} in
        f) can_run_as_root=1;;
        *) break;;
    esac
done

# Disable sentry logging
export ERROR_REPORTING=FALSE

# Do not reinstall existing pip packages
export PIP_IGNORE_INSTALLED=0

# Pretty print
delimiter="################################################################"

printf "\n%s\n" "${delimiter}"
printf "\e[1m\e[32mStable Diffusion WebUI — macOS (Apple Silicon)\n"
printf "\e[1m\e[34mOptimized for M-series Macs with MPS acceleration\e[0m"
printf "\n%s\n" "${delimiter}"

# Do not run as root
if [[ $(id -u) -eq 0 && can_run_as_root -eq 0 ]]; then
    printf "\n%s\n" "${delimiter}"
    printf "\e[1m\e[31mERROR: This script must not be launched as root, aborting...\e[0m"
    printf "\n%s\n" "${delimiter}"
    exit 1
else
    printf "\n%s\n" "${delimiter}"
    printf "Running on \e[1m\e[32m%s\e[0m user" "$(whoami)"
    printf "\n%s\n" "${delimiter}"
fi

if [[ -d "$SCRIPT_DIR/.git" ]]; then
    printf "\n%s\n" "${delimiter}"
    printf "Repo already cloned, using it as install directory"
    printf "\n%s\n" "${delimiter}"
    install_dir="${SCRIPT_DIR}/../"
    clone_dir="${SCRIPT_DIR##*/}"
fi

# Check prerequisites
for preq in "${GIT}" "${python_cmd}"
do
    if ! hash "${preq}" &>/dev/null
    then
        printf "\n%s\n" "${delimiter}"
        printf "\e[1m\e[31mERROR: %s is not installed, aborting...\e[0m" "${preq}"
        printf "\n%s\n" "${delimiter}"
        exit 1
    fi
done

if [[ $use_venv -eq 1 ]] && ! "${python_cmd}" -c "import venv" &>/dev/null
then
    printf "\n%s\n" "${delimiter}"
    printf "\e[1m\e[31mERROR: python3-venv is not installed, aborting...\e[0m"
    printf "\n%s\n" "${delimiter}"
    exit 1
fi

cd "${install_dir}"/ || { printf "\e[1m\e[31mERROR: Can't cd to %s/, aborting...\e[0m" "${install_dir}"; exit 1; }
if [[ -d "${clone_dir}" ]]; then
    cd "${clone_dir}"/ || { printf "\e[1m\e[31mERROR: Can't cd to %s/%s/, aborting...\e[0m" "${install_dir}" "${clone_dir}"; exit 1; }
else
    printf "\n%s\n" "${delimiter}"
    printf "Clone stable-diffusion-webui"
    printf "\n%s\n" "${delimiter}"
    "${GIT}" clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git "${clone_dir}"
    cd "${clone_dir}"/ || { printf "\e[1m\e[31mERROR: Can't cd to %s/%s/, aborting...\e[0m" "${install_dir}" "${clone_dir}"; exit 1; }
fi

if [[ $use_venv -eq 1 ]] && [[ -z "${VIRTUAL_ENV}" ]]; then
    printf "\n%s\n" "${delimiter}"
    printf "Create and activate python venv"
    printf "\n%s\n" "${delimiter}"
    cd "${install_dir}"/"${clone_dir}"/ || { printf "\e[1m\e[31mERROR: Can't cd to %s/%s/, aborting...\e[0m" "${install_dir}" "${clone_dir}"; exit 1; }
    if [[ ! -d "${venv_dir}" ]]; then
        "${python_cmd}" -m venv "${venv_dir}"
        "${venv_dir}"/bin/python -m pip install --upgrade pip
        first_launch=1
    fi
    # shellcheck source=/dev/null
    if [[ -f "${venv_dir}"/bin/activate ]]; then
        source "${venv_dir}"/bin/activate
        python_cmd="${venv_dir}"/bin/python
    else
        printf "\n%s\n" "${delimiter}"
        printf "\e[1m\e[31mERROR: Cannot activate python venv, aborting...\e[0m"
        printf "\n%s\n" "${delimiter}"
        exit 1
    fi
else
    printf "\n%s\n" "${delimiter}"
    printf "python venv already activate or run without venv: ${VIRTUAL_ENV}"
    printf "\n%s\n" "${delimiter}"
fi

# Launch loop (supports restart via tmp/restart file)
KEEP_GOING=1
export SD_WEBUI_RESTART=tmp/restart
while [[ "$KEEP_GOING" -eq "1" ]]; do
    if [[ ! -z "${ACCELERATE}" ]] && [ ${ACCELERATE}="True" ] && [ -x "$(command -v accelerate)" ]; then
        printf "\n%s\n" "${delimiter}"
        printf "Accelerating launch.py..."
        printf "\n%s\n" "${delimiter}"
        accelerate launch --num_cpu_threads_per_process=6 "${LAUNCH_SCRIPT}" "$@"
    else
        printf "\n%s\n" "${delimiter}"
        printf "Launching launch.py..."
        printf "\n%s\n" "${delimiter}"
        "${python_cmd}" -u "${LAUNCH_SCRIPT}" "$@"
    fi

    if [[ ! -f tmp/restart ]]; then
        KEEP_GOING=0
    fi
done
