#!/bin/bash

urle () { 
    [[ "${1}" ]] || return 1
    local LANG=C i x
    for (( i = 0; i < ${#1}; i++ )); do 
        x="${1:i:1}"
        [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"
    done
    echo
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR_DEFAULT="${SCRIPT_DIR}/data"
DATA_DIR="${TOKENHMR_DATA_DIR:-${DATA_DIR_DEFAULT}}"
DOWNLOAD_DIR="${TOKENHMR_DOWNLOAD_DIR:-${DATA_DIR}/.downloads}"

ensure_data_dir() {
    if [[ "${DATA_DIR}" != "${DATA_DIR_DEFAULT}" ]]; then
        mkdir -p "${DATA_DIR}"
        if [[ -e "${DATA_DIR_DEFAULT}" && ! -L "${DATA_DIR_DEFAULT}" ]]; then
            echo "Error: ${DATA_DIR_DEFAULT} exists and is not a symlink. Please move or remove it first." >&2
            return 1
        fi
        if [[ ! -e "${DATA_DIR_DEFAULT}" ]]; then
            ln -s "${DATA_DIR}" "${DATA_DIR_DEFAULT}"
        fi
    else
        mkdir -p "${DATA_DIR_DEFAULT}"
    fi

    mkdir -p "${DOWNLOAD_DIR}"
}

check_disk_space() {
    local required_kb=6291456
    local check_path="${DATA_DIR}"
    local avail_kb
    avail_kb="$(df -Pk "${check_path}" | awk 'NR==2 {print $4}')"
    if [[ -n "${avail_kb}" && "${avail_kb}" -lt "${required_kb}" ]]; then
        echo "Error: not enough disk space on ${check_path} filesystem." >&2
        echo "Please free space or set TOKENHMR_DATA_DIR to a path on a larger mount, e.g. /root/tokenhmr_data." >&2
        return 1
    fi
}

# Function to download, unzip, and remove the zip file
download_and_unzip() {
    local url=$1
    local output_file_name
    local output_file
    output_file_name="$(basename "$url" | sed 's/.*sfile=//')"
    output_file="${DOWNLOAD_DIR}/${output_file_name}"

    if ! wget --post-data "username=${username}&password=${password}" "$url" -O "$output_file" --no-check-certificate --continue; then
        echo "Error: download failed for ${output_file_name}." >&2
        return 1
    fi

    if ! unzip -q "$output_file" -d "${SCRIPT_DIR}"; then
        echo "Error: unzip failed for ${output_file_name}." >&2
        return 1
    fi

    rm -f "$output_file"
}

# Prompt for credentials
echo -e "\nYou need to register at https://tokenhmr.is.tue.mpg.de"
read -r -p "Username:" username
read -r -sp "Password:" password
echo

username="$(urle "$username")"
password="$(urle "$password")"

if ! ensure_data_dir; then
    exit 1
fi

if ! check_disk_space; then
    exit 1
fi

# Download and unzip the first two zip files by default
download_and_unzip 'https://download.is.tue.mpg.de/download.php?domain=tokenhmr&sfile=data.zip'
download_and_unzip 'https://download.is.tue.mpg.de/download.php?domain=tokenhmr&sfile=tokenhmr_model_latest.zip'
