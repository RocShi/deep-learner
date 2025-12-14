#!/bin/bash

set -e

project_dir=$(
    cd "$(dirname "${BASH_SOURCE[0]}")"
    pwd
)
source $project_dir/build.sh

function run() {
    cd $project_dir/bin
    ./print_index
}

run
