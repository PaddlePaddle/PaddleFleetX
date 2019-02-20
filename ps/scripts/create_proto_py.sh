#!/bin/env sh

set -ex

./third_party/bin/protoc --proto_path=./proto proto/ps.proto --python_out=proto
