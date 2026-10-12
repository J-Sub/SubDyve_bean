#!/usr/bin/env bash
# TOXIC 전체 파이프라인 실행:  ./run_TOXIC.sh [--gpu ID] [--top N] [--stages bean,ssm,subdyve,analysis] [--clean]
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_pipeline.sh" TOXIC "$@"
