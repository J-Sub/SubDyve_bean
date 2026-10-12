#!/usr/bin/env bash
# SIDER 전체 파이프라인 실행:  ./run_SIDER.sh [--gpu ID] [--top N] [--stages bean,ssm,subdyve,analysis] [--clean]
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_pipeline.sh" SIDER "$@"
