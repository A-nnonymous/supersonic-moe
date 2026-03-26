# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

install:
	git submodule update --init --recursive
	pip install .

install-dev:
	git submodule update --init --recursive
	pip install -e .

test:
	pytest tests

test-blackwell:
	USE_QUACK_GEMM=1 python -m pytest -q tests/moe_blackwell_test.py

test-blackwell-full:
	USE_QUACK_GEMM=1 python -m pytest -q tests/fp8_protocol_test.py tests/moe_blackwell_test.py tests/moe_test.py

PYTEST_WORKERS ?= 2

test-blackwell-parallel:
	USE_QUACK_GEMM=1 python -m pytest -q -n $(PYTEST_WORKERS) --dist loadscope tests/fp8_protocol_test.py tests/moe_blackwell_test.py tests/moe_test.py

FP8_OPERATOR_OPTS ?=
FP8_LARGE_PROJECT_BENCH_SHAPE ?= 8192,4096,1024,128,8

test-large-project-baseline:
	USE_QUACK_GEMM=1 python -m pytest -q tests/fp8_large_project_contract_test.py

test-large-project-opt:
	@test -n "$(FP8_OPERATOR_OPTS)" || (echo "Usage: make test-large-project-opt FP8_OPERATOR_OPTS='SONIC_MOE_OPT_NATIVE_FP8_UPPROJ=1 ...'" && exit 1)
	USE_QUACK_GEMM=1 env $(FP8_OPERATOR_OPTS) python -m pytest -q tests/fp8_large_project_contract_test.py

bench-large-project-baseline:
	USE_QUACK_GEMM=1 python benchmarks/moe-cute.py --thiek $(FP8_LARGE_PROJECT_BENCH_SHAPE) --skip_test

bench-large-project-opt:
	@test -n "$(FP8_OPERATOR_OPTS)" || (echo "Usage: make bench-large-project-opt FP8_OPERATOR_OPTS='SONIC_MOE_OPT_NATIVE_FP8_UPPROJ=1 ...'" && exit 1)
	USE_QUACK_GEMM=1 env $(FP8_OPERATOR_OPTS) python benchmarks/moe-cute.py --thiek $(FP8_LARGE_PROJECT_BENCH_SHAPE) --skip_test --fp8_protocol blackwell --report_fp8_metrics --report_stage_memory --report_fp8_analysis

BLACKWELL_TEST_GPUS ?= 0,1,2

test-blackwell-multigpu:
	python tools/run_blackwell_test_shards.py --gpus $(BLACKWELL_TEST_GPUS)

test-debug:
	DEBUG_CUTOTUNE=1 TRITON_PRINT_AUTOTUNING=1 pytest -s tests

update-precommit:
	pre-commit autoupdate

style:
	pre-commit run --all-files

cutotune-cache:
	DEBUG_CUTOTUNE=1 LOAD_CUTOTUNE_CACHE=1 TORCH_CUDA_ARCH_LIST=9.0 python tools/build_cutotune_cache.py

warp-serve:
	./tools/warp-control-plane.sh serve

warp-up:
	./tools/warp-control-plane.sh up

warp-stop-agents:
	./tools/warp-control-plane.sh stop-agents

warp-silent:
	./tools/warp-control-plane.sh silent

warp-stop-all:
	./tools/warp-control-plane.sh stop-all

CLUSTER_IDLE_UTIL_MAX ?= 10
CLUSTER_IDLE_MEM_MAX_MIB ?= 5000
CLUSTER_IDLE_GPU_COUNT ?= 1
CLUSTER_SCAN_LIMIT ?=
CLUSTER_COMMAND ?=

cluster-scan:
	python tools/cluster_idle_launch.py scan \
		--util-max $(CLUSTER_IDLE_UTIL_MAX) \
		--mem-max-mib $(CLUSTER_IDLE_MEM_MAX_MIB) \
		$(if $(CLUSTER_SCAN_LIMIT),--limit-hosts $(CLUSTER_SCAN_LIMIT),)

cluster-launch:
	@test -n "$(CLUSTER_COMMAND)" || (echo "Usage: make cluster-launch CLUSTER_COMMAND='...'" && exit 1)
	python tools/cluster_idle_launch.py launch \
		--gpu-count $(CLUSTER_IDLE_GPU_COUNT) \
		--util-max $(CLUSTER_IDLE_UTIL_MAX) \
		--mem-max-mib $(CLUSTER_IDLE_MEM_MAX_MIB) \
		--workdir $(CURDIR) \
		--command "$(CLUSTER_COMMAND)" \
		$(if $(CLUSTER_SCAN_LIMIT),--limit-hosts $(CLUSTER_SCAN_LIMIT),)
