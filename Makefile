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
