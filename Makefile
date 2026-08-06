HIPCC   := hipcc
FLAGS   := -O3 -Iinclude --offload-arch=gfx1101
KERNELS := kernels/kernels.hip

.PHONY: bench test resources isa clean

# No argument benchmarks every variant; `make bench VARIANT=naive` picks one,
# and several names may be passed: `make bench VARIANT="naive block"`.
bench: bin/bench
	@./bin/bench $(VARIANT)

test: bin/test
	@./bin/test

bin/bench: src/bench.cpp $(KERNELS) include/variants.hpp include/kernels.hpp
	@mkdir -p bin
	$(HIPCC) $(FLAGS) src/bench.cpp $(KERNELS) -o $@

bin/test: test/kernel_test.cpp $(KERNELS) include/variants.hpp include/kernels.hpp
	@mkdir -p bin
	$(HIPCC) $(FLAGS) test/kernel_test.cpp $(KERNELS) -o $@

# LDS, VGPR, spills and occupancy, straight from the compiler. Needs the kernel
# source, not a compiled binary.
resources:
	@$(HIPCC) $(FLAGS) -Rpass-analysis=kernel-resource-usage -c $(KERNELS) -o /dev/null 2>&1 \
		| c++filt | sed 's/ \[-Rpass.*//' | grep 'remark:' | sed 's/^.*remark: *//'

isa:
	@mkdir -p profiling
	@$(HIPCC) $(FLAGS) --offload-device-only -S $(KERNELS) -o - 2>/dev/null | c++filt > profiling/kernels.s
	@echo "-> profiling/kernels.s"

clean:
	rm -rf bin
