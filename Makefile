.PHONY: build run clean test

build:
	@mkdir -p bin profiling
	hipcc -O3 -Iinclude src/main.cpp kernels/kernels.hip -o bin/out --offload-arch=gfx1101

run:
	./bin/out

clean:
	rm -f bin/out

test:
	@mkdir -p bin profiling
	hipcc -O3 -Iinclude test/kernel_test.cpp kernels/kernels.hip -o bin/test --offload-arch=gfx1101
	./bin/test