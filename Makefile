.PHONY: build run clean

build:
	@mkdir -p bin profiling
	hipcc -O3 -Iinclude main.cpp kernels/kernels.hip -o bin/out --offload-arch=gfx1101

run:
	./bin/out

clean:
	rm -f bin/out
