CC = emcc
CFLAGS = -s EXPORTED_FUNCTIONS="['_getDrives', '_freeStringArray']" -s EXTRA_EXPORTED_RUNTIME_METHODS="['ccall', 'cwrap']" -s MODULARIZE=1

all: system-io.js

system-io.wasm: system-io.c
	$(CC) $(CFLAGS) -o system-io.wasm system-io.c

system-io.js: system-io.wasm
	wasm-pack -s --target web -o system-io.js system-io.wasm

clean:
	rm -f system-io.wasm system-io.js