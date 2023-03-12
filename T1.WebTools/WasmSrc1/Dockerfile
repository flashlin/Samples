FROM gcc:latest
COPY . /app
WORKDIR /app/
RUN gcc -o wasm-system-io system-io.c
CMD ["./wasm-system-io"]