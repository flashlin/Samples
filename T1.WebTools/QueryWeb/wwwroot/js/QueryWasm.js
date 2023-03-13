const loadWasmModule = async (wasmUrl) => {
  const response = await fetch(wasmUrl);
  const buffer = await response.arrayBuffer();
  const module = await WebAssembly.instantiate(buffer);

  const getString = (ptr) => {
    const mem = new Uint8Array(module.exports.memory.buffer);
    let str = "";
    while (mem[ptr] !== 0) {
      str += String.fromCharCode(mem[ptr]);
      ptr++;
    }
    return str;
  };

  return {
    exports: module.exports,
    getString,
  };
};

const getDrivesAsync = async () => {
  //const response = await fetch("system-io.wasm");
  //const buffer = await response.arrayBuffer();
  //const module = await WebAssembly.instantiate(buffer);
  const module = await loadWasmModule("/js/QueryWasm.wasm");
  const getDrives = module.exports._getDrives;
  const freeStringArray = module.exports._freeStringArray;

  //   const getString = (ptr) => {
  //     const mem = new Uint8Array(module.exports.memory.buffer);
  //     let str = "";
  //     while (mem[ptr] !== 0) {
  //       str += String.fromCharCode(mem[ptr]);
  //       ptr++;
  //     }
  //     return str;
  //   };

  const resultPtr = getDrives();
  const result = [];
  let i = 0;
  while (resultPtr[i] !== 0) {
    result.push(module.getString(resultPtr[i]));
    i++;
  }
  freeStringArray(resultPtr);

  return result;
};

window.getDrivesAsync = getDrivesAsync;
