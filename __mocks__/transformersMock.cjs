// This file is used to mock the transformers module when running jest tests.
// jest.config.js:
// moduleNameMapper: {
//   "@xenova(.*)":
//     "<rootDir>/../node_modules/@xenova/transformers/__mocks__/transformersMock.cjs",
// },

module.exports = {
  env: {
    allowRemoteModels: false,
    localModelPath: "",
    backends: {
      onnx: {
        wasm: {
          numThreads: 1,
          wasmPaths: "",
        },
      },
    },
  },
  // Needs to a return a promise that resolves to a function
  pipeline: new Promise((res) => res(() => {})),
};
