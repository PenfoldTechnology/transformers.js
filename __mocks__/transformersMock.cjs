// This file is used to mock the transformers module when running jest tests.
// jest.config.js:
// moduleNameMapper: {
//   "@xenova(.*)":
//     "<rootDir>/../node_modules/@xenova/transformers/__mocks__/transformersMock.cjs",
// },

module.exports = {
  env: {
    allowRemoteModels: false,
  },
  backends: {
    onnx: {
      wasm: {
        numThreads: 1,
        wasmPaths: "",
      },
    },
  },
  localModelPath: "",
  pipeline: () => new Promise((res) => res()),
};
