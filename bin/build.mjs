#!/usr/bin/env ts-node
import { build } from "esbuild";
import * as path from "path";
import shell from "shelljs";

function p(...strings) {
  return path.resolve(import.meta.dirname, "..", ...strings);
}

async function run() {
  console.log("clearing dist...");

  shell.rm("-rf", "dist");

  console.log("running esbuild...");

  await build({
    bundle: true,
    platform: "node",
    target: "node16",
    sourcemap: "external",
    keepNames: true,
    external: ["sharp"],
    entryPoints: [p("src/transformers.js")],
    outdir: p("dist"),
    inject: [path.resolve(import.meta.dirname, "importMetaUrl.js")],
    define: {
      // The next line shims the ES definition of import.meta
      // The `import_meta_url` is defined in './importMetaUrl.js' and injected `inject` config option
      // More on this here https://github.com/evanw/esbuild/issues/1492#issuecomment-893144483
      "import.meta.url": "import_meta_url",
    },
    loader: {
      ".node": "file",
    },
  });

  console.log("done");
}

run().catch((err) => {
  console.log(err);
  process.exit(1);
});
