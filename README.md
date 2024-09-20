Fork of [https://github.com/xenova/transformers.js](https://github.com/xenova/transformers.js)

For more info on how this module is used and why it exists as a fork, check out [this Notion doc](https://www.notion.so/penfold/Feature-Extraction-Service-dd88a60c131c487c8ee92671786e2b14)

# Building

When running `yarn ap dev` from the web monorepo locally, we run the code using `ts-node-dev`. When this module is treated as an ESM, it is not transpiling correctly when ran through `ts-node-dev`. Therefore, we need to provide a CJS bundle, which removes the need of transforming ESM to CJS both during local dev and deployment.

To transpile this module and to build the type definitions, make sure you run:

```shell
$ npm i
$ npm run build
```

<p align="center">
    <br/>
    <picture> 
        <source media="(prefers-color-scheme: dark)" srcset="https://github.com/xenova/transformers.js/assets/26504141/bd047e0f-aca9-4ff7-ba07-c7ca55442bc4" width="500" style="max-width: 100%;">
        <source media="(prefers-color-scheme: light)" srcset="https://github.com/xenova/transformers.js/assets/26504141/84a5dc78-f4ea-43f4-96f2-b8c791f30a8e" width="500" style="max-width: 100%;">
        <img alt="transformers.js javascript library logo" src="https://github.com/xenova/transformers.js/assets/26504141/84a5dc78-f4ea-43f4-96f2-b8c791f30a8e" width="500" style="max-width: 100%;">
    </picture>
    <br/>
</p>
