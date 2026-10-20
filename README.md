# elite

qrank


## Runtime

- `stack_choice`: `fullstack`
- `ci`: `none`
- `runtime`: `Python + GPU + Node.js`
- `python_version`: `3.11`
- `base_python_image`: `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel`

- `pytorch_cuda_tag`: `2.7.1-cuda12.8-cudnn9-devel`


- `with_node`: `True`

- `node_version`: `24`
- `node_manager`: `nvm`
- `nvm_version`: `0.40.3`
- `pnpm_version`: `latest`
- `package_manager`: `pnpm`



## Package Layout

- `src/qrank/` contains the Python package.
- `tests/` contains test coverage for the package.
- `docs/` contains the MkDocs content for the project site.
- `.devcontainer/` contains the development container setup.

## Development

Open this folder in VS Code and run `Dev Containers: Reopen in Container`.


Node.js tooling is installed and managed with `nvm` inside the devcontainer. The selected default version is `24`. This adds Node and pnpm to the environment, but it does not scaffold a frontend app.

```sh
nvm current
pnpm --version
```


To preview the docs locally, run:

```sh
uv run mkdocs serve
```


To update this project from the template later, run:

```sh
uvx copier update
```


## Preparing PrepBench eliteset



