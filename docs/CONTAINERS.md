# Container images

The [Container workflow](../.github/workflows/container.yml) builds Gufo's
production package and publishes a Docker-compatible image to
`ghcr.io/<owner>/<repository>`. In the NuMind fork this is
`ghcr.io/numindai/gufo`.

CMake still owns compilation and Nix supplies the pinned toolchain. The image
contains the production runtime, its runtime dependencies and CA certificates.
It retains ROCm's transitive dependencies but does not add the development
shell, source checkout or model weights.
The target is Linux amd64 on AMD Strix Halo (`gfx1151`); this is not a CPU
inference image or a multi-architecture build.

## Automatic publishing

Push the workflow and its Nix image definition to the repository. GitHub
Actions builds and publishes on pushes to `main`, `fork-enhancements`, or the
`codex/jinja2-template` preview branch, and on stable version tags. It does not
publish from pull requests.

| Trigger | Published tags |
| --- | --- |
| Branch push | `branch-<sanitized branch name>` and `sha-<full commit>` |
| Default-branch push | Also updates `edge` |
| Stable `vX.Y.Z` tag | `X.Y.Z`, `latest`, and `release-sha-<full commit>` |
| Manual branch run | `branch-<sanitized branch name>` and `sha-<full commit>`; also `edge` if it is the default branch |

Release tags must match `version.txt`. Release builds use the existing
`release` package, while branch builds report `development`. The separate
revision tags distinguish those binaries when the source commit is the same.
`latest` follows the most recently published stable tag; use an exact version
or image digest when deploying a fixed build.

The Jinja preview branch publishes `branch-codex-jinja2-template` automatically,
without updating `edge` or `latest`. For other feature branches, choose
**Actions → Container → Run workflow** and select the branch.
GitHub requires a manually dispatched workflow to exist on the
default branch first. In a fork, enable Actions if GitHub has disabled inherited
workflows. Wait for a successful run before using its image.

The workflow grants `GITHUB_TOKEN` `contents: read` and `packages: write`.
It authenticates to GHCR with that token, so no registry password or personal
access token needs to be stored as a publishing secret. It also attaches the
source repository and commit as OCI labels. An existing GHCR package may need
its Actions access linked to this repository.

New packages are private by default. Change package visibility to public if
anonymous pulls are wanted; otherwise authenticate Docker with a token allowed
to read the package. See GitHub's
[container registry documentation](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry).

The job uses an Ubuntu x86-64 runner, limits parallel compilation to two cores,
and reuses the Nix cache. ROCm builds need substantial disk space, so the job
removes unused Android, .NET and GHC installations from its disposable runner.
The first build can take substantially longer than cached builds. It runs
version/help smoke checks without a GPU; inference correctness still requires
a Strix Halo machine.

This workflow publishes directly to the fork's registry namespace. The
existing upstream [Toolboxes release flow](RELEASING.md#release-flow) is
separate and is not needed to publish these images.

## Run with a local GGUF

After a successful default-branch publication:

```sh
docker pull ghcr.io/numindai/gufo:edge

docker run --rm \
  --device /dev/kfd --device /dev/dri \
  --group-add "$(stat -c '%g' /dev/kfd)" \
  --group-add "$(stat -c '%g' /dev/dri/renderD128)" \
  -p 127.0.0.1:8080:8080 \
  -v /path/to/models:/models:ro \
  ghcr.io/numindai/gufo:edge \
  serve llm --model /models/model.gguf --context 16384 \
  --host 0.0.0.0 --port 8080
```

Use the render node for the target GPU if it differs from `renderD128`.
The host supplies the AMD kernel driver and device nodes; the image supplies
the userspace runtime. The image runs as UID/GID `1000:1000`; supplemental
groups grant access to the GPU. Bind-mounted models must be readable by that
user. Override Docker's `--user` when local file ownership requires it.

For NuExtract, add `--jinja --served-model-name nuextract --speculative off`
after `serve llm` and use the request in the
[chat template guide](CHAT-TEMPLATES.md#serve-an-embedded-template).
Jinja remains opt-in in the image.

The entrypoint is `gufo`, so arguments are ordinary Gufo commands:

```sh
docker run --rm ghcr.io/numindai/gufo:edge --version
docker run --rm ghcr.io/numindai/gufo:edge serve llm --help
```

For disk caching, mount a writable directory and pass its container path to
`--cache-disk`. Runtime configuration and weights are supplied at launch;
they do not require rebuilding the image.

## Build locally

Stage new task-owned files before using a Git flake: Nix excludes untracked
files. Build the development image, load its stream into Docker, and run a
smoke check:

```sh
nix build .#docker --out-link result-image
./result-image | docker load
docker run --rm gufo:build --version
```

`docker-release` selects the release package instead. Both outputs stream a
local image named `gufo:build`; the Actions workflow adds repository labels and
publication tags after loading. The image definition is
[`.devops/nix/image.nix`](../.devops/nix/image.nix).
