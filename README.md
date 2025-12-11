### SAM3 testing sandbox

**1. Clone the repo with submodules**

```bash
git clone --recurse-submodules https://github.com/weijieyong/sam3-box.git
```

**2. Build the docker image**

```bash
docker compose build
```

**3. Log in to Hugging Face**
You only need to do this once.

```bash
hf auth login
```

**4. Run the script**
The first run takes a while since it needs to download the checkpoint.

```bash
docker compose run --rm sam3-app python3 test_scripts/run_sam3.py
```

Tested on RTX 5080.
