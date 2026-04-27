# deploy/

Template files that are scrubbed in this public repo. The originals
(authors, real data paths, sample-benchmark runs, rendered PDFs) live
in a private backup at:

```
gdrive (rclone remote `i:`):/_rk/metadata-aware-nwp-in-sla/deploy/
```

## How to use

The committed `metadata/authors.tex` and `codebase/configs/paths.yaml`
are anonymized / placeholder versions that let the project build and
import out of the box. To run with real values, edit them locally —
the templates here are just reference shapes:

```bash
cp deploy/authors.tex.example metadata/authors.tex      # then fill in
cp deploy/paths.yaml.example codebase/configs/paths.yaml  # then fill in
```

Do not commit your local edits.
