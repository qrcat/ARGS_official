
download modelsplat

```bash
huggingface-cli download --repo-type dataset --resume-download ShapeSplats/ModelNet_Splats --local-dir modelsplat --local-dir-use-symlinks False
```

unzip modelsplat
```bash
python unzip --input /path/to/modelsplat --output /path/to/modelsplat_unzip
```

build data

```bash
# fetch cpu
cat /proc/cpuinfo | grep "cpu cores" | uniq
# workers <= fetch cpu
python build_merge_list2.py --input ../modelsplat_unzip/train/ --output ../modelsplat_pkl --filter '*/*.ply' --workers 96
```
