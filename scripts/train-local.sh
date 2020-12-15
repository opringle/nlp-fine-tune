python -m trainer.task \
    --job-dir . \
    --train-data-file gs://ai-platform-bucket-ollie/data/df.pickle \
    --epochs 30 \
    --batch-size 1