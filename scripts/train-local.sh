python -m trainer.task \
    --job-dir . \
    --train-data-file gs://ai-platform-bucket-ollie/data/df.pickle \
    --batch-size 1 \
    --max_epochs 10 \
