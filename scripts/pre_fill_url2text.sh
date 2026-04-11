#!/bin/bash

# Configuration
SPLIT="train"           # Options: train, val, test
USE_PLAYWRIGHT=true

START_IDX=0
END_IDX=10

echo "Split: $SPLIT, Use Playwright: $USE_PLAYWRIGHT"

INPUT_BASE="dataset/AVerImaTeC_Shared_Task/Knowledge_Store/${SPLIT}/text_related"
OUTPUT_BASE="dataset/AVerImaTeC_Shared_Task/Knowledge_Store/${SPLIT}/text_related"

# 1. Process text_related_store_text
echo ""
echo "=== [1/2] Processing text_related_store_text_${SPLIT} ==="
python src/retrieval/fill_url2text.py \
    --input_dir ${INPUT_BASE}/text_related_store_text_${SPLIT} \
    --output_dir ${OUTPUT_BASE}/text_related_store_text_${SPLIT}_filled \
    --start_idx $START_IDX \
    --end_idx $END_IDX \
    ${USE_PLAYWRIGHT:+--use_playwright}

# 2. Process image_related_store_text
echo ""
echo "=== [2/2] Processing image_related_store_text_${SPLIT} ==="
python src/retrieval/fill_url2text.py \
    --input_dir ${INPUT_BASE}/image_related_store_text_${SPLIT} \
    --output_dir ${OUTPUT_BASE}/image_related_store_text_${SPLIT}_filled \
    --start_idx $START_IDX \
    --end_idx $END_IDX \
    ${USE_PLAYWRIGHT:+--use_playwright}

echo ""
echo "=== All 2 stores processed ==="
