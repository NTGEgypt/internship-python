# Egyptian ID OCR — module layout

```
ocr_egyptian_id/
├── main.py                  # CLI entry point: image path -> JSON on stdout
├── config.py                # ROOT/MODEL_DIR paths, DEVICE selection
├── models/                  # put detect_id_card.pt, detect_odjects.pt, detect_id.pt here
├── requirements.txt
└── core/
    ├── constants.py         # digit tables, address stopwords, governorate codes
    ├── models.py             # loads YOLO + PaddleOCR + EasyOCR once at import time
    ├── text_utils.py         # clean_space, arabic_normalize, numeric_text, order_rtl_lines
    ├── image_quality.py      # check_image_quality gate (blur/brightness/glare/contrast)
    ├── preprocessing.py      # enhance_card_for_ocr, build_variants, nid_variants, crops
    ├── ocr_engines.py        # ocr_ensemble: runs all 4 OCR engines, tags results with x/y
    ├── field_selection.py    # choose_field, choose_address (RTL-aware)
    ├── nid_extraction.py     # decode_nid, yolo_digit_text, vote_nid_candidates
    └── pipeline.py           # process_image() — orchestrates every stage above
```

## Where to make changes

- **A field is being read wrong (name/address/serial):** `core/field_selection.py`
  or the crop shape in `core/preprocessing.py`.
- **National ID is null too often:** `core/nid_extraction.py` — specifically
  `vote_nid_candidates`'s acceptance threshold, or `nid_variants` in
  `preprocessing.py` for adding a new image-enhancement pass.
- **Multi-word Arabic text comes out in the wrong order:** `core/text_utils.py`,
  `order_rtl_lines` — tune `line_tol` if lines are being merged/split incorrectly.
- **Want to change what counts as a "bad" capture:** `core/image_quality.py`.
- **Adding a new field type entirely (e.g. expiry date):** add it to `LABELS` in
  `core/pipeline.py`, add a `choose_*` function in `field_selection.py` if it
  needs custom logic, then wire it into `process_image`'s return dict.

## Running

```bash
pip install -r requirements.txt
# place YOLO weights in models/
python main.py path/to/id_card.jpg
```

## Mobile API

Start the API from the `Backend` directory:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

The mobile app should send a `multipart/form-data` request with an `image`
field containing a JPEG or PNG (maximum 15 MB):

```text
POST /api/v1/ocr/process
```

The backend processes the image and saves the original upload and OCR result
to MySQL before replying. On success it returns HTTP `200`, a database
submission ID, and the OCR response:

```json
{
  "success": true,
  "submission_id": 42,
  "result": { "national_id": "...", "full_name": "..." }
}
```

`GET /health` can be used to check whether the API process is available.
Configure MySQL once by copying `.env.example` to `.env` and replacing its
placeholder values. The `.env` file is ignored by Git and is loaded
automatically when the backend starts. System environment variables, if set,
take precedence over `.env` values.

## MySQL persistence for the Streamlit app

Create the `egyptian_id_ocr.id_ocr_results` table using the supplied database
script, then set these environment variables before starting Streamlit:

```powershell
$env:MYSQL_HOST = "127.0.0.1"       # optional; this is the default
$env:MYSQL_PORT = "3306"            # optional; this is the default
$env:MYSQL_DATABASE = "egyptian_id_ocr" # optional; this is the default
$env:MYSQL_USER = "your_application_user"
$env:MYSQL_PASSWORD = "your_password"
streamlit run app.py
```

Each submitted image and its unmodified `process_image()` JSON result are
inserted into `id_ocr_results` with the default review status of `PENDING`.
No credentials are stored in the source code.
