import pandas as pd

from scripts.ingest import run_ingestion


def _write_batch(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def test_run_ingestion_scores_and_writes_output(tmp_path, tiny_baseline_model):
    incoming = tmp_path / "incoming"
    processed = tmp_path / "incoming" / "processed"
    triaged = tmp_path / "triaged"
    incoming.mkdir(parents=True)

    _write_batch(incoming / "batch1.csv", {
        "text": ["wildfire spreading through the valley", "lovely sunny picnic day"],
        "keyword": ["fire", None],
    })

    written = run_ingestion(incoming, processed, triaged, model=tiny_baseline_model)

    assert len(written) == 1
    output = pd.read_csv(written[0])
    assert len(output) == 2
    assert "label" in output.columns
    assert "probability" in output.columns


def test_run_ingestion_moves_processed_files(tmp_path, tiny_baseline_model):
    incoming = tmp_path / "incoming"
    processed = tmp_path / "incoming" / "processed"
    triaged = tmp_path / "triaged"
    incoming.mkdir(parents=True)

    _write_batch(incoming / "batch1.csv", {"text": ["fire in the hills"]})

    run_ingestion(incoming, processed, triaged, model=tiny_baseline_model)

    assert not (incoming / "batch1.csv").exists()
    assert (processed / "batch1.csv").exists()


def test_run_ingestion_skips_files_without_text_column(tmp_path, tiny_baseline_model, capsys):
    incoming = tmp_path / "incoming"
    processed = tmp_path / "incoming" / "processed"
    triaged = tmp_path / "triaged"
    incoming.mkdir(parents=True)

    _write_batch(incoming / "bad.csv", {"not_text": ["whatever"]})

    written = run_ingestion(incoming, processed, triaged, model=tiny_baseline_model)

    assert written == []
    assert "skipping" in capsys.readouterr().out


def test_run_ingestion_no_files_is_noop(tmp_path, tiny_baseline_model):
    incoming = tmp_path / "incoming"
    processed = tmp_path / "incoming" / "processed"
    triaged = tmp_path / "triaged"

    written = run_ingestion(incoming, processed, triaged, model=tiny_baseline_model)
    assert written == []
