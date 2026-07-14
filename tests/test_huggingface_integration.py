from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from researchforge.integrations.huggingface import HuggingFaceIntegration


def test_search_dataset_ranks_overlap_and_returns_streaming_preview():
    api = MagicMock()
    api.list_datasets.return_value = [
        SimpleNamespace(id="owner/other", description="weather", tags=[], likes=100, downloads=1000),
        SimpleNamespace(id="owner/ipl-winner", description="IPL cricket match winner", tags=["cricket"], likes=2, downloads=3),
    ]
    stream = iter([{"winner": "Mumbai Indians"}])
    with patch("huggingface_hub.HfApi", return_value=api), \
         patch("datasets.get_dataset_split_names", return_value=["train"]), \
         patch("datasets.load_dataset", return_value=stream) as load:
        result = HuggingFaceIntegration().search_datasets("IPL match winner")

    assert result[0]["id"] == "owner/ipl-winner"
    assert result[0]["sample"] == {"winner": "Mumbai Indians"}
    assert load.call_count == 2
    assert result[0]["provenance"]["agent"] == "dataset"


def test_search_models_returns_metadata_and_provenance():
    api = MagicMock()
    api.list_models.return_value = [
        SimpleNamespace(id="google/gemma", pipeline_tag="text-generation", library_name="transformers", tags=["text"], likes=5, downloads=20)
    ]
    with patch("huggingface_hub.HfApi", return_value=api):
        result = HuggingFaceIntegration().search_models("text generation")

    assert result[0]["id"] == "google/gemma"
    assert result[0]["pipeline_tag"] == "text-generation"
    assert result[0]["provenance"]["agent"] == "training_planner"


def test_download_dataset_uses_dataset_snapshot(tmp_path):
    with patch("huggingface_hub.snapshot_download", return_value="downloaded") as download:
        result = HuggingFaceIntegration(token="hf-test").download_dataset("owner/data", str(tmp_path))

    assert result == "downloaded"
    assert download.call_args.kwargs["repo_type"] == "dataset"
    assert download.call_args.kwargs["token"] == "hf-test"
    assert str(tmp_path / "owner_data") == download.call_args.kwargs["local_dir"]


def test_search_is_safe_when_huggingface_is_offline():
    with patch("huggingface_hub.HfApi", side_effect=RuntimeError("offline")):
        assert HuggingFaceIntegration().search_datasets("anything") == []
