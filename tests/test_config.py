import pytest

from twotor.config import Config


def test_configuration_rejects_large_tts_and_invalid_limits(tmp_path, monkeypatch):
    monkeypatch.setenv("TTS_MODEL", "KittenML/kitten-tts-2")
    with pytest.raises(ValueError, match="lightweight ONNX"):
        Config.load(tmp_path / "missing.env")
    monkeypatch.setenv("TTS_MODEL", Config.tts_model)
    monkeypatch.setenv("MEMORY_LIMIT_MB", "4000")
    with pytest.raises(ValueError, match="256–3000"):
        Config.load(tmp_path / "missing.env")


def test_env_key_and_device_config(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("INPUT_DEVICE", raising=False)
    env = tmp_path / "test.env"
    env.write_text("DEEPSEEK_API_KEY=test-value\nINPUT_DEVICE=3\n")
    cfg = Config.load(env)
    assert cfg.api_key == "test-value"
    assert cfg.input_device == 3


def test_recognition_pause_and_keyterms_config(tmp_path, monkeypatch):
    monkeypatch.setenv("STT_END_PAUSE_SECONDS", "1.6")
    monkeypatch.setenv("STT_KEYTERMS", "DeepSeek, 2tor, , Kubernetes")
    cfg = Config.load(tmp_path / "missing.env")
    assert cfg.stt_end_pause == 1.6
    assert cfg.stt_keyterms == ("DeepSeek", "2tor", "Kubernetes")
    monkeypatch.setenv("STT_END_PAUSE_SECONDS", "0.1")
    with pytest.raises(ValueError, match="STT_END_PAUSE_SECONDS"):
        Config.load(tmp_path / "missing.env")
