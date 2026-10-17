from webui import apply_settings_update, setting_widget_value


def test_a_section_save_keeps_the_other_settings():
    current = {"uvr_models": "spleeter", "crf": 23, "lang": "en_US"}
    merged = apply_settings_update(current, {"uvr_models": "UVR-MDX-NET-Inst_HQ_4"})
    assert merged["uvr_models"] == "UVR-MDX-NET-Inst_HQ_4"
    assert merged["crf"] == 23
    assert current["uvr_models"] == "spleeter"


def test_the_bgm_dropdown_reloads_the_saved_model():
    shown = setting_widget_value(
        "uvr_models", {"uvr_models": "UVR-MDX-NET-Inst_HQ_4"}
    )
    assert shown == "UVR-MDX-NET-Inst_HQ_4"


def test_an_unknown_bgm_model_falls_back_to_the_first_choice():
    assert setting_widget_value("uvr_models", {"uvr_models": "missing"}) == "spleeter"
