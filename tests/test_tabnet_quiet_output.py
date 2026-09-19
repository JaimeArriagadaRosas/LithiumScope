from lithiumscope.model_1.training.tabnet import _fit_quietly


class _NoisyModel:
    def fit(self, *args, **kwargs):
        print("internal library chatter")
        return self


def test_tabnet_internal_stdout_is_captured(capsys):
    captured = _fit_quietly(_NoisyModel())

    terminal = capsys.readouterr()
    assert terminal.out == ""
    assert "internal library chatter" in captured
