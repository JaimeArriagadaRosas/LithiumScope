from io import StringIO

from lithiumscope.runtime.console_status import Spinner


def test_spinner_does_not_spam_non_tty_streams():
    stream = StringIO()
    spinner = Spinner("Preparando", stream=stream).start()
    spinner.update("Paso 1")
    spinner.update("Paso 2")
    spinner.succeed("Completado")

    output = stream.getvalue().splitlines()
    assert output == ["Preparando", "[OK] Completado"]
