import pytest

@pytest.fixture(scope="session")
def test_db_path(tmp_path_factory):
    # เสก folder ชั่วคราวให้ test โดยเฉพาะ
    tmp_dir = tmp_path_factory.mktemp("walkie_test_db_data")
    return str(tmp_dir)
