from lithiumscope.datasets.georoc_acquisition import (
    parse_andean_arc_files,
)


def test_parse_andean_arc_files_from_dataverse_payload():
    payload = {
        "status": "OK",
        "data": {
            "latestVersion": {
                "files": [
                    {
                        "dataFile": {
                            "id": 101,
                            "filename": "2026_ANDEAN_ARC_part2.csv",
                            "filesize": 200,
                            "persistentId": "doi:part2",
                            "checksum": {
                                "type": "MD5",
                                "value": "b" * 32,
                            },
                        }
                    },
                    {
                        "dataFile": {
                            "id": 100,
                            "filename": "2026_ANDEAN_ARC_part1.csv",
                            "filesize": 100,
                            "persistentId": "doi:part1",
                            "checksum": {
                                "type": "MD5",
                                "value": "a" * 32,
                            },
                        }
                    },
                    {
                        "dataFile": {
                            "id": 102,
                            "filename": "2026_ANDEAN_ARC_part3.csv",
                            "filesize": 300,
                            "persistentId": "doi:part3",
                            "checksum": {
                                "type": "MD5",
                                "value": "c" * 32,
                            },
                        }
                    },
                    {
                        "dataFile": {
                            "id": 999,
                            "filename": "OTHER_ARC.csv",
                            "filesize": 10,
                        }
                    },
                ]
            }
        },
    }

    files = parse_andean_arc_files(payload)

    assert [item.part for item in files] == [1, 2, 3]
    assert [item.file_id for item in files] == [100, 101, 102]
    assert files[0].checksum_type == "MD5"
    assert files[2].persistent_id == "doi:part3"
