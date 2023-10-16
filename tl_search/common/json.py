import zlib, json, base64


def json_zip(j, zip_json_key: str = "base64(zip(o))"):

    j = {
        zip_json_key: base64.b64encode(
            zlib.compress(json.dumps(j).encode("utf-8"))
        ).decode("ascii")
    }

    return j


def json_unzip(j, zip_json_key: str = "base64(zip(o))", insist=True):
    try:
        assert j[zip_json_key]
        assert set(j.keys()) == {zip_json_key}
    except:
        if insist:
            raise RuntimeError(
                "JSON not in the expected format {" + str(zip_json_key) + ": zipstring}"
            )
        else:
            return j

    try:
        j = zlib.decompress(base64.b64decode(j[zip_json_key]))
    except:
        raise RuntimeError("Could not decode/unzip the contents")

    try:
        j = json.loads(j)
    except:
        raise RuntimeError("Could interpret the unzipped contents")

    return j
