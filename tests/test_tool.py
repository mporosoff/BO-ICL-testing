import os

from boicl import BOICLTool, Pool


def test_tell_tools(tmp_path):
    pool_list = ["red", "green", "blue"]
    pool = Pool(pool_list)
    tool = BOICLTool(pool)
    test_csv = tmp_path / "test.csv"
    with open(test_csv, "w") as f:
        f.write("yellow, 1\n")
        f.write("teal, 2\n")
    tool("Ask")
    s = tool(f"Tell {test_csv}")
    assert s == "Added 2 training examples."
    if os.environ.get("RUN_LIVE_API_TESTS") == "1":
        tool("Ask")
        tool("Best")
