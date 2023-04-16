from components.utils import get_path


class TestGetPath:
    def test_returns_list(self):
        start = (0, 0)
        end = (5, 5)
        path = get_path(start, end)
        assert isinstance(path, list)

    def test_returns_expected_path_1(self):
        start = (0, 0)
        end = (2, 2)
        path = get_path(start, end)
        expected_path = [(1, 0), (2, 0), (2, 1), (2, 2)]
        assert path == expected_path

    def test_returns_expected_path_2(self):
        start = (0, 0)
        end = (1, 2)
        path = get_path(start, end)
        expected_path = [(1, 0), (1, 1), (1, 2)]
        assert path == expected_path

    def test_handles_negative_deltas(self):
        start = (0, 0)
        end = (-2, -2)
        path = get_path(start, end)
        expected_path = [(-1, 0), (-2, 0), (-2, -1), (-2, -2)]
        assert path == expected_path

    def test_handles_horizontal_line(self):
        start = (0, 0)
        end = (5, 0)
        path = get_path(start, end)
        expected_path = [(1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]
        assert path == expected_path

    def test_handles_vertical_line(self):
        start = (0, 0)
        end = (0, 5)
        path = get_path(start, end)
        expected_path = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]
        assert path == expected_path

    def test_empty_list_for_same_start_end(self):
        start = (0, 0)
        end = (0, 0)
        path = get_path(start, end)
        expected_path = []
        assert path == expected_path
