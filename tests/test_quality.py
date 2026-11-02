from twinpix.model import Photo
from twinpix.quality import is_better, pick_keeper


def photo(name, w=4000, h=3000, size=3_000_000, sharp=100.0, mtime=0):
    return Photo(name, size, mtime, w, h, sharpness=sharp)


def test_higher_resolution_wins_even_if_less_sharp():
    big = photo("big", sharp=60)
    small = photo("small", w=1600, h=1200, sharp=120)
    assert pick_keeper([small, big]) is big


def test_sharpest_burst_frame_wins():
    frames = [photo("a", sharp=40), photo("b", sharp=95, size=2_000_000), photo("c", sharp=55)]
    assert pick_keeper(frames).path == "b"


def test_small_differences_fall_through_to_file_size():
    a = photo("a", w=4000, sharp=100, size=2_000_000)
    b = photo("b", w=3950, sharp=108, size=3_500_000)  # within 5 % / 15 % tolerances
    assert pick_keeper([a, b]) is b


def test_older_file_wins_a_full_tie():
    old, new = photo("z_old", mtime=1), photo("a_new", mtime=2)
    assert pick_keeper([new, old]) is old


def test_complete_tie_is_deterministic():
    a, b = photo("a"), photo("b")
    assert pick_keeper([a, b]) is a and pick_keeper([b, a]) is a
    assert not is_better(a, a)
