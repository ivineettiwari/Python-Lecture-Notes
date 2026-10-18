import config
from database import init_db, repository


def test_session_lifecycle(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "t.db")
    init_db()

    sid, created = repository.get_or_create_session("a.pdf", "hash1", "some text")
    assert created
    sid2, created2 = repository.get_or_create_session("a.pdf", "hash1", "some text")
    assert sid2 == sid and not created2          # same doc -> same session

    repository.save_generation(sid, "seed", 5, 0.7, "seed out")
    assert len(repository.list_generations(sid)) == 1

    repository.delete_session(sid)
    assert repository.get_session(sid) is None
