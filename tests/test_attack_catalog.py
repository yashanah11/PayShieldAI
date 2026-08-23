import json

def test_attack_catalog():
    with open("attacks/attack_catalog.json", encoding="utf-8-sig") as f:
        attacks = json.load(f)

    assert len(attacks) == 8

    ids = [a["attack_id"] for a in attacks]
    assert len(set(ids)) == 8

    for attack in attacks:
        assert attack["name"]
        assert attack["category"]
        assert attack["payment_rail"]
        assert attack["description"]
        assert isinstance(attack["signals"], list)
