"""301 rules: scoring, busts, check-outs, undo, turn handling."""

import pytest

from darts.game import Game, GameConfig, checkout_hint


def game(names=("A", "B"), **kw):
    return Game.new(list(names), GameConfig(**kw))


def throw_all(g, *labels):
    for label in labels:
        g.throw_label(label)


class TestScoring:
    def test_darts_come_off_the_score(self):
        g = game()
        throw_all(g, "T20", "T20", "T20")
        assert g.players[0].score == 301 - 180
        assert g.turn_score == 180

    def test_turn_locks_after_three_darts(self):
        g = game()
        throw_all(g, "S1", "S1", "S1")
        assert g.turn_end == "complete"
        assert g.darts_left == 0

    def test_a_locked_turn_ignores_extra_darts(self):
        # The camera keeps watching while the player walks up to pull their
        # darts; a stray blob must not become a fourth dart.
        g = game()
        throw_all(g, "S1", "S1", "S1", "T20")
        assert len(g.turn) == 3
        assert g.players[0].score == 298

    def test_one_eighty_is_called(self):
        g = game()
        g.throw_label("T20")
        g.throw_label("T20")
        calls = g.throw_label("T20")
        assert "scored_180" in calls

    def test_average_is_per_three_darts(self):
        g = game()
        throw_all(g, "T20", "S20", "S20")  # 100 off 3 darts
        assert g.players[0].average == pytest.approx(100.0)


class TestBusts:
    def test_going_below_zero_busts(self):
        g = game()
        g.players[0].score = 20
        calls = g.throw_label("T20")
        assert g.turn_end == "bust"
        assert "bust" in calls
        assert g.players[0].score == 20

    def test_landing_on_one_busts_when_doubling_out(self):
        g = game(double_out=True)
        g.players[0].score = 21
        g.throw_label("S20")
        assert g.turn_end == "bust"
        assert g.players[0].score == 21

    def test_landing_on_one_is_fine_without_double_out(self):
        g = game(double_out=False)
        g.players[0].score = 21
        g.throw_label("S20")
        assert g.turn_end == ""
        assert g.players[0].score == 1

    def test_reaching_zero_on_a_non_double_busts(self):
        g = game(double_out=True)
        g.players[0].score = 20
        g.throw_label("S20")
        assert g.turn_end == "bust"
        assert g.players[0].score == 20

    def test_a_bust_reverts_the_whole_turn(self):
        g = game()
        g.players[0].score = 100
        g.throw_label("T20")  # 40 left
        assert g.players[0].score == 40
        g.throw_label("T20")  # would go to -20
        assert g.turn_end == "bust"
        assert g.players[0].score == 100, "earlier darts in the turn must be given back"

    def test_bust_still_counts_the_darts_thrown(self):
        g = game()
        g.players[0].score = 20
        g.throw_label("T20")
        assert g.players[0].darts_thrown == 1


class TestWinning:
    def test_finishing_on_a_double_wins(self):
        g = game(double_out=True)
        g.players[0].score = 32
        calls = g.throw_label("D16")
        assert g.winner == 0
        assert g.players[0].score == 0
        assert "game_shot" in calls

    def test_bull_finishes_a_double_out_leg(self):
        g = game(double_out=True)
        g.players[0].score = 50
        g.throw_label("BULL")
        assert g.winner == 0

    def test_straight_out_finishes_on_anything(self):
        g = game(double_out=False)
        g.players[0].score = 20
        g.throw_label("S20")
        assert g.winner == 0

    def test_no_darts_register_after_the_win(self):
        g = game()
        g.players[0].score = 32
        g.throw_label("D16")
        assert g.throw_label("T20") == []
        assert g.players[0].score == 0


class TestDoubleIn:
    def test_nothing_counts_until_a_double_lands(self):
        g = game(double_in=True)
        throw_all(g, "T20", "S20")
        assert g.players[0].score == 301
        assert not g.players[0].started

    def test_the_opening_double_counts(self):
        g = game(double_in=True)
        g.throw_label("D20")
        assert g.players[0].started
        assert g.players[0].score == 301 - 40


class TestTurnsAndUndo:
    def test_next_player_cycles(self):
        g = game(("A", "B", "C"))
        assert g.player.name == "A"
        g.next_player()
        assert g.player.name == "B"
        g.next_player()
        g.next_player()
        assert g.player.name == "A"

    def test_next_player_clears_the_turn(self):
        g = game()
        throw_all(g, "S1", "S1")
        g.next_player()
        assert g.turn == []
        assert g.turn_end == ""

    def test_undo_restores_the_score(self):
        g = game()
        g.throw_label("T20")
        assert g.players[0].score == 241
        assert g.undo()
        assert g.players[0].score == 301
        assert g.turn == []

    def test_undo_reverses_a_bust(self):
        g = game()
        g.players[0].score = 100
        throw_all(g, "T20", "T20")
        assert g.turn_end == "bust"
        g.undo()
        assert g.turn_end == ""
        assert g.players[0].score == 40, "back to the state after the first dart"

    def test_undo_reverses_a_player_change(self):
        g = game()
        g.throw_label("T20")
        g.next_player()
        assert g.current == 1
        g.undo()
        assert g.current == 0
        assert len(g.turn) == 1

    def test_undo_on_a_fresh_game_is_a_no_op(self):
        assert not game().undo()

    def test_reset_restores_everyone(self):
        g = game()
        throw_all(g, "T20", "T20", "T20")
        g.next_player()
        g.throw_label("T20")
        g.reset()
        assert [p.score for p in g.players] == [301, 301]
        assert g.current == 0
        assert all(p.darts_thrown == 0 for p in g.players)


class TestCheckouts:
    @pytest.mark.parametrize("score,expected", [
        (40, ("D20",)),
        (32, ("D16",)),
        (50, ("BULL",)),
        (2, ("D1",)),
        (170, ("T20", "T20", "BULL")),
    ])
    def test_known_finishes(self, score, expected):
        assert checkout_hint(score) == expected

    @pytest.mark.parametrize("score", [169, 168, 166, 165, 163, 162, 159])
    def test_bogey_numbers_have_no_finish(self, score):
        assert checkout_hint(score) is None

    def test_out_of_range_scores(self):
        assert checkout_hint(171) is None
        assert checkout_hint(0) is None
        assert checkout_hint(1) is None

    def test_every_checkout_actually_adds_up(self):
        values = {"BULL": 50, "25": 25}
        for score in range(2, 171):
            out = checkout_hint(score)
            if out is None:
                continue
            total = 0
            for name in out:
                if name in values:
                    total += values[name]
                else:
                    mult = {"S": 1, "D": 2, "T": 3}[name[0]]
                    total += mult * int(name[1:])
            assert total == score, f"{score} -> {out} sums to {total}"

    def test_every_checkout_ends_on_a_double(self):
        for score in range(2, 171):
            out = checkout_hint(score, double_out=True)
            if out is None:
                continue
            assert out[-1].startswith("D") or out[-1] == "BULL", f"{score} -> {out}"

    def test_checkouts_are_at_most_three_darts(self):
        for score in range(2, 171):
            out = checkout_hint(score)
            assert out is None or len(out) <= 3


class TestSerialisation:
    def test_state_dict_is_json_shaped(self):
        import json

        g = game()
        throw_all(g, "T20", "D5")
        blob = json.dumps(g.to_dict())
        assert '"T20"' in blob
        assert json.loads(blob)["players"][0]["score"] == 301 - 70


class TestBustReason:
    """"Bust" on its own is the most confusing message in darts.

    Hitting your exact remaining score and losing the turn for it looks like a
    broken scoreboard unless you already know the double-out rule did it -- and
    it was reported as one.
    """

    def _game(self, score, **cfg):
        g = Game.new(["A"], GameConfig(start_score=301, **cfg))
        g.players[0].score = score
        return g

    def test_landing_on_zero_without_a_double_says_so(self):
        g = self._game(20, double_out=True)
        g.throw_label("S20")
        assert g.turn_end == "bust"
        assert g.bust_reason == "not_a_double"

    def test_the_same_dart_wins_with_double_out_off(self):
        g = self._game(20, double_out=False)
        g.throw_label("S20")
        assert g.turn_end == "win"
        assert g.bust_reason == ""

    def test_a_double_on_zero_still_wins(self):
        g = self._game(40, double_out=True)
        g.throw_label("D20")
        assert g.turn_end == "win"

    def test_going_past_zero_is_reported_as_overshooting(self):
        g = self._game(20, double_out=True)
        g.throw_label("T20")
        assert g.bust_reason == "overshot"

    def test_leaving_one_is_reported_separately(self):
        g = self._game(21, double_out=True)
        g.throw_label("S20")
        assert g.bust_reason == "left_one"

    def test_the_reason_clears_when_the_turn_does(self):
        g = self._game(20, double_out=True)
        g.throw_label("S20")
        g.next_player()
        assert g.bust_reason == ""

    def test_undo_restores_the_reason(self):
        """_restore reads every key it snapshots; a missing one is a KeyError."""
        g = self._game(20, double_out=True)
        g.throw_label("S20")
        g.undo()
        assert g.bust_reason == ""
        assert g.turn_end == ""

    def test_it_reaches_the_client(self):
        g = self._game(20, double_out=True)
        g.throw_label("S20")
        assert g.to_dict()["bust_reason"] == "not_a_double"
