from dataclasses import dataclass

import numpy as np

from lux.kit import GameState, Board


@dataclass
class ExtendedGameState:
    game_state: GameState
    player: str

    @property
    def player_factories(self) -> np.array:
        player_factories = np.zeros_like(self.game_state.board.rubble)
        for _, factory in self.game_state.factories[self.player]:
            player_factories[factory.pos_slice] = 1

        return player_factories

    @property
    def board(self) -> Board:
        return self.game_state.board

