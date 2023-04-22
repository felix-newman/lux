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
        for _, factory in self.game_state.factories[self.player].items():
            player_factories[factory.pos_slice] = 1

        return player_factories

    def get_factory_slice_at_position(self, position: np.array) -> np.array:
        for _, factory in self.game_state.factories[self.player].items():
            if factory.pos[0] - 1 <= position[0] <= factory.pos[0] + 1 and factory.pos[1] - 1 <= position[1] <= factory.pos[1] + 1:
                return factory.pos_slice
        return None

    @property
    def board(self) -> Board:
        return self.game_state.board

    @property
    def real_env_steps(self) -> int:
        return self.game_state.real_env_steps

