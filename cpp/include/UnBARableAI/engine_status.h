#pragma once

#include <cstdint>

namespace UnBARableAINS {

/**
 * \brief Enum to be serialized into shared-memory.
 *
 * The entries are always with respect to the AI associated with the shared-memory, meaning if two
 * of our AIs were to battle it out, one would issue \ref GAME_WON and the other \ref GAME_LOST.
 */
enum class EngineStatus : std::uint8_t {
    /**
     * \brief Signals the initial engine process startup.
     *
     * This is only used by the python side which launches the process. After that, the engine is
     * configured so it restarts after each episode, reducing the waiting time. While waiting for
     * that, the state will reamin either \ref GAME_WON or \ref GAME_LOST. Once the engine has been
     * reloaded, we switch to \ref READY.
     */
    STARTING = 0,
    /**
     * \brief Engine is finished loading and ready to play a game.
     *
     * This state is meant to describe the engine to be ready for use but not yet actually being
     * used, i.e. that the process should be halted by the python side.
     */
    READY = 1,
    /// \brief A game is running.
    RUNNING = 2,
    /// \brief The AI won the game,
    GAME_WON = 3,
    /// \brief The AI lost the game.
    GAME_LOST = 4,
};

}  // namespace UnBARableAINS
