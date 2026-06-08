Assumptions I am making
These are my assumptions, because you asked me to assume the shared-memory side already exists:

1 Your C++ side exports a read-only world snapshot through shared memory.
2 The Python side can access that snapshot through a reader object with:

connect()
close()
read_snapshot() -> dict


3 read_snapshot() returns something like:
{
    "frame": 12345,
    "units_by_id": {
        17: {
            "unit_id": 17,
            "unit_def_id": 35,
            "unit_def_name": "armwar",
            "human_name": "Warrior",
            "team_id": 0,
            "ally_team_id": 0,
            "health": 450.0,
            "max_health": 750.0,
            "pos_x": 1024.0,
            "pos_y": 85.0,
            "pos_z": 2048.0,
            "los_radius": 490.0,
            "air_los_radius": 490.0,
            "is_dead": False,
            "being_built": False,

            "build_progress": 1.0,
            "capture_progress": 0.0,
            "paralyze_damage": 0.0,
        },
        ...
    }
}
4 Your Python agent ids are equal to engine unit_ids. If not, you need a separate agent_id -> unit_id mapping layer.

5 Your exporter already handles synchronization / double buffering, so Python always sees a consistent snapshot.s