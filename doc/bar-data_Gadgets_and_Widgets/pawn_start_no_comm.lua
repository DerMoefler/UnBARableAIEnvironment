function gadget:GetInfo()
    return {
        name    = "RL Configurable Start",
        desc    = "Spawns configured units and reports terminal results",
        author  = "dorian",
        date    = "2026",
        license = "GPLv2",
        layer   = 999999,
        enabled = true,
    }
end


--------------------------------------------------------------------------------
-- Unsynced:
-- Synchronisierte Endnachrichten an die Skirmish AIs weiterleiten.
--------------------------------------------------------------------------------

if not gadgetHandler:IsSyncedCode() then

    local function ForwardGameOver(
        _,
        receivingTeamID,
        message
    )
        receivingTeamID =
            tonumber(receivingTeamID)

        if receivingTeamID == nil
            or type(message) ~= "string"
        then
            Spring.Echo(
                "[RLStart] ERROR: invalid terminal message "
                .. "from synced gadget"
            )

            return
        end

        local result =
            Spring.SendSkirmishAIMessage(
                receivingTeamID,
                message
            )

        Spring.Echo(
            string.format(
                "[RLStart] Sent terminal message to Skirmish AI: "
                .. "teamID=%d message=%s result=%s",
                receivingTeamID,
                message,
                tostring(result)
            )
        )
    end


    function gadget:Initialize()
        gadgetHandler:AddSyncAction(
            "RL_GAME_OVER",
            ForwardGameOver
        )
    end


    function gadget:Shutdown()
        gadgetHandler:RemoveSyncAction(
            "RL_GAME_OVER"
        )
    end


    return
end


--------------------------------------------------------------------------------
-- Spring aliases
--------------------------------------------------------------------------------

local spCreateUnit           = Spring.CreateUnit
local spDestroyUnit          = Spring.DestroyUnit
local spEcho                 = Spring.Echo
local spGetGameFrame         = Spring.GetGameFrame
local spGetGaiaTeamID        = Spring.GetGaiaTeamID
local spGetGroundHeight      = Spring.GetGroundHeight
local spGetModOptions        = Spring.GetModOptions
local spGetTeamList          = Spring.GetTeamList
local spGetTeamStartPosition = Spring.GetTeamStartPosition
local spGetTeamUnits         = Spring.GetTeamUnits
local spGetUnitDefID         = Spring.GetUnitDefID


--------------------------------------------------------------------------------
-- Grenzen und Standardwerte
--------------------------------------------------------------------------------

local DEFAULT_SPACING       = 80
local MIN_SPACING           = 20
local MAX_SPACING           = 500
local MAX_UNITS_PER_TEAM    = 500
local DEFAULT_ENFORCE       = true
local DEFAULT_REMOVE_MODE   = "commanders"
local DEFAULT_REMOVE_DECOS  = false
local DEFAULT_DEATH_MODE    = "killall"

local gaiaTeamID =
    spGetGaiaTeamID()


--------------------------------------------------------------------------------
-- Laufzeitdaten
--------------------------------------------------------------------------------

-- Wird erst true, nachdem alle Startunits gespawnt und die ursprünglichen
-- Startunits entfernt wurden.
local scenarioStarted = false

-- Verhindert jede weitere Sieg-/Niederlagenmeldung.
local gameOverSent = false

local enforceAllowedUnits =
    DEFAULT_ENFORCE

local removeDecorations =
    DEFAULT_REMOVE_DECOS

local removeInitialMode =
    DEFAULT_REMOVE_MODE

-- Vorhandene BAR/Spring-Modoption.
local deathMode =
    DEFAULT_DEATH_MODE


--------------------------------------------------------------------------------
-- Teamdaten
--------------------------------------------------------------------------------

-- Sortierte Liste aller spielbaren Engine-Teams.
local playableTeamIDs = {}

-- playableTeamSet[teamID] = true
local playableTeamSet = {}

-- defeatedTeams[teamID] = true
local defeatedTeams = {}

-- lossMessageSentByTeam[teamID] = true
local lossMessageSentByTeam = {}

-- winMessageSentByTeam[teamID] = true
local winMessageSentByTeam = {}

-- Anzahl relevanter lebender Units je Team.
local aliveUnitsByTeam = {}

-- Anzahl relevanter lebender Commander je Team.
local aliveCommandersByTeam = {}


--------------------------------------------------------------------------------
-- Unitdaten
--------------------------------------------------------------------------------

-- Erlaubte UnitDefIDs je Engine-Team:
--
-- allowedDefIDsByTeam[teamID][unitDefID] = true
local allowedDefIDsByTeam = {}

-- Units, die das Gadget selbst gespawnt hat:
--
-- scenarioUnitIDs[unitID] = true
local scenarioUnitIDs = {}

-- Alle vom Szenario gezählten Units:
--
-- trackedUnits[unitID] = {
--     teamID = ...,
--     unitDefID = ...,
--     isCommander = ...
-- }
local trackedUnits = {}

local isCommanderDef = {}
local isDecorationDef = {}


--------------------------------------------------------------------------------
-- Commander und Decorations erkennen
--------------------------------------------------------------------------------

for unitDefID, unitDef in pairs(UnitDefs) do
    local customParams =
        unitDef.customParams

    if customParams then
        if customParams.iscommander then
            isCommanderDef[unitDefID] = true
        end

        if customParams.decoration then
            isDecorationDef[unitDefID] = true
        end
    end
end


--------------------------------------------------------------------------------
-- Allgemeine Hilfsfunktionen
--------------------------------------------------------------------------------

local function Trim(value)
    if value == nil then
        return nil
    end

    return tostring(value):match("^%s*(.-)%s*$")
end


local function Clamp(
    value,
    minimum,
    maximum
)
    return math.max(
        minimum,
        math.min(maximum, value)
    )
end


local function ParseBoolean(
    value,
    defaultValue
)
    if value == nil then
        return defaultValue
    end

    value =
        string.lower(Trim(value))

    if value == "1"
        or value == "true"
        or value == "yes"
        or value == "on"
    then
        return true
    end

    if value == "0"
        or value == "false"
        or value == "no"
        or value == "off"
    then
        return false
    end

    return defaultValue
end


local function Split(
    value,
    separator
)
    local result = {}

    if value == nil or value == "" then
        return result
    end

    separator =
        separator or "|"

    for part in string.gmatch(
        tostring(value),
        "[^" .. separator .. "]+"
    ) do
        part = Trim(part)

        if part and part ~= "" then
            result[#result + 1] = part
        end
    end

    return result
end


local function DestroyNoWreck(unitID)
    -- selfDestruct = false
    -- reclaimed = true
    --
    -- Dadurch entstehen weder Wrack noch reclaimbare Reste.

    spDestroyUnit(
        unitID,
        false,
        true
    )
end


local function GetPlayableTeams()
    local playableTeams = {}
    local allTeams =
        spGetTeamList()

    for i = 1, #allTeams do
        local teamID =
            allTeams[i]

        if teamID ~= gaiaTeamID then
            playableTeams[#playableTeams + 1] =
                teamID
        end
    end

    -- Dadurch ist rl_team1 immer das Team mit der kleinsten TeamID,
    -- rl_team2 das nächste Team usw.
    table.sort(playableTeams)

    return playableTeams
end


--------------------------------------------------------------------------------
-- Deathmode
--------------------------------------------------------------------------------

local function IsKillAllMode()
    return deathMode == "killall"
end


local function GetDeathReason()
    if IsKillAllMode() then
        return "all_units_destroyed"
    end

    return "commander_destroyed"
end


--------------------------------------------------------------------------------
-- Spawnliste parsen
--
-- Syntax:
--     armpw:3
--     armpw:3|armrock:2
--------------------------------------------------------------------------------

local function ParseSpawnDefinition(
    value,
    optionName
)
    local result = {}
    local entries =
        Split(value, "|")

    local totalCount = 0

    for i = 1, #entries do
        local entry =
            entries[i]

        local unitName, countString =
            entry:match(
                "^([%w_%-]+)%s*:%s*(%d+)$"
            )

        if not unitName then
            spEcho(
                "[RLStart] ERROR: invalid entry in "
                .. tostring(optionName)
                .. ": "
                .. tostring(entry)
                .. " - expected unitname:count"
            )
        else
            unitName =
                string.lower(unitName)

            local unitDef =
                UnitDefNames[unitName]

            local count =
                tonumber(countString) or 0

            if not unitDef then
                spEcho(
                    "[RLStart] ERROR: unknown UnitDefName in "
                    .. tostring(optionName)
                    .. ": "
                    .. tostring(unitName)
                )

            elseif count <= 0 then
                spEcho(
                    "[RLStart] ERROR: count must be greater than 0: "
                    .. tostring(entry)
                )

            elseif totalCount >= MAX_UNITS_PER_TEAM then
                spEcho(
                    "[RLStart] WARNING: maximum unit count reached for "
                    .. tostring(optionName)
                )

                break
            else
                local remaining =
                    MAX_UNITS_PER_TEAM - totalCount

                local acceptedCount =
                    math.min(count, remaining)

                result[#result + 1] = {
                    unitName = unitName,
                    unitDefID = unitDef.id,
                    count = acceptedCount,
                }

                totalCount =
                    totalCount + acceptedCount

                if acceptedCount < count then
                    spEcho(
                        "[RLStart] WARNING: "
                        .. tostring(optionName)
                        .. " was limited to "
                        .. tostring(MAX_UNITS_PER_TEAM)
                        .. " units"
                    )
                end
            end
        end
    end

    return result, totalCount
end


--------------------------------------------------------------------------------
-- Allowlist parsen
--
-- Syntax:
--     armpw
--     armpw|armrock|armham
--------------------------------------------------------------------------------

local function ParseAllowedDefinition(
    value,
    optionName
)
    local allowedDefIDs = {}
    local validCount = 0
    local entries =
        Split(value, "|")

    for i = 1, #entries do
        local unitName =
            string.lower(entries[i])

        local unitDef =
            UnitDefNames[unitName]

        if not unitDef then
            spEcho(
                "[RLStart] ERROR: unknown UnitDefName in "
                .. tostring(optionName)
                .. ": "
                .. tostring(unitName)
            )
        else
            if not allowedDefIDs[unitDef.id] then
                allowedDefIDs[unitDef.id] = true
                validCount = validCount + 1
            end
        end
    end

    return allowedDefIDs, validCount
end


--------------------------------------------------------------------------------
-- Falls keine Allowlist angegeben ist:
-- Spawn-Units automatisch erlauben.
--------------------------------------------------------------------------------

local function BuildAllowedFromSpawn(
    spawnDefinition
)
    local allowedDefIDs = {}

    for i = 1, #spawnDefinition do
        local entry =
            spawnDefinition[i]

        allowedDefIDs[entry.unitDefID] =
            true
    end

    return allowedDefIDs
end


--------------------------------------------------------------------------------
-- Startposition bestimmen
--------------------------------------------------------------------------------

local function GetStartPosition(
    teamID,
    teamIndex,
    teamCount
)
    local x, y, z =
        spGetTeamStartPosition(teamID)

    if not x
        or not z
        or x < 0
        or z < 0
    then
        local fraction =
            teamIndex / (teamCount + 1)

        x = Game.mapSizeX * fraction
        z = Game.mapSizeZ * 0.5
        y = spGetGroundHeight(x, z)

        spEcho(
            "[RLStart] WARNING: team "
            .. tostring(teamID)
            .. " has no valid start position; using fallback"
        )
    end

    return x, y, z
end


--------------------------------------------------------------------------------
-- Quadratische Formation erzeugen
--------------------------------------------------------------------------------

local function BuildFormationOffsets(
    unitCount,
    spacing
)
    local offsets = {}

    if unitCount <= 0 then
        return offsets
    end

    local columns =
        math.ceil(math.sqrt(unitCount))

    local rows =
        math.ceil(unitCount / columns)

    local centerColumn =
        (columns - 1) * 0.5

    local centerRow =
        (rows - 1) * 0.5

    for index = 1, unitCount do
        local zeroIndex =
            index - 1

        local column =
            zeroIndex % columns

        local row =
            math.floor(zeroIndex / columns)

        offsets[index] = {
            x = (column - centerColumn) * spacing,
            z = (row - centerRow) * spacing,
        }
    end

    return offsets
end


--------------------------------------------------------------------------------
-- Units sollen zur Kartenmitte schauen
--------------------------------------------------------------------------------

local function GetFacingTowardMapCenter(x, z)
    local deltaX =
        Game.mapSizeX * 0.5 - x

    local deltaZ =
        Game.mapSizeZ * 0.5 - z

    if math.abs(deltaX) > math.abs(deltaZ) then
        if deltaX > 0 then
            return 1
        end

        return 3
    end

    if deltaZ > 0 then
        return 2
    end

    return 0
end


--------------------------------------------------------------------------------
-- Units tracken
--------------------------------------------------------------------------------

local function TrackUnit(
    unitID,
    unitDefID,
    teamID
)
    if unitID == nil
        or unitDefID == nil
        or teamID == nil
        or teamID == gaiaTeamID
        or not playableTeamSet[teamID]
    then
        return
    end

    if trackedUnits[unitID] then
        return
    end

    local commander =
        isCommanderDef[unitDefID] == true

    trackedUnits[unitID] = {
        teamID = teamID,
        unitDefID = unitDefID,
        isCommander = commander,
    }

    aliveUnitsByTeam[teamID] =
        (aliveUnitsByTeam[teamID] or 0) + 1

    if commander then
        aliveCommandersByTeam[teamID] =
            (aliveCommandersByTeam[teamID] or 0) + 1
    end

    spEcho(
        string.format(
            "[RLStart] tracked unit: "
            .. "unitID=%d teamID=%d unitDefID=%d "
            .. "commander=%s aliveUnits=%d aliveCommanders=%d",
            unitID,
            teamID,
            unitDefID,
            tostring(commander),
            aliveUnitsByTeam[teamID] or 0,
            aliveCommandersByTeam[teamID] or 0
        )
    )
end


--------------------------------------------------------------------------------
-- Terminalnachricht senden
--------------------------------------------------------------------------------

local function SendResultToTeam(
    receivingTeamID,
    result,
    winnerTeamID,
    eliminatedTeamID,
    frame,
    reason,
    remainingTeams
)
    local message =
        string.format(
            "UNBARABLE_GAME_OVER"
            .. ";result=%s"
            .. ";team=%d"
            .. ";winner=%d"
            .. ";eliminated=%d"
            .. ";frame=%d"
            .. ";mode=%s"
            .. ";reason=%s"
            .. ";remaining=%d",
            result,
            receivingTeamID,
            winnerTeamID or -1,
            eliminatedTeamID or -1,
            frame,
            deathMode,
            reason,
            remainingTeams or 0
        )

    SendToUnsynced(
        "RL_GAME_OVER",
        receivingTeamID,
        message
    )

    spEcho(
        string.format(
            "[RLStart] queued terminal message: "
            .. "teamID=%d result=%s message=%s",
            receivingTeamID,
            result,
            message
        )
    )
end


--------------------------------------------------------------------------------
-- Noch lebende Teams bestimmen
--------------------------------------------------------------------------------

local function GetRemainingTeams()
    local remainingTeams = {}

    for i = 1, #playableTeamIDs do
        local teamID =
            playableTeamIDs[i]

        if not defeatedTeams[teamID] then
            remainingTeams[#remainingTeams + 1] =
                teamID
        end
    end

    return remainingTeams
end


--------------------------------------------------------------------------------
-- Ein Team ausscheiden lassen
--------------------------------------------------------------------------------

local function EliminateTeam(
    eliminatedTeamID,
    reason
)
    if gameOverSent
        or defeatedTeams[eliminatedTeamID]
        or not playableTeamSet[eliminatedTeamID]
    then
        return
    end

    defeatedTeams[eliminatedTeamID] =
        true

    local frame =
        spGetGameFrame()

    local remainingTeams =
        GetRemainingTeams()

    spEcho(
        string.format(
            "[RLStart] TEAM ELIMINATED: "
            .. "teamID=%d frame=%d mode=%s reason=%s "
            .. "remainingTeams=%d",
            eliminatedTeamID,
            frame,
            deathMode,
            reason,
            #remainingTeams
        )
    )

    --------------------------------------------------------------------------
    -- Das ausgeschiedene Team bekommt sofort eine Loss-Nachricht.
    --------------------------------------------------------------------------

    if not lossMessageSentByTeam[eliminatedTeamID] then
        lossMessageSentByTeam[eliminatedTeamID] =
            true

        local currentWinner =
            #remainingTeams == 1
            and remainingTeams[1]
            or -1

        SendResultToTeam(
            eliminatedTeamID,
            "loss",
            currentWinner,
            eliminatedTeamID,
            frame,
            reason,
            #remainingTeams
        )
    end

    --------------------------------------------------------------------------
    -- Bei mehr als einem verbleibenden Team läuft das Match weiter.
    --------------------------------------------------------------------------

    if #remainingTeams > 1 then
        return
    end

    --------------------------------------------------------------------------
    -- Genau ein Team lebt noch: dieses Team hat gewonnen.
    --------------------------------------------------------------------------

    if #remainingTeams == 1 then
        local winnerTeamID =
            remainingTeams[1]

        gameOverSent = true

        if not winMessageSentByTeam[winnerTeamID] then
            winMessageSentByTeam[winnerTeamID] =
                true

            SendResultToTeam(
                winnerTeamID,
                "win",
                winnerTeamID,
                eliminatedTeamID,
                frame,
                reason,
                1
            )
        end

        spEcho(
            string.format(
                "[RLStart] GAME OVER: "
                .. "winnerTeamID=%d lastEliminatedTeamID=%d "
                .. "frame=%d mode=%s reason=%s",
                winnerTeamID,
                eliminatedTeamID,
                frame,
                deathMode,
                reason
            )
        )

        return
    end

    --------------------------------------------------------------------------
    -- Kein Team mehr übrig: Draw.
    --
    -- Das kann beispielsweise bei gleichzeitiger Zerstörung im selben
    -- Simulationsframe auftreten. Bereits ausgeschiedene Teams haben ihre
    -- Loss-Nachricht schon erhalten. Falls nötig, kann C++ "draw" zusätzlich
    -- unterstützen. Standardmäßig wird hier lediglich geloggt.
    --------------------------------------------------------------------------

    gameOverSent = true

    spEcho(
        string.format(
            "[RLStart] GAME OVER: draw, no teams remaining "
            .. "frame=%d mode=%s reason=%s",
            frame,
            deathMode,
            reason
        )
    )
end


--------------------------------------------------------------------------------
-- Ein Team spawnen
--------------------------------------------------------------------------------

local function SpawnTeam(
    teamID,
    teamIndex,
    teamCount,
    spawnDefinition,
    totalUnitCount,
    spacing
)
    local startX, _, startZ =
        GetStartPosition(
            teamID,
            teamIndex,
            teamCount
        )

    local offsets =
        BuildFormationOffsets(
            totalUnitCount,
            spacing
        )

    local facing =
        GetFacingTowardMapCenter(
            startX,
            startZ
        )

    local offsetIndex = 1
    local spawnedCount = 0

    for groupIndex = 1, #spawnDefinition do
        local group =
            spawnDefinition[groupIndex]

        for _ = 1, group.count do
            local offset =
                offsets[offsetIndex]

            if not offset then
                break
            end

            local spawnX =
                Clamp(
                    startX + offset.x,
                    16,
                    Game.mapSizeX - 16
                )

            local spawnZ =
                Clamp(
                    startZ + offset.z,
                    16,
                    Game.mapSizeZ - 16
                )

            local spawnY =
                spGetGroundHeight(
                    spawnX,
                    spawnZ
                )

            local unitID =
                spCreateUnit(
                    group.unitDefID,
                    spawnX,
                    spawnY,
                    spawnZ,
                    facing,
                    teamID
                )

            if unitID then
                scenarioUnitIDs[unitID] =
                    true

                TrackUnit(
                    unitID,
                    group.unitDefID,
                    teamID
                )

                spawnedCount =
                    spawnedCount + 1
            else
                spEcho(
                    string.format(
                        "[RLStart] ERROR: CreateUnit failed: "
                        .. "teamID=%d unit=%s position=(%.1f, %.1f)",
                        teamID,
                        group.unitName,
                        spawnX,
                        spawnZ
                    )
                )
            end

            offsetIndex =
                offsetIndex + 1
        end
    end

    return spawnedCount
end


--------------------------------------------------------------------------------
-- Ursprüngliche BAR-Startunits entfernen
--
-- rl_remove_initial:
--
-- none
--     Nichts entfernen.
--
-- commanders
--     Nur Commander entfernen.
--
-- all
--     Alle Units entfernen, die nicht von diesem Gadget gespawnt wurden.
--------------------------------------------------------------------------------

local function RemoveOriginalStartUnits(teamID)
    if removeInitialMode == "none" then
        return
    end

    local teamUnits =
        spGetTeamUnits(teamID)

    for i = 1, #teamUnits do
        local unitID =
            teamUnits[i]

        -- Eigene Szenario-Units niemals entfernen.
        if not scenarioUnitIDs[unitID] then
            local unitDefID =
                spGetUnitDefID(unitID)

            if unitDefID then
                if removeInitialMode == "all" then
                    DestroyNoWreck(unitID)

                elseif removeInitialMode == "commanders" then
                    if isCommanderDef[unitDefID] then
                        DestroyNoWreck(unitID)

                    elseif removeDecorations
                        and isDecorationDef[unitDefID]
                    then
                        DestroyNoWreck(unitID)
                    end
                end
            end
        end
    end
end


--------------------------------------------------------------------------------
-- GameStart
--------------------------------------------------------------------------------

function gadget:GameStart()
    local modOptions =
        spGetModOptions() or {}

    playableTeamIDs =
        GetPlayableTeams()

    if #playableTeamIDs < 2 then
        spEcho(
            "[RLStart] ERROR: at least two playable teams are required"
        )

        gadgetHandler:RemoveGadget(self)
        return
    end

    --------------------------------------------------------------------------
    -- Laufzeitdaten zurücksetzen.
    --------------------------------------------------------------------------

    gameOverSent = false
    scenarioStarted = false

    playableTeamSet = {}
    defeatedTeams = {}
    lossMessageSentByTeam = {}
    winMessageSentByTeam = {}

    aliveUnitsByTeam = {}
    aliveCommandersByTeam = {}

    allowedDefIDsByTeam = {}
    scenarioUnitIDs = {}
    trackedUnits = {}

    for i = 1, #playableTeamIDs do
        local teamID =
            playableTeamIDs[i]

        playableTeamSet[teamID] = true
        aliveUnitsByTeam[teamID] = 0
        aliveCommandersByTeam[teamID] = 0
    end

    --------------------------------------------------------------------------
    -- Vorhandene deathmode-Modoption verwenden.
    --------------------------------------------------------------------------

    deathMode =
        string.lower(
            Trim(modOptions.deathmode)
            or DEFAULT_DEATH_MODE
        )

    if deathMode == "" then
        deathMode =
            DEFAULT_DEATH_MODE
    end

    if IsKillAllMode() then
        spEcho(
            "[RLStart] deathmode=killall: "
            .. "team loses after its last tracked unit is destroyed"
        )
    else
        spEcho(
            "[RLStart] deathmode="
            .. tostring(deathMode)
            .. ": team loses when a tracked commander is destroyed"
        )
    end

    --------------------------------------------------------------------------
    -- Allgemeine Optionen.
    --------------------------------------------------------------------------

    local spacing =
        tonumber(modOptions.rl_spawn_spacing)
        or DEFAULT_SPACING

    spacing =
        Clamp(
            spacing,
            MIN_SPACING,
            MAX_SPACING
        )

    enforceAllowedUnits =
        ParseBoolean(
            modOptions.rl_enforce_allowed,
            DEFAULT_ENFORCE
        )

    removeDecorations =
        ParseBoolean(
            modOptions.rl_remove_decorations,
            DEFAULT_REMOVE_DECOS
        )

    removeInitialMode =
        string.lower(
            Trim(modOptions.rl_remove_initial)
            or DEFAULT_REMOVE_MODE
        )

    if removeInitialMode ~= "none"
        and removeInitialMode ~= "commanders"
        and removeInitialMode ~= "all"
    then
        spEcho(
            "[RLStart] WARNING: invalid rl_remove_initial='"
            .. tostring(removeInitialMode)
            .. "'; using 'commanders'"
        )

        removeInitialMode =
            "commanders"
    end

    --------------------------------------------------------------------------
    -- Konfiguration aller Teams lesen und validieren.
    --------------------------------------------------------------------------

    local teamConfigurations = {}
    local configurationValid = true

    for teamIndex = 1, #playableTeamIDs do
        local teamID =
            playableTeamIDs[teamIndex]

        local spawnOptionName =
            "rl_team"
            .. teamIndex
            .. "_spawn"

        local allowedOptionName =
            "rl_team"
            .. teamIndex
            .. "_allowed"

        local spawnValue =
            modOptions[spawnOptionName]

        if not spawnValue
            or Trim(spawnValue) == ""
        then
            spEcho(
                "[RLStart] ERROR: missing required option "
                .. spawnOptionName
            )

            configurationValid =
                false
        else
            local spawnDefinition, totalUnitCount =
                ParseSpawnDefinition(
                    spawnValue,
                    spawnOptionName
                )

            if totalUnitCount <= 0 then
                spEcho(
                    "[RLStart] ERROR: "
                    .. spawnOptionName
                    .. " contains no valid units"
                )

                configurationValid =
                    false
            end

            local allowedDefIDs
            local allowedValue =
                modOptions[allowedOptionName]

            if allowedValue
                and Trim(allowedValue) ~= ""
            then
                local allowedCount

                allowedDefIDs, allowedCount =
                    ParseAllowedDefinition(
                        allowedValue,
                        allowedOptionName
                    )

                if enforceAllowedUnits
                    and allowedCount <= 0
                then
                    spEcho(
                        "[RLStart] ERROR: "
                        .. allowedOptionName
                        .. " contains no valid units"
                    )

                    configurationValid =
                        false
                end
            else
                allowedDefIDs =
                    BuildAllowedFromSpawn(
                        spawnDefinition
                    )

                spEcho(
                    "[RLStart] INFO: "
                    .. allowedOptionName
                    .. " not set; using spawn units as allowlist"
                )
            end

            teamConfigurations[teamIndex] = {
                teamID = teamID,
                spawn = spawnDefinition,
                totalUnitCount = totalUnitCount,
                allowedDefIDs = allowedDefIDs,
            }
        end
    end

    --------------------------------------------------------------------------
    -- Bei ungültiger Konfiguration nichts verändern.
    --------------------------------------------------------------------------

    if not configurationValid then
        spEcho(
            "[RLStart] ERROR: invalid configuration; "
            .. "scenario start aborted"
        )

        gadgetHandler:RemoveGadget(self)
        return
    end

    --------------------------------------------------------------------------
    -- Allowlist registrieren.
    --------------------------------------------------------------------------

    for teamIndex = 1, #teamConfigurations do
        local config =
            teamConfigurations[teamIndex]

        allowedDefIDsByTeam[config.teamID] =
            config.allowedDefIDs
    end

    --------------------------------------------------------------------------
    -- Zuerst alle Szenario-Units spawnen.
    --------------------------------------------------------------------------

    local totalSpawned = 0

    for teamIndex = 1, #teamConfigurations do
        local config =
            teamConfigurations[teamIndex]

        local spawnedCount =
            SpawnTeam(
                config.teamID,
                teamIndex,
                #teamConfigurations,
                config.spawn,
                config.totalUnitCount,
                spacing
            )

        totalSpawned =
            totalSpawned + spawnedCount

        spEcho(
            string.format(
                "[RLStart] teamIndex=%d teamID=%d "
                .. "requested=%d spawned=%d",
                teamIndex,
                config.teamID,
                config.totalUnitCount,
                spawnedCount
            )
        )
    end

    --------------------------------------------------------------------------
    -- Jedes Team muss mindestens eine Unit erhalten haben.
    --------------------------------------------------------------------------

    for i = 1, #playableTeamIDs do
        local teamID =
            playableTeamIDs[i]

        if (aliveUnitsByTeam[teamID] or 0) <= 0 then
            spEcho(
                string.format(
                    "[RLStart] ERROR: team %d has no spawned units",
                    teamID
                )
            )

            configurationValid =
                false
        end
    end

    if totalSpawned <= 0
        or not configurationValid
    then
        spEcho(
            "[RLStart] ERROR: invalid spawned unit state; "
            .. "original start units will not be removed"
        )

        gadgetHandler:RemoveGadget(self)
        return
    end

    --------------------------------------------------------------------------
    -- Für Nicht-killall werden Commander benötigt.
    --------------------------------------------------------------------------

    if not IsKillAllMode() then
        for i = 1, #playableTeamIDs do
            local teamID =
                playableTeamIDs[i]

            if (aliveCommandersByTeam[teamID] or 0) <= 0 then
                spEcho(
                    string.format(
                        "[RLStart] ERROR: deathmode='%s' requires "
                        .. "at least one commander in rl_team%d_spawn "
                        .. "for engine team %d",
                        deathMode,
                        i,
                        teamID
                    )
                )

                configurationValid =
                    false
            end
        end
    end

    if not configurationValid then
        spEcho(
            "[RLStart] ERROR: deathmode configuration invalid; "
            .. "scenario start aborted"
        )

        gadgetHandler:RemoveGadget(self)
        return
    end

    --------------------------------------------------------------------------
    -- Erst jetzt ursprüngliche Startunits entfernen.
    --
    -- scenarioStarted ist noch false. Dadurch zählen die absichtlich
    -- entfernten BAR-Startunits nicht als Szenario-Todesfälle.
    --------------------------------------------------------------------------

    for teamIndex = 1, #teamConfigurations do
        RemoveOriginalStartUnits(
            teamConfigurations[teamIndex].teamID
        )
    end

    --------------------------------------------------------------------------
    -- Ab jetzt neue Units und Todesfälle überwachen.
    --------------------------------------------------------------------------

    scenarioStarted = true

    spEcho(
        string.format(
            "[RLStart] OK: teams=%d totalSpawned=%d "
            .. "spacing=%d enforceAllowed=%s "
            .. "removeInitial=%s deathmode=%s",
            #teamConfigurations,
            totalSpawned,
            spacing,
            tostring(enforceAllowedUnits),
            tostring(removeInitialMode),
            deathMode
        )
    )
end


--------------------------------------------------------------------------------
-- Später erzeugte Units kontrollieren
--------------------------------------------------------------------------------

function gadget:UnitCreated(
    unitID,
    unitDefID,
    unitTeam
)
    if not scenarioStarted
        or gameOverSent
    then
        return
    end

    if unitTeam == gaiaTeamID
        or not playableTeamSet[unitTeam]
    then
        return
    end

    --------------------------------------------------------------------------
    -- Später erzeugte Commander blockieren.
    --
    -- Die über rl_teamX_spawn erzeugten Commander sind bereits vorhanden,
    -- bevor scenarioStarted=true gesetzt wird, und werden deshalb hiervon
    -- nicht blockiert.
    --------------------------------------------------------------------------

    if isCommanderDef[unitDefID] then
        spEcho(
            "[RLStart] blocked additional commander: unitID="
            .. tostring(unitID)
        )

        DestroyNoWreck(unitID)
        return
    end

    if removeDecorations
        and isDecorationDef[unitDefID]
    then
        spEcho(
            "[RLStart] blocked decoration: unitID="
            .. tostring(unitID)
        )

        DestroyNoWreck(unitID)
        return
    end

    --------------------------------------------------------------------------
    -- Allowlist anwenden.
    --------------------------------------------------------------------------

    if enforceAllowedUnits then
        local allowedDefIDs =
            allowedDefIDsByTeam[unitTeam]

        if not allowedDefIDs
            or not allowedDefIDs[unitDefID]
        then
            local unitDef =
                UnitDefs[unitDefID]

            local unitName =
                unitDef
                and unitDef.name
                or tostring(unitDefID)

            spEcho(
                string.format(
                    "[RLStart] blocked unit: "
                    .. "teamID=%d unit=%s unitID=%d",
                    unitTeam,
                    tostring(unitName),
                    unitID
                )
            )

            DestroyNoWreck(unitID)
            return
        end
    end

    --------------------------------------------------------------------------
    -- Erlaubte Unit zählen.
    --------------------------------------------------------------------------

    TrackUnit(
        unitID,
        unitDefID,
        unitTeam
    )
end


--------------------------------------------------------------------------------
-- Unit zerstört
--------------------------------------------------------------------------------

function gadget:UnitDestroyed(
    unitID,
    unitDefID,
    unitTeam,
    attackerID,
    attackerDefID,
    attackerTeam
)
    if not scenarioStarted
        or gameOverSent
    then
        return
    end

    local tracked =
        trackedUnits[unitID]

    -- Nicht vom Szenario getrackte Units beeinflussen die Siegbedingung nicht.
    if not tracked then
        return
    end

    --------------------------------------------------------------------------
    -- Sofort aus dem Tracking entfernen.
    --
    -- Damit kann derselbe Todesfall niemals doppelt gezählt werden.
    --------------------------------------------------------------------------

    trackedUnits[unitID] = nil
    scenarioUnitIDs[unitID] = nil

    local teamID =
        tracked.teamID

    aliveUnitsByTeam[teamID] =
        math.max(
            0,
            (aliveUnitsByTeam[teamID] or 0) - 1
        )

    if tracked.isCommander then
        aliveCommandersByTeam[teamID] =
            math.max(
                0,
                (aliveCommandersByTeam[teamID] or 0) - 1
            )
    end

    spEcho(
        string.format(
            "[RLStart] UnitDestroyed: "
            .. "unitID=%d teamID=%d commander=%s "
            .. "aliveUnits=%d aliveCommanders=%d "
            .. "attackerID=%s attackerTeam=%s frame=%d",
            unitID,
            teamID,
            tostring(tracked.isCommander),
            aliveUnitsByTeam[teamID] or 0,
            aliveCommandersByTeam[teamID] or 0,
            tostring(attackerID),
            tostring(attackerTeam),
            spGetGameFrame()
        )
    )

    --------------------------------------------------------------------------
    -- killall:
    -- Team verliert nach dem Tod seiner letzten getrackten Unit.
    --------------------------------------------------------------------------

    if IsKillAllMode() then
        if aliveUnitsByTeam[teamID] <= 0 then
            EliminateTeam(
                teamID,
                "all_units_destroyed"
            )
        end

        return
    end

    --------------------------------------------------------------------------
    -- Andere deathmode-Werte:
    -- Team verliert unmittelbar beim Tod eines Commanders.
    --
    -- Das entspricht der gewünschten Semantik:
    -- "Wenn Commander stirbt, ist das Spiel für das Team vorbei."
    --------------------------------------------------------------------------

    if tracked.isCommander then
        EliminateTeam(
            teamID,
            "commander_destroyed"
        )
    end
end