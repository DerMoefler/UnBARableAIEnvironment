local widget = widget ---@type Widget

function widget:GetInfo()
    return {
        name = "UnBARable Disable Human UI",
        desc = "Removes build and faction-selection UI",
        author = "UnBARableAI",
        date = "2026",
        license = "GNU GPL, v2 or later",
        layer = 100000,
        enabled = true,
        handler = true,
    }
end


local targetNames = {
    -- build menus
    "Build menu",
    "Grid menu",
    "Pregame Queue",

    -- to pick a faction
    "Factionpicker",
}


local removed = {}


local function removeTarget(name)
    if removed[name] then
        return
    end

    local targetWidget =
        widgetHandler:FindWidget(name)

    if not targetWidget then
        return
    end

    Spring.Echo(
        "[UnBARable] Removing active widget: "
        .. name
    )

    widgetHandler:RemoveWidgetRaw(
        targetWidget
    )

    removed[name] = true

    Spring.Echo(
        "[UnBARable] Removed widget: "
        .. name
    )
end


local function allTargetsRemoved()
    for _, name in ipairs(targetNames) do
        if not removed[name] then
            return false
        end
    end

    return true
end


function widget:Initialize()
    Spring.Echo(
        "[UnBARable] Disable Human UI loaded"
    )
end


function widget:Update(dt)
    for _, name in ipairs(targetNames) do
        removeTarget(name)
    end

    if allTargetsRemoved() then
        Spring.Echo(
            "[UnBARable] Target UI widgets removed"
        )

        widgetHandler:RemoveWidgetCallIn(
            "Update",
            self
        )
    end
end