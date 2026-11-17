#!/usr/bin/osascript

# Required parameters:
# @raycast.schemaVersion 1
# @raycast.title Unminimize Window
# @raycast.mode silent

tell application "System Events"
	set frontApp to first application process whose frontmost is true
	tell frontApp
		repeat with w in windows
			try
				if value of attribute "AXMinimized" of w is true then
					set value of attribute "AXMinimized" of w to false
					perform action "AXRaise" of w
				end if
			end try
		end repeat
	end tell
end tell
