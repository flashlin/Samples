using System.ComponentModel;

namespace WCodeSnippetX.Models;

public class GlobalKeyboardHookEventArgs : HandledEventArgs
{
	public GlobalKeyboardHook.KeyboardState KeyboardState { get; private set; }
	public GlobalKeyboardHook.LowLevelKeyboardInputEvent KeyboardData { get; private set; }

	public GlobalKeyboardHookEventArgs(
		GlobalKeyboardHook.LowLevelKeyboardInputEvent keyboardData,
		GlobalKeyboardHook.KeyboardState keyboardState)
	{
		KeyboardData = keyboardData;
		KeyboardState = keyboardState;
	}
}