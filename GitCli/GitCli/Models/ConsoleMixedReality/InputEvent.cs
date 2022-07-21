namespace GitCli.Models.ConsoleMixedReality;

public class InputEvent
{
	public static InputEvent Empty = new();

	public bool HasControl { get; set; }
	public bool HasAlt { get; set; }
	public bool HasShift { get; set; }
	public ConsoleKey Key { get; set; }
	public bool Handled { get; set; }
	public char KeyChar { get; set; }

	public static InputEvent From(ConsoleKeyInfo key)
	{
		return new InputEvent
		{
			HasControl = key.Modifiers.HasFlag(ConsoleModifiers.Control),
			HasAlt = key.Modifiers.HasFlag(ConsoleModifiers.Alt),
			HasShift = key.Modifiers.HasFlag(ConsoleModifiers.Shift),
			Key = key.Key,
			KeyChar = key.KeyChar,
		};
	}
}