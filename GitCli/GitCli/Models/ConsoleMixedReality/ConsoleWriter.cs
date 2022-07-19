using System.Net.Mime;
using System.Reactive.Linq;
using System.Reactive.Subjects;
using System.Text;
using Microsoft.EntityFrameworkCore.Metadata.Internal;

namespace GitCli.Models.ConsoleMixedReality;

public class ConsoleWriter : IConsoleWriter
{
	private readonly ConsoleColor _originBackgroundColor;
	private readonly ConsoleColor _originForegroundColor;
	private ConsoleColor _foregroundColor;
	private ConsoleColor _backgroundColor;

	public ConsoleWriter()
	{
		_originBackgroundColor = Console.BackgroundColor;
		_backgroundColor = _originBackgroundColor;
		_originForegroundColor = Console.ForegroundColor;
		_foregroundColor = _originForegroundColor;
	}

	public bool IsInsertMode { get; set; } = true;

	public ConsoleInputObserver KeyEvents { get; } = new ConsoleInputObserver();

	public Size GetSize()
	{
		return new Size()
		{
			Width = Console.WindowWidth,
			Height = Console.WindowHeight,
		};
	}

	public void SetForegroundColor(ConsoleColor color)
	{
		_foregroundColor = color;
		Console.ForegroundColor = color;
	}

	public void SetBackgroundColor(ConsoleColor color)
	{
		_backgroundColor = color;
		Console.BackgroundColor = color;
	}

	public void Write(Position position, Character character)
	{
		if (character.IsEmpty)
		{
			WriteCharacter(' ', Color.White, Color.Black);
			return;
		}
		
		var content = character.Content;
		var foreground = character.Foreground;
		var background = character.Background;
		if (content == '\n') content = ' ';

		Console.SetCursorPosition(position.X, position.Y);
		WriteCharacter(content, foreground, background);
	}

	private static void WriteCharacter(char? character,Color foreground, Color background)
	{
		Console.Write(
			$"\x1b[38;2;{foreground.Red};{foreground.Green};{foreground.Blue}m\x1b[48;2;{background.Red};{background.Green};{background.Blue}m{character}");
	}

	public void ResetWriteColor()
	{
		Console.BackgroundColor = _backgroundColor;
		Console.ForegroundColor = _foregroundColor;
	}

	public void ResetColor()
	{
		Console.BackgroundColor = _originBackgroundColor;
		Console.ForegroundColor = _originForegroundColor;
	}

	public void Initialize()
	{
		SetUtf8();
		//HideCursor();
		Clear();
	}

	public void Clear()
	{
		Console.ForegroundColor = ConsoleColor.White;
		Console.BackgroundColor = ConsoleColor.Black;
		Console.Clear();
	}

	private static void SetUtf8()
	{
		Console.OutputEncoding = Encoding.UTF8;
	}

	public void HideCursor()
	{
		Console.CursorVisible = false;
	}

	public void ShowCursor()
	{
		Console.CursorVisible = true;
	}

	public InputEvent ReadKey()
	{
		var key = Console.ReadKey(true);
		return InputEvent.From(key);
	}

	public void SetCursorPosition(Position position)
	{
		if (position.IsEmpty)
		{
			return;
		}
		Console.SetCursorPosition(position.X, position.Y);
		SetVirtualCursorPosition(position.X, position.Y);
	}

	private void SetVirtualCursorPosition(int col, int row)
	{
		//Console.Write($"\x1b[{row + 1};{col + 1}H");
		HideCursor();
		Console.CursorSize = 50;
		Console.SetCursorPosition(col, row);
	}
}