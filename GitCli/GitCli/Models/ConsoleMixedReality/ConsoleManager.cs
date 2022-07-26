namespace GitCli.Models.ConsoleMixedReality;

public class ConsoleManager : IConsoleManager
{
	private static readonly ConsoleBuffer Buffer = new ConsoleBuffer();
	private static FreezeLock _freezeLock = new FreezeLock();
	private readonly CancellationTokenSource _cancellationTokenSource = new();
	private readonly IConsoleWriter _console;
	private readonly ConsoleInputObserver _inputObserver = new ConsoleInputObserver();
	private IConsoleElement _focusedElement = EmptyElement.Default;

	public ConsoleManager(IConsoleWriter console)
	{
		_console = console;
		Content = EmptyElement.Default;
	}

	public Size BufferSize => Buffer.Size;
	public IConsoleWriter Console => _console;
	public IConsoleElement Content { get; set; }

	public IConsoleElement FocusedElement
	{
		get => _focusedElement;
		set => SetFocusElement(value);
	}

	public bool FirstSetFocusElement(IConsoleElement element)
	{
		if (_focusedElement != EmptyElement.Default)
		{
			return false;
		}
		return SetFocusElement(element);
	}

	private bool SetFocusElement(IConsoleElement element)
	{
		var lastFocused = element.GetLeafChild();
		var hasChanged = _focusedElement != lastFocused;
		_focusedElement = lastFocused;
		return hasChanged;
	}

	public Color HighlightBackgroundColor1 { get; set; } = ConsoleColor.Gray;
	public Color HighlightBackgroundColor2 { get; set; } = ConsoleColor.DarkGray;
	public Color InputBackgroundColor { get; set; } = ConsoleColor.DarkBlue;
	public ConsoleInputObserver InputObserver => _inputObserver;
	public Color ViewBackgroundColor { get; set; } = ConsoleColor.DarkYellow;
	public Size WindowSize => _console.GetSize();
	public void AdjustBufferSize()
	{
		if (WindowSize != BufferSize)
		{
			Resize(WindowSize);
		}
	}

	public void AdjustWindowSize()
	{
		if (WindowSize != BufferSize)
		{
			Resize(BufferSize);
		}
	}

	public void Resize(Size size)
	{
		Buffer.Initialize(size);
		Initialize();
	}

	public void Start()
	{
		Content.OnCreate(Rect.Empty, (IConsoleManager)this);
		AdjustBufferSize();
		while (!_cancellationTokenSource.IsCancellationRequested)
		{
			_console.SetCursorPosition(Content.CursorPosition);
			ProcessInputEvent(_console.ReadKey());
		}
	}

	private void Initialize()
	{
		_console.Initialize();
		Buffer.Clear();
		_freezeLock.Freeze();
		_freezeLock.Unfreeze();
		Redraw();
	}

	private void ProcessInputEvent(InputEvent @event)
	{
		if (@event.HasControl && @event.Key == ConsoleKey.X)
		{
			_cancellationTokenSource.Cancel();
			_console.ResetColor();
			return;
		}

		var isHandled = Content.OnInput(@event);
		if (isHandled)
		{
			Redraw();
		}
	}

	private void Redraw()
	{
		var contentRect = Content.ViewRect;
		//contentRect = contentRect.ExtendBy(contentRect.BottomRightCorner.Next);
		Update(contentRect);
	}

	private void Update(Rect rect)
	{
		rect = rect.Intersect(Rect.OfSize(BufferSize));
		rect = rect.Intersect(Rect.OfSize(WindowSize));

		for (int y = rect.Top; y <= rect.Bottom; y++)
		{
			for (int x = rect.Left; x <= rect.Right; x++)
			{
				var position = new Position(x, y);
				var character = Content[position];
				//_drawBuffer.Update(position, character);
				if (!Buffer.Update(position, character)) continue;
				_console.Write(position, character);

				//character = _drawBuffer[position];
				//_console.Write(position, character);
			}
		}
	}

	public void SetFocusElementOrChild(IConsoleElement element, IConsoleElement child)
	{
		if (FocusedElement == element)
		{
			FocusedElement = child;
		}
	}
}

