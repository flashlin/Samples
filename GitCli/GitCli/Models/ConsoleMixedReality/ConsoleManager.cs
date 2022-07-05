namespace GitCli.Models.ConsoleMixedReality;

public class ConsoleManager
{
	private static readonly ConsoleBuffer _buffer = new ConsoleBuffer();
	private static FreezeLock _freezeLock = new FreezeLock();
	private readonly IConsoleWriter _console;
    private readonly CancellationTokenSource _cancellationTokenSource = new();
    
	public ConsoleManager(IConsoleWriter console)
	{
		_console = console;
		Content = new EmptyElement();
	}

	public IConsoleElement Content { get; set; }

	public Size WindowSize => _console.GetSize();
	public Size BufferSize => _buffer.Size;

	private void Initialize()
	{
		_console.Initialize();
		_buffer.Clear();
		_freezeLock.Freeze();
		_freezeLock.Unfreeze();
		Redraw();
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
				if (!_buffer.Update(position, character)) continue;
				_console.Write(position, character);
				
				//character = _drawBuffer[position];
				//_console.Write(position, character);
			}
		}


		//for (int y = rect.Top; y <= rect.Bottom; y++)
		//{
		//	for (int x = rect.Left; x <= rect.Right; x++)
		//	{
		//		var position = new Position(x, y);
		//		var character = _drawBuffer[position];
		//		if (!_buffer.Update(position, character)) continue;
		//		_console.Write(position, character);
		//	}
		//}
	}

	public void Resize(Size size)
	{
		_buffer.Initialize(size);
		Initialize();
	}

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

	public void Start()
	{
		//var task = Task.Run(() =>
		//{
		//	while (true)
		//	{
		//		Write(new Position { X = 0, Y = 0 }, $"{DateTime.Now}");
		//		Redraw();
		//		Thread.Sleep(500);
		//	}
		//});

		Content.OnCreated(_console);
		AdjustBufferSize();
		while (!_cancellationTokenSource.IsCancellationRequested)
		{
			_console.SetCursorPosition(Content.CursorPosition);
			ProcessInputEvent(_console.ReadKey());
		}
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
}