namespace GitCli.Models.ConsoleMixedReality;

public class ConsoleManager
{
    private static readonly ConsoleBuffer _buffer = new ConsoleBuffer();
    private static FreezeLock _freezeLock = new FreezeLock();
    private readonly IConsoleWriter _console;

    public ConsoleManager(IConsoleWriter console)
    {
        _console = console;
        Content = new EmptyElement(console);
        _console.KeyEvents.Subscribe(ReadInput);
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
        Update(Content.GetViewRect());
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

                if (!_buffer.Update(position, character)) continue;
                _console.Write(position, character);
            }
        }
    }

    public void Resize(Size size)
    {
        _buffer.Initialize(size);
        Initialize();
    }

    public void AdjustBufferSize()
    {
        if (WindowSize != BufferSize)
            Resize(WindowSize);
    }

    public void AdjustWindowSize()
    {
        if (WindowSize != BufferSize)
            Resize(BufferSize);
    }

    public void ReadInput(InputEvent @event)
    {
        Content.OnInput(@event);
    }
}