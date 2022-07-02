using System.Reactive.Disposables;

namespace GitCli.Models.ConsoleMixedReality;

public class ConsoleInputObserver : IObservable<InputEvent>
{
	private IObserver<InputEvent>? _observer;
	private Task? _task;
	private readonly CancellationTokenSource _cancellationTokenSource = new();

	public IDisposable Subscribe(IObserver<InputEvent> observer)
	{
		_observer = observer;
		return Disposable.Empty;
	}

	public CancellationTokenSource Cancellation => _cancellationTokenSource;

	public void StartReadKey()
	{
		if (_task != null)
		{
			throw new InvalidProgramException();
		}
		_task = new Task(ReadKeyWithoutWait);
		_task.Start();
	}

	private void ReadKeyWithoutWait()
	{
		while (!_cancellationTokenSource.IsCancellationRequested)
		{
			// if (Console.KeyAvailable)
			// {
			// 	var key = Console.ReadKey(true);
			// 	RaiseKey(key);
			// }
			//Thread.Sleep(10);
			var key = Console.ReadKey(true);
			RaiseKey(key);
		}
	}

	private void RaiseKey(ConsoleKeyInfo key)
	{
		var inputEvent = new InputEvent
		{
			HasControl = key.Modifiers.HasFlag(ConsoleModifiers.Control),
			HasAlt = key.Modifiers.HasFlag(ConsoleModifiers.Alt),
			HasShift = key.Modifiers.HasFlag(ConsoleModifiers.Shift),
			Key = key.Key,
			KeyChar = key.KeyChar,
		};
		_observer?.OnNext(inputEvent);
	}
}