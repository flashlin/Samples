using System;
using System.Reactive.Disposables;
using System.Threading;

namespace T1.ConsoleUiMixedReality;

public class ConsoleInputObserver : IObservable<ConsoleElementEvent>
{
    private IObserver<ConsoleElementEvent>? _observer;

    public IDisposable Subscribe(IObserver<ConsoleElementEvent> observer)
    {
        _observer = observer;
        return Disposable.Empty;
    }

    public CancellationTokenSource Cancellation { get; } = new();

    private void RaiseKey(IConsoleElement element, InputEvent inputEvent)
    {
        _observer?.OnNext(new ConsoleElementEvent
        {
            InputEvent = inputEvent,
            Element = element,
        });
    }
}
