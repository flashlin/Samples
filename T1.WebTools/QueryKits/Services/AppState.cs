using Prism.Events;

namespace QueryKits.Services;

public class AppState : IAppState
{
    private readonly IEventAggregator _eventAggregator;

    public AppState(IEventAggregator eventAggregator)
    {
        _eventAggregator = eventAggregator;
    }

    public bool IsLoading { get; set; }

    public void Publish(Action<AppState> changeFn)
    {
        changeFn(this);
        _eventAggregator.GetEvent<UpdateAppEventArgs>().Publish(new UpdateAppContext());
    }

    public void Subscribe(Action<UpdateAppContext> handler)
    {
        _eventAggregator.GetEvent<UpdateAppEventArgs>().Subscribe(handler);
    }
}

public class UpdateAppContext : EventArgs
{
    
}

public class UpdateAppEventArgs : PubSubEvent<UpdateAppContext>
{
}