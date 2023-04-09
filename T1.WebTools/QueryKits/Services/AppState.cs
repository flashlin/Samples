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
        _eventAggregator.GetEvent<PubSubEvent<UpdateAppReqEvent>>().Publish(new UpdateAppReqEvent());
    }
    
    public void PublishEvent<T>(T eventArgs)
        where T: EventArgs
    {
        _eventAggregator.GetEvent<PubSubEvent<T>>().Publish(eventArgs);
    }


    public void SubscribeEvent<T>(Action<T> handler)
        where T: EventArgs
    {
        _eventAggregator.GetEvent<PubSubEvent<T>>().Subscribe(handler);
    }
}

public class UpdateAppReqEvent : EventArgs
{
}

public class MergeTableReqEvent : EventArgs
{
    public string LeftTableName { get; set; }
    public string RightTableName { get; set; }
}

// public class UpdateAppEventArgs : PubSubEvent<UpdateAppRequest>
// {
// }