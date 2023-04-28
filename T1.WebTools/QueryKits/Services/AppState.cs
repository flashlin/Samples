using QueryKits.Extensions;

namespace QueryKits.Services;

public class AppState : IAppState
{
    //private readonly IEventAggregator _eventAggregator;

    // public AppState(IEventAggregator eventAggregator)
    // {
    //     _eventAggregator = eventAggregator;
    // }

    public bool IsLoading { get; set; }

    public void Publish(Action<AppState> changeFn)
    {
        changeFn(this);
        //_eventAggregator.GetEvent<PubSubEvent<UpdateAppReqEvent>>().Publish(new UpdateAppReqEvent());

        this.Publish(new UpdateAppReqEvent().AsTask());
    }
    
    public void PublishEvent<T>(T eventArgs)
        where T: EventArgs, new()
    {
        this.Publish(new T().AsTask());
        //_eventAggregator.GetEvent<PubSubEvent<T>>().Publish(eventArgs);
    }


    public void SubscribeEvent<T>(Func<T, Task> handler)
        where T: EventArgs
    {
        this.Subscribe<T>(async (eventTask) =>
        {
            var args = await eventTask;
            await handler(args);
        });
        //_eventAggregator.GetEvent<PubSubEvent<T>>().Subscribe(MyHandle);
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

public class RefreshTableReqEvent : EventArgs
{
    
}

// public class UpdateAppEventArgs : PubSubEvent<UpdateAppRequest>
// {
// }