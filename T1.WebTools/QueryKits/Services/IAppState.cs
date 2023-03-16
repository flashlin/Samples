namespace QueryKits.Services;

public interface IAppState
{
    bool IsLoading { get; set; }

    void Publish(Action<AppState> changeFn);
    void Subscribe(Action<UpdateAppContext> handler);
}