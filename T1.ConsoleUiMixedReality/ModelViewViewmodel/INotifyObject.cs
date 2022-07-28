using System;

namespace T1.ConsoleUiMixedReality.ModelViewViewmodel;

public interface INotifyObject<T>
{
	event EventHandler<NotifyEventArgs<T>>? OnNotify;
}