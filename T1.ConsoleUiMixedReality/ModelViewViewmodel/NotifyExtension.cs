using System;
using System.Collections.Generic;

namespace T1.ConsoleUiMixedReality.ModelViewViewmodel;

public static class NotifyExtension
{
	public static void RaiseUpdateValue<TValue>(this EventHandler<NotifyEventArgs<TValue>>? handler, TValue value)
	{
		if (handler == null)
		{
			return;
		}
		var notifyValues = new List<TValue>()
		{
			value
		};
		handler.Invoke(handler, new NotifyEventArgs<TValue>()
		{
			LastItems = notifyValues,
			Items = notifyValues,
			Status = ChangeStatus.Updated,
		});
	}
}