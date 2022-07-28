#nullable enable
using System;
using System.Collections.Generic;
using GitCli.Models;

namespace T1.ConsoleUiMixedReality.ModelViewViewmodel;

public class NotifyObject<TValue> : INotifyObject<TValue>
{
	private TValue? _value;

	public TValue? Value
	{
		get => _value;
		set
		{
			if (_value == null && value == null)
			{
				return;
			}
			if (_value.IsEquals(value))
			{
				return;
			}
			InternalUpdateValue(value);
		}
	}

	private void InternalUpdateValue(TValue? value)
	{
		_value = value;
		OnNotify!.RaiseUpdateValue(value);
	}

	public event EventHandler<NotifyEventArgs<TValue>>? OnNotify;
}