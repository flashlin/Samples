using System;
using System.Collections.Generic;
using GitCli.Models;

namespace T1.ConsoleUiMixedReality.ModelViewViewmodel;

public class NotifyProperty<TValue, TOwner> : INotifyObject<TValue>
	where TOwner : IRaisePropertyChanged
{
	private TValue? _value;
	private readonly TOwner _owner;
	private readonly string _propertyName;

	public NotifyProperty(TOwner owner, string propertyName)
	{
		_propertyName = propertyName;
		_owner = owner;
	}

	public TValue? Value
	{
		get => _value;
		set
		{
			if (_value.IsEquals(value))
			{
				return;
			}
			_value = value;
			OnNotify!.RaiseUpdateValue(value);
			_owner.RaisePropertyChanged(_propertyName, value);
		}
	}

	public event EventHandler<NotifyEventArgs<TValue>>? OnNotify;
}