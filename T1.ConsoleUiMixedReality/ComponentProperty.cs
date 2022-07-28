using System.Collections.Generic;

namespace T1.ConsoleUiMixedReality;

public class ComponentProperty<TValue, TOwner>
	where TOwner : IRaisePropertyChanged
{
	private TValue? _value;
	private readonly TOwner _owner;
	private readonly string _propertyName;

	public ComponentProperty(TOwner owner, string propertyName)
	{
		_propertyName = propertyName;
		_owner = owner;
	}

	public TValue? Value
	{
		get => _value;
		set
		{
			if (EqualityComparer<TValue>.Default.Equals(_value, value))
			{
				return;
			}
			_value = value;
			_owner.RaisePropertyChanged(_propertyName, value);
		}
	}
}