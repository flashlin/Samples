using System.ComponentModel;

namespace MaCodeSnippet.Models;

public class NotifyObject<T> : INotifyPropertyChanged
{
	private T _value;
	private readonly string _propertyName;
	private readonly INotifyObject _owner;

	public NotifyObject(INotifyObject owner, string propertyName, T value)
	{
		_owner = owner;
		_propertyName = propertyName;
		_value = value;
	}

	public event PropertyChangedEventHandler? PropertyChanged;

	public T Value
	{
		get => _value;
		set
		{
			if (_value.IsEquals(value))
			{
				return;
			}
			_value = value;
			OnPropertyChanged();
		}
	}

	protected virtual void OnPropertyChanged()
	{
		var eventArgs = new PropertyChangedEventArgs(_propertyName);
		PropertyChanged?.Invoke(this, eventArgs);
		_owner?.RaisePropertyChanged(_propertyName, eventArgs);
	}
}