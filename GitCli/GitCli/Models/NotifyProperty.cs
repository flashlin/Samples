namespace GitCli.Models;

public class NotifyProperty<T>
{
	private readonly IObjectNotifyPropertyChanged _owner;

	public NotifyProperty(IObjectNotifyPropertyChanged owner, string name, T? initialValue)
	{
		_owner = owner;
		Name = name;
		Value = initialValue;
	}

	public string Name { get; }
	public T? Value { get; private set; }

	public void SetValue(T newValue)
	{
		if (Value != null && !Value.Equals(newValue))
		{
			Value = newValue;
			_owner.RaisePropertyChanged(this.Name);
			return;
		}

		if (newValue != null && !newValue.Equals(Value))
		{
			Value = newValue;
			_owner.RaisePropertyChanged(this.Name);
		}
	}
}