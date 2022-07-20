namespace GitCli.Models;

public class Bind<T>
{
	public T? Value { get; private set; }

	public List<Action<T>> SetupList = new List<Action<T>>();

	public void Setup(Action<T> fn)
	{
		if (Value == null)
		{
			SetupList.Add(fn);
			return;
		}
		fn(Value);
	}

	public void SetValue(T value)
	{
		Value = value;
		foreach (var setup in SetupList)
		{
			setup(value);
		}
	}
}