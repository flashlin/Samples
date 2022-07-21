using System.ComponentModel;

namespace GitCli.Models;

public class BaseNotifyObject : IObjectNotifyPropertyChanged
{
	public event PropertyChangedEventHandler? PropertyChanged;

	protected virtual void OnPropertyChanged(PropertyChangedEventArgs args)
	{
		PropertyChanged?.Invoke(this, args);
	}

	public void RaisePropertyChanged(string propertyName)
	{
		OnPropertyChanged(new PropertyChangedEventArgs(propertyName));
	}
}