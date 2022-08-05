using System.ComponentModel;

namespace CodeSnippeter.Models;

public interface INotifyObject : INotifyPropertyChanged
{
	void RaisePropertyChanged(string propertyName, PropertyChangedEventArgs eventArgs);
}