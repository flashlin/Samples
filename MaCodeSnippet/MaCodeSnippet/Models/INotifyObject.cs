using System.ComponentModel;

namespace MaCodeSnippet.Models;

public interface INotifyObject : INotifyPropertyChanged
{
	void RaisePropertyChanged(string propertyName, PropertyChangedEventArgs eventArgs);
}