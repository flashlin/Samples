using System.ComponentModel;

namespace T1.ConsoleUiMixedReality;

public interface IRaisePropertyChanged : INotifyPropertyChanged
{
	void RaisePropertyChanged(string propertyName, object? value);
}