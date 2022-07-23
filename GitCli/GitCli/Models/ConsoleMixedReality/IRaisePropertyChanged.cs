using System.ComponentModel;

namespace GitCli.Models.ConsoleMixedReality;

public interface IRaisePropertyChanged : INotifyPropertyChanged
{
	void RaisePropertyChanged(string propertyName, object? value);
}