using System.ComponentModel;

namespace GitCli.Models;

public interface IObjectNotifyPropertyChanged : INotifyPropertyChanged
{
	void RaisePropertyChanged(string propertyName);
}