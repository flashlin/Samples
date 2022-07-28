using System.Collections.Generic;
using System.Linq;

namespace T1.ConsoleUiMixedReality;

public class NotifyEventArgs<T>
{
	public ChangeStatus Status { get; set; }
	public List<T> Items { get; set; } = new();
	public List<T> LastItems { get; set; } = Enumerable.Empty<T>().ToList();
}