using System.Collections.Generic;
using System.Linq;

namespace GitCli.Models;

public class NotifyEventArgs<T>
{
	public ChangeStatus Status { get; set; }
	public List<T> Items { get; init; } = new();
	public List<T> LastItems { get; set; } = Enumerable.Empty<T>().ToList();
}