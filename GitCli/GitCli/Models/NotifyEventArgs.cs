using GitCli.Models.ConsoleMixedReality;
using T1.Standard.Collections.Generics;

namespace GitCli.Models;

public class NotifyEventArgs<T>
{
	public ChangeStatus Status { get; set; }
	public ConcurrentOnlyAddList<T> Items { get; init; }
}