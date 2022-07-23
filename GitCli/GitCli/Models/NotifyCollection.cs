using GitCli.Models.ConsoleMixedReality;
using T1.Standard.Collections.Generics;

namespace GitCli.Models;

public interface INotifyCollection<T>
{
	event EventHandler<NotifyEventArgs<T>> OnNotify;
}

public class NotifyCollection<T> : INotifyCollection<T>
{
	private readonly object _locker = new object();
	private ConcurrentOnlyAddList<T> _items = new();
	private ConcurrentOnlyAddList<T> _addingItems = new();
	private ConcurrentOnlyAddList<T> _removingItems = new();
	private ConcurrentOnlyAddList<T> _updatingItems = new();

	public T this[int index]
	{
		get
		{
			lock (_locker)
			{
				return _items[index];
			}
		}
	}

	public event EventHandler<NotifyEventArgs<T>> OnNotify = null!;

	public List<T> ToList()
	{
		lock (_locker)
		{
			return _items.ToList();
		}
	}

	public void Updating(T item)
	{
		_updatingItems.Add(item);
	}

	public void Adding(T item)
	{
		_addingItems.Add(item);
	}

	public void Removing(T item)
	{
		_removingItems.Add(item);
	}

	public void Notify()
	{
		ConcurrentOnlyAddList<T> addList;
		ConcurrentOnlyAddList<T> removeList;
		ConcurrentOnlyAddList<T> updateList;
		lock (_locker)
		{
			addList = _addingItems;
			removeList = _removingItems;
			updateList = _updatingItems;
			_items = CloneDataList();
			_addingItems = new ConcurrentOnlyAddList<T>();
			_removingItems = new ConcurrentOnlyAddList<T>();
			_updatingItems = new ConcurrentOnlyAddList<T>();
		}

		if (removeList.Count > 0)
		{
			OnNotify?.Invoke(this, new NotifyEventArgs<T>
			{
				Items = removeList,
				Status = ChangeStatus.Removed
			});
		}

		if (addList.Count > 0)
		{
			OnNotify?.Invoke(this, new NotifyEventArgs<T>
			{
				Items = addList,
				Status = ChangeStatus.Added
			});
		}

		if (updateList.Count > 0)
		{
			OnNotify?.Invoke(this, new NotifyEventArgs<T>
			{
				Items = updateList,
				Status = ChangeStatus.Updated
			});
		}
	}

	private ConcurrentOnlyAddList<T> CloneDataList()
	{
		var newDataList = new ConcurrentOnlyAddList<T>();
		foreach (var item in _items)
		{
			if (!_removingItems.Contains(item))
			{
				newDataList.Add(item);
			}
		}

		foreach (var item in _addingItems)
		{
			newDataList.Add(item);
		}

		return newDataList;
	}
}