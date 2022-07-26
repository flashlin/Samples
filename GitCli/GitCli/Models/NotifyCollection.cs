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
	private List<T> _items = new();
	private List<T> _addingItems = new();
	private List<T> _removingItems = new();
	private List<T> _updatingItems = new();

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

	public int Count => _items.Count;

	public event EventHandler<NotifyEventArgs<T>> OnNotify = null!;

	public List<T> ToList()
	{
		lock (_locker)
		{
			return _items.ToList();
		}
	}

	public void Init(IEnumerable<T> items)
	{
		lock (_locker)
		{
			_items.Clear();
		}
		foreach (var item in items)
		{
			Adding(item);
		}
		Notify();
	}

	public void Clear()
	{
		var oldItems = _items.ToList();
		OnNotify?.Invoke(this, new NotifyEventArgs<T>
		{
			Items = oldItems,
			Status = ChangeStatus.Removed,
			LastItems = _items.ToList(),
		});
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
		List<T> addList;
		List<T> removeList;
		List<T> updateList;
		lock (_locker)
		{
			addList = _addingItems;
			removeList = _removingItems;
			updateList = _updatingItems;
			_items = CloneDataList();
			_addingItems = new List<T>();
			_removingItems = new List<T>();
			_updatingItems = new List<T>();
		}

		var lastItems = _items.ToList();

		if (removeList.Count > 0)
		{
			OnNotify?.Invoke(this, new NotifyEventArgs<T>
			{
				Items = removeList,
				Status = ChangeStatus.Removed,
				LastItems = lastItems,
			});
		}

		if (addList.Count > 0)
		{
			OnNotify?.Invoke(this, new NotifyEventArgs<T>
			{
				Items = addList,
				Status = ChangeStatus.Added,
				LastItems = lastItems,
			});
		}

		if (updateList.Count > 0)
		{
			OnNotify?.Invoke(this, new NotifyEventArgs<T>
			{
				Items = updateList,
				Status = ChangeStatus.Updated,
				LastItems = lastItems,
			});
		}
	}

	private List<T> CloneDataList()
	{
		var newDataList = new List<T>();
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