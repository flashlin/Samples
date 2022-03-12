using System;
using System.Collections.Generic;
using System.Text;

namespace PreviewLibrary.Helpers
{
	public interface ITwoWayEnumerator<T> : IEnumerator<T>
	{
		bool MovePrevious();
	}

	public class TwoWayEnumerator<T> : ITwoWayEnumerator<T>
	{
		private IEnumerator<T> _enumerator;
		private List<T> _buffer;
		private int _index;

		public TwoWayEnumerator(IEnumerator<T> enumerator)
		{
			if (enumerator == null)
				throw new ArgumentNullException("enumerator");

			_enumerator = enumerator;
			_buffer = new List<T>();
			_index = -1;
		}

		public bool MovePrevious()
		{
			if (_index <= 0)
			{
				return false;
			}

			--_index;
			return true;
		}

		public bool MoveNext()
		{
			if (_index < _buffer.Count - 1)
			{
				++_index;
				return true;
			}

			if (_enumerator.MoveNext())
			{
				_buffer.Add(_enumerator.Current);
				++_index;
				return true;
			}

			_index = _buffer.Count;
			return false;
		}

		public T Current
		{
			get
			{
				if (_index == _buffer.Count)
				{
					return default(T);
				}

				if (_index < 0 || _index >= _buffer.Count)
					throw new InvalidOperationException();

				return _buffer[_index];
			}
		}

		public void Reset()
		{
			_enumerator.Reset();
			_buffer.Clear();
			_index = -1;
		}

		public void Dispose()
		{
			_enumerator.Dispose();
		}

		object System.Collections.IEnumerator.Current
		{
			get { return Current; }
		}
	}

	public static class TwoWayEnumeratorHelper
	{
		public static ITwoWayEnumerator<T> GetTwoWayEnumerator<T>(this IEnumerable<T> source)
		{
			if (source == null)
				throw new ArgumentNullException("source");

			return new TwoWayEnumerator<T>(source.GetEnumerator());
		}

		public static ITwoWayEnumerator<T> GetTwoWayEnumerator<T>(this IEnumerator<T> source)
		{
			if (source == null)
				throw new ArgumentNullException("source");
			return new TwoWayEnumerator<T>(source);
		}
	}
}
