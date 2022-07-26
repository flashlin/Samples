namespace GitCli.Models.ConsoleMixedReality;

public class ConsoleBuffer
{
	private Character[,] _buffer = new Character[0, 0];

	public Character this[Position pos]
	{
		get
		{
			return _buffer[pos.X, pos.Y];
		}
	}

	public Size Size => new Size
	{
		Width = _buffer.GetLength(0),
		Height = _buffer.GetLength(1),
	};

	public void Initialize(Size size)
	{
		_buffer = new Character[size.Width, size.Height];
	}

	public void Clear()
	{
		for (int i = 0; i < _buffer.GetLength(0); i++)
			for (int j = 0; j < _buffer.GetLength(1); j++)
				_buffer[i, j] = Character.Empty;
	}

	public bool Update(Position position, Character newCell)
	{
		ref var cell = ref _buffer[position.X, position.Y];
		//if (newCell.IsEmpty && cell.IsEmpty)
		//{
		//	return false;
		//}
		var characterChanged = cell != newCell;
		cell = newCell;
		return characterChanged;
	}
}