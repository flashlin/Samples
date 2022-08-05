using System.Data;
using WCodeSnippetX.Models;
using WCodeSnippetX.ViewComponents;

namespace WCodeSnippetX
{
	public partial class FormMain : Form
	{
		readonly List<CodeSnippet> _table = new();
		List<CodeSnippet> _result = Enumerable.Empty<CodeSnippet>().ToList();
		readonly BindingSource _bindingSource = new();
		readonly DataGridView _dataGridView = new();
		private int _selectedRow = 0;

		public FormMain()
		{
			InitializeComponent();
			Init();
		}

		void Init()
		{
			_table.Add(new CodeSnippet { Id = 1, Content = "Datagridview and richtextbox for bold substring in C#" });
			_table.Add(new CodeSnippet { Id = 2, Content = "Sample htextbox for bold substring in C#" });
			_table.Add(new CodeSnippet { Id = 3, Content = "public class { \r\n public string Name; }" });
			_result = _table;


			//_bindingSource.Add(new CodeSnippet { Id = 1, Content = "Datagridview and richtextbox for bold substring in C#" });
			//_bindingSource.Add(new CodeSnippet { Id = 2, Content = "Sample htextbox for bold substring in C#" });
			//_bindingSource.Add(new CodeSnippet { Id = 3, Content = "public class { \r\n public string Name; }" });
			_bindingSource.DataSource = _result;

			_dataGridView.Columns.Add(new DataGridViewTextBoxColumn()
			{
				Name = "index",
				DataPropertyName = "Id",
				ReadOnly = true,
				Width = 3 * 12,
			});
			_dataGridView.Columns.Add(new DataGridViewRichTextBoxColumn()
			{
				Name = "Code Content",
				DataPropertyName = "content",
				ReadOnly = true,
				Width = 30 * 12,
			});

			_dataGridView.ReadOnly = true;
			_dataGridView.AutoGenerateColumns = false;
			_dataGridView.DataSource = _bindingSource;
			_dataGridView.KeyDown += OnDataGridViewKeyDown;
			ResizeDataGridView();
			this.Controls.Add(_dataGridView);

			textBoxSearch.TextChanged += (sender, args) =>
			{
				_result = textBoxSearch.Text == string.Empty ?
					_table : 
					_table.Where(x => x.Content.Contains(textBoxSearch.Text))
						.ToList();
				_bindingSource.DataSource = _result;
			};
		}

		private void OnDataGridViewKeyDown(object? sender, KeyEventArgs e)
		{
			if (e.KeyCode == Keys.Enter && _dataGridView.SelectedRows.Count > 0)
			{
				var item = _result[_selectedRow].Content;
				MessageBox.Show(item);
				return;
			}
		}

		private void ResizeDataGridView()
		{
			//foreach (var vBar in _dataGridView.Controls.OfType<HScrollBar>())
			//{
			//	vBar.Enabled = false;
			//}
			_dataGridView.SelectionMode = DataGridViewSelectionMode.FullRowSelect;
			_dataGridView.ScrollBars = ScrollBars.Vertical;
			_dataGridView.RowTemplate.Height = 12 * 10;
			_dataGridView.Width = ClientSize.Width - 3;
			_dataGridView.Height = ClientSize.Height - buttonSearch.Height - 6;
		}

		private void SetDataGridViewSelected(int idx)
		{
			_dataGridView.CurrentCell = _dataGridView.Rows[idx].Cells[0];
		}

		private void FormMain_ResizeEnd(object sender, EventArgs e)
		{
			ResizeDataGridView();
		}

		private void textBoxSearch_KeyDown(object sender, KeyEventArgs e)
		{
			HandleKeyDown(e);
		}

		private void HandleKeyDown(KeyEventArgs e)
		{
			if (e.KeyCode == Keys.Down)
			{
				if (_selectedRow + 1 < _dataGridView.RowCount)
				{
					_selectedRow++;
				}

				SetDataGridViewSelected(_selectedRow);
				return;
			}

			if (e.KeyCode == Keys.Up)
			{
				_selectedRow = Math.Max(0, _selectedRow - 1);
				SetDataGridViewSelected(_selectedRow);
				return;
			}

			if (e.KeyCode == Keys.Enter && _dataGridView.SelectedRows.Count > 0)
			{
				var item = _result[_selectedRow].Content;
				MessageBox.Show(item);
				return;
			}
		}
	}
}