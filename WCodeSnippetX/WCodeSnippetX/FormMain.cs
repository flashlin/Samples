using System.ComponentModel;
using System.Data;
using WCodeSnippetX.Models;
using WCodeSnippetX.ViewComponents;
using System.Windows.Forms;

//https://docs.microsoft.com/zh-tw/windows/win32/inputdev/virtual-key-codes
namespace WCodeSnippetX
{
	public partial class FormMain : Form
	{
		List<CodeSnippetEntity> _result = Enumerable.Empty<CodeSnippetEntity>().ToList();
		readonly BindingSource _bindingSource = new();
		readonly DataGridView _dataGridView = new();
		private int _selectedRow = 0;
		private readonly GlobalKeyboardHook _globalKeyboardHook = new();

		public FormMain(ICodeSnippetRepo repo, FormEditCode formEditCode)
		{
			_formEditCode = formEditCode;
			_repo = repo;
			InitializeComponent();
			_globalKeyboardHook.KeyboardPressed += OnKeyPressed;
			this.Closing += OnClosing;
			this.KeyDown += OnKeyDown;
			Init();
		}

		private void OnKeyDown(object? sender, KeyEventArgs e)
		{
			if (e.KeyCode == Keys.Escape)
			{
				e.Handled = true;
			}
		}

		private void OnClosing(object? sender, CancelEventArgs e)
		{
			_globalKeyboardHook.Dispose();
		}

		bool _leftAlt = false;
		private ICodeSnippetRepo _repo;
		private FormEditCode _formEditCode;

		private void OnKeyPressed(object? sender, GlobalKeyboardHookEventArgs e)
		{
			if (e.KeyboardState != GlobalKeyboardHook.KeyboardState.KeyDown &&
				 e.KeyboardState != GlobalKeyboardHook.KeyboardState.SysKeyDown)
			{
				_leftAlt = false;
				return;
			}

			if (e.KeyboardData.VirtualCode == VirtualCode.VkLeftAlt)
			{
				_leftAlt = true;
				e.Handled = true;
				return;
			}

			if (e.KeyboardData.VirtualCode == VirtualCode.Vk0 && _leftAlt)
			{
				BringMeToFront();
				e.Handled = true;
				return;
			}
		}

		void BringMeToFront()
		{
			this.WindowState = FormWindowState.Minimized;
			this.Show();
			this.WindowState = FormWindowState.Normal;
		}

		void Init()
		{
			_result = _repo.QueryCode(textBoxSearch.Text).ToList();

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
			SetupDataGridViewMenu();
			ResizeDataGridView();
			this.Controls.Add(_dataGridView);

			textBoxSearch.TextChanged += (sender, args) =>
			{
				RefreshSearchCode();
			};
		}

		private void DataGridViewOnMouseClick(object? sender, MouseEventArgs e)
		{
			if (e.Button != MouseButtons.Right)
			{
				return;
			}

			//var p = _dataGridView.PointToClient(new Point(e.X, e.Y));
			var currentMouseOverRow = _dataGridView.HitTest(e.X, e.Y).RowIndex;
			if (currentMouseOverRow >= 0)
			{
				_selectedRow = currentMouseOverRow;
				SetDataGridViewSelected(_selectedRow);
				//m.Show(_dataGridView, new Point(e.X, e.Y));
			}
		}

		private void RefreshSearchCode()
		{
			_result = _repo.QueryCode(textBoxSearch.Text).ToList();
			_bindingSource.DataSource = _result;
		}

		private ContextMenuStrip SetupDataGridViewMenu()
		{
			var m = new ContextMenuStrip();
			var toolStripMenuEdit = new ToolStripMenuItem("Edit");
			toolStripMenuEdit.Click += (o, args) =>
			{
				if (_dataGridView.SelectedRows.Count == 0)
				{
					return;
				}
				_selectedRow = _dataGridView.SelectedRows[0].Index;
				_formEditCode.SetValue(_result[_selectedRow]);
				_formEditCode.ShowDialog();
			};
			m.Items.Add(toolStripMenuEdit);

			var toolStripMenuAdd = new ToolStripMenuItem("Add");
			toolStripMenuAdd.Click += (o, args) =>
			{
				_formEditCode.SetValue(new CodeSnippetEntity());
				_formEditCode.ShowDialog();
				RefreshSearchCode();
			};
			m.Items.Add(toolStripMenuAdd);
			_dataGridView.ContextMenuStrip = m;
			return m;
		}

		private void OnDataGridViewKeyDown(object? sender, KeyEventArgs e)
		{
			if (e.KeyCode == Keys.Enter && _dataGridView.SelectedRows.Count > 0)
			{
				CopySelectedItem();
				return;
			}

			if (e.KeyCode == Keys.Escape)
			{
				HideMe();
				return;
			}
		}

		private void HideMe()
		{
			textBoxSearch.Text = "";
			this.WindowState = FormWindowState.Minimized;
		}

		private void CopySelectedItem()
		{
			Clipboard.SetText(_result[_selectedRow].Content);
			textBoxSearch.Text = "";
			HideMe();
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
				CopySelectedItem();
				return;
			}

			if (e.KeyCode == Keys.Escape)
			{
				HideMe();
				return;
			}
		}

		private void FormMain_Activated(object sender, EventArgs e)
		{
			textBoxSearch.Focus();
		}
	}
}